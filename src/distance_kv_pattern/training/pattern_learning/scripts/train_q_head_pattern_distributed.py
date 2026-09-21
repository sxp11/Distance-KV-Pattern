#!/usr/bin/env python3
"""Train Q-head distance gates with single-node PyTorch DDP.

The frozen language model is replicated once per GPU. Only the Hard Concrete gate
module is wrapped in DistributedDataParallel. A global optimizer step still
covers the complete manifest; ranks process disjoint contexts and DDP reduces
their globally weighted task gradients exactly once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer


METHOD_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from distance_kv_pattern import (  # noqa: E402
    BlockLayout,
    IndependentQueryDatasetCollection,
    InstructMaterializer,
    LogProgressBar,
    backward_distributed_q_head_branches,
    build_context_shards,
    context_occurrences_for_step,
    exact_topk_mask,
    formal_manifest_collection_id,
    gradient_competition_statistics,
    initialize_q_head_gates_from_retrieval,
    l0_candidate_first_step_statistics,
    manifest_group_index_for_step,
    manifest_distance_coverage,
    normalized_step_branch_weights,
    planned_gate_indices,
    polarization_candidate_continuation_statistics,
    q_head_training_gate_sample_seed,
    scheduled_l0_coefficient,
    st_topk_keep_ratio_for_context,
    validate_continuation_checkpoint,
)
from distance_kv_pattern.core.randomness import set_reproducibility  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="Formal manifest; repeat to train on multiple condition/round manifests.",
    )
    parser.add_argument(
        "--manifest-group",
        type=Path,
        action="append",
        nargs="+",
        default=None,
        help=(
            "One optimizer-step manifest group; repeat this option to define a "
            "deterministic round-robin schedule. Cannot be combined with --manifest."
        ),
    )
    parser.add_argument(
        "--retrieval-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--retrieval-metric", default="retrieval_score_at_k")
    parser.add_argument("--initial-keep-floor", type=float, default=0.95)
    parser.add_argument("--initial-keep-gain", type=float, default=0.04)
    parser.add_argument(
        "--initial-keep-mapping",
        choices=("linear", "logistic"),
        default="linear",
    )
    parser.add_argument("--initial-keep-target-mean", type=float, default=0.80)
    parser.add_argument("--initial-keep-score-slope", type=float, default=4.0)
    parser.add_argument("--initial-keep-probability-min", type=float, default=1e-3)
    parser.add_argument("--initial-keep-probability-max", type=float, default=1.0 - 1e-3)

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l0-lambda", type=float, default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--optimizer-steps-this-run", type=int, default=None)
    parser.add_argument("--l0-start-step", type=int, default=0)
    parser.add_argument("--l0-ramp-steps", type=int, default=0)
    parser.add_argument("--polarization-lambda", type=float, default=0.0)
    parser.add_argument("--polarization-start-step", type=int, default=0)
    parser.add_argument("--polarization-ramp-steps", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hard-mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gate-forward-mode",
        choices=("relaxed_hard_concrete", "st_topk"),
        default="relaxed_hard_concrete",
        help="Training forward gate; ST Top-K is binary forward/relaxed backward.",
    )
    parser.add_argument(
        "--st-topk-keep-ratios",
        type=float,
        nargs="+",
        default=(),
        help=(
            "Fixed global Q-head gate budgets used by st_topk. Contexts are "
            "assigned evenly and their assignment rotates between steps."
        ),
    )
    parser.add_argument(
        "--l0-candidate-audit-only",
        action="store_true",
        help=(
            "Collect one fixed-parameter task gradient without taking an optimizer "
            "step, then analytically audit every --l0-candidate-audit-lambdas value."
        ),
    )
    parser.add_argument(
        "--l0-candidate-audit-lambdas",
        type=float,
        nargs="+",
        default=(),
        help="L0 coefficients evaluated from the shared task gradient in audit mode.",
    )
    parser.add_argument(
        "--polarization-candidate-audit-only",
        action="store_true",
        help=(
            "Import a continuation checkpoint, collect one fixed-parameter task "
            "gradient without an optimizer step, and simulate candidate "
            "interior-mass penalties using the imported AdamW state."
        ),
    )
    parser.add_argument(
        "--polarization-candidate-audit-lambdas",
        type=float,
        nargs="+",
        default=(),
        help="Interior-mass coefficients evaluated in continuation audit mode.",
    )
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument(
        "--shuffle-contexts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--checkpoint-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ddp-timeout-minutes", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continuation-checkpoint",
        type=Path,
        default=None,
        help=(
            "One-time source checkpoint imported only when output-dir/latest.pt "
            "does not exist. All protected training-signature fields must match."
        ),
    )
    parser.add_argument(
        "--continuation-expected-step",
        type=int,
        default=None,
        help="Required completed step in --continuation-checkpoint.",
    )
    parser.add_argument(
        "--allow-manifest-change-on-continuation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--milestone-steps", type=int, nargs="*", default=())

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-world-size", type=int, default=8)
    return parser.parse_args(argv)


def args_from_config(path: Path) -> argparse.Namespace:
    config = json.loads(path.read_text(encoding="utf-8"))
    argv: list[str] = []
    for name, value in config.items():
        if value is None:
            continue
        option = "--" + name.replace("_", "-")
        if name == "manifest_group":
            for group in value:
                argv.append(option)
                argv.extend(str(item) for item in group)
        elif isinstance(value, bool):
            argv.append(option if value else "--no-" + name.replace("_", "-"))
        elif isinstance(value, list) and value:
            argv.append(option)
            argv.extend(str(item) for item in value)
        elif isinstance(value, list):
            continue
        else:
            argv.extend((option, str(value)))
    return parse_args(argv)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def validate_args(args: argparse.Namespace) -> None:
    if args.manifest is not None and args.manifest_group is not None:
        raise ValueError("--manifest and --manifest-group cannot be combined")
    if args.l0_candidate_audit_only and args.polarization_candidate_audit_only:
        raise ValueError("L0 and polarization audit modes are mutually exclusive")
    if args.l0_candidate_audit_only and len(args.manifest_groups) != 1:
        raise ValueError("L0 candidate audit accepts exactly one manifest group")
    if not 0.0 < args.initial_keep_target_mean < 1.0:
        raise ValueError("initial-keep-target-mean must lie inside (0, 1)")
    if not math.isfinite(args.initial_keep_score_slope) or args.initial_keep_score_slope < 0:
        raise ValueError("initial-keep-score-slope must be finite and non-negative")
    if not 0.0 < args.initial_keep_probability_min < args.initial_keep_probability_max < 1.0:
        raise ValueError("initial keep probability bounds must satisfy 0 < min < max < 1")
    audit_lambdas = tuple(float(value) for value in args.l0_candidate_audit_lambdas)
    if any(not math.isfinite(value) or value < 0.0 for value in audit_lambdas):
        raise ValueError("L0 audit candidates must be finite and non-negative")
    if len(set(audit_lambdas)) != len(audit_lambdas):
        raise ValueError("L0 audit candidates must be unique")
    if args.l0_candidate_audit_only and not audit_lambdas:
        raise ValueError("audit-only mode requires --l0-candidate-audit-lambdas")
    if audit_lambdas and not args.l0_candidate_audit_only:
        raise ValueError("L0 audit candidates require --l0-candidate-audit-only")
    polarization_audit_lambdas = tuple(
        float(value) for value in args.polarization_candidate_audit_lambdas
    )
    if any(
        not math.isfinite(value) or value < 0.0
        for value in polarization_audit_lambdas
    ):
        raise ValueError("polarization audit candidates must be finite and non-negative")
    if len(set(polarization_audit_lambdas)) != len(polarization_audit_lambdas):
        raise ValueError("polarization audit candidates must be unique")
    if args.polarization_candidate_audit_only and not polarization_audit_lambdas:
        raise ValueError(
            "polarization audit-only mode requires candidate lambdas"
        )
    if polarization_audit_lambdas and not args.polarization_candidate_audit_only:
        raise ValueError(
            "polarization audit candidates require audit-only mode"
        )
    st_topk_keep_ratios = tuple(float(value) for value in args.st_topk_keep_ratios)
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in st_topk_keep_ratios
    ):
        raise ValueError("ST Top-K keep ratios must lie inside (0, 1]")
    if len(set(st_topk_keep_ratios)) != len(st_topk_keep_ratios):
        raise ValueError("ST Top-K keep ratios must be unique")
    if args.gate_forward_mode == "st_topk" and not st_topk_keep_ratios:
        raise ValueError("st_topk mode requires --st-topk-keep-ratios")
    if args.gate_forward_mode != "st_topk" and st_topk_keep_ratios:
        raise ValueError("ST Top-K keep ratios require --gate-forward-mode st_topk")
    if (
        args.gate_forward_mode == "st_topk"
        and (args.l0_candidate_audit_only or args.polarization_candidate_audit_only)
    ):
        raise ValueError("candidate audit modes require relaxed Hard Concrete forward")
    if (
        args.continuation_expected_step is not None
        and args.continuation_checkpoint is None
    ):
        raise ValueError(
            "--continuation-expected-step requires --continuation-checkpoint"
        )
    if (
        args.allow_manifest_change_on_continuation
        and args.continuation_checkpoint is None
    ):
        raise ValueError(
            "--allow-manifest-change-on-continuation requires "
            "--continuation-checkpoint"
        )
    if (
        args.continuation_expected_step is not None
        and args.continuation_expected_step <= 0
    ):
        raise ValueError("--continuation-expected-step must be positive")
    if args.l0_candidate_audit_only and args.continuation_checkpoint is not None:
        raise ValueError("audit-only mode cannot import a continuation checkpoint")
    if (
        args.polarization_candidate_audit_only
        and args.continuation_checkpoint is None
    ):
        raise ValueError(
            "polarization audit-only mode requires --continuation-checkpoint"
        )
    if (
        args.continuation_expected_step is not None
        and args.max_optimizer_steps is not None
        and args.continuation_expected_step >= args.max_optimizer_steps
    ):
        raise ValueError(
            "continuation expected step must be below max-optimizer-steps"
        )
    if (
        args.continuation_checkpoint is not None
        and not args.continuation_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"continuation checkpoint does not exist: {args.continuation_checkpoint}"
        )
    if args.dry_run:
        if args.dry_run_world_size <= 0:
            raise ValueError("dry-run-world-size must be positive")
        return
    if (
        args.learning_rate is None
        or args.l0_lambda is None
        or args.max_optimizer_steps is None
        or args.output_dir is None
    ):
        raise ValueError(
            "real training requires --learning-rate, --l0-lambda, "
            "--max-optimizer-steps, and --output-dir"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.l0_lambda) or args.l0_lambda < 0:
        raise ValueError("l0-lambda must be finite and non-negative")
    if not math.isfinite(args.polarization_lambda) or args.polarization_lambda < 0:
        raise ValueError("polarization-lambda must be finite and non-negative")
    if args.max_optimizer_steps <= 0:
        raise ValueError("max-optimizer-steps must be positive")
    if args.optimizer_steps_this_run is not None and args.optimizer_steps_this_run <= 0:
        raise ValueError("optimizer-steps-this-run must be positive")
    if args.l0_start_step < 0 or args.l0_ramp_steps < 0:
        raise ValueError("L0 schedule steps cannot be negative")
    if args.polarization_start_step < 0 or args.polarization_ramp_steps < 0:
        raise ValueError("polarization schedule steps cannot be negative")
    if not math.isfinite(args.gradient_clip_norm) or args.gradient_clip_norm <= 0:
        raise ValueError("gradient-clip-norm must be finite and positive")
    if not 0 <= args.hard_mask_threshold <= 1:
        raise ValueError("hard-mask-threshold must lie inside [0, 1]")
    if args.ddp_timeout_minutes <= 0:
        raise ValueError("ddp-timeout-minutes must be positive")
    if args.dry_run_world_size <= 0:
        raise ValueError("dry-run-world-size must be positive")
    if args.l0_candidate_audit_only:
        if args.l0_lambda != 0.0:
            raise ValueError("audit-only mode requires --l0-lambda 0")
        if args.max_optimizer_steps != 1:
            raise ValueError("audit-only mode requires --max-optimizer-steps 1")
        if args.optimizer_steps_this_run not in (None, 1):
            raise ValueError(
                "audit-only mode permits only --optimizer-steps-this-run 1"
            )
        if args.resume:
            raise ValueError("audit-only mode requires --no-resume")
        if args.milestone_steps:
            raise ValueError("audit-only mode does not create milestone checkpoints")
    if args.polarization_candidate_audit_only:
        if args.polarization_lambda != 0.0:
            raise ValueError(
                "polarization audit-only mode requires --polarization-lambda 0"
            )
        if args.optimizer_steps_this_run not in (None, 1):
            raise ValueError(
                "polarization audit-only mode permits only one planned step"
            )
        if args.resume:
            raise ValueError(
                "polarization audit-only mode requires --no-resume"
            )
        if args.milestone_steps:
            raise ValueError(
                "polarization audit-only mode does not create milestones"
            )
        if (
            args.continuation_expected_step is not None
            and args.max_optimizer_steps != args.continuation_expected_step + 1
        ):
            raise ValueError(
                "polarization audit plan must end one step after its continuation"
            )
    milestones = tuple(sorted(set(args.milestone_steps)))
    if milestones and (milestones[0] <= 0 or milestones[-1] > args.max_optimizer_steps):
        raise ValueError("milestone steps must lie inside the training plan")


def init_distributed(timeout_minutes: int) -> tuple[int, int, int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "distributed training must be launched by torchrun; missing "
            + ", ".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL DDP requires CUDA")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=timeout_minutes),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def reduce_sum(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(value) for value in tensor.cpu().tolist()]


@torch.no_grad()
def replicated_parameter_max_difference(parameter: torch.Tensor) -> float:
    reference = parameter.detach().clone()
    dist.broadcast(reference, src=0)
    difference = parameter.detach().sub(reference).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return float(difference.item())


def training_signature(
    args: argparse.Namespace,
    *,
    manifest_id: str,
    world_size: int,
) -> dict[str, Any]:
    signature = {
        "schema_version": 3,
        "trainer": "official_pytorch_ddp_q_head",
        "model_name_or_path": str(args.model_name_or_path),
        "manifests": [str(path.resolve()) for path in args.manifests],
        "manifest_id": manifest_id,
        "retrieval_checkpoint": str(args.retrieval_checkpoint.resolve()),
        "retrieval_metric": args.retrieval_metric,
        "initial_keep_floor": args.initial_keep_floor,
        "initial_keep_gain": args.initial_keep_gain,
        "initial_keep_mapping": args.initial_keep_mapping,
        "initial_keep_target_mean": args.initial_keep_target_mean,
        "initial_keep_score_slope": args.initial_keep_score_slope,
        "initial_keep_probability_min": args.initial_keep_probability_min,
        "initial_keep_probability_max": args.initial_keep_probability_max,
        "world_size": world_size,
        "learning_rate": args.learning_rate,
        "l0_lambda": args.l0_lambda,
        "l0_candidate_audit_only": args.l0_candidate_audit_only,
        "l0_candidate_audit_lambdas": list(args.l0_candidate_audit_lambdas),
        "max_optimizer_steps": args.max_optimizer_steps,
        "l0_start_step": args.l0_start_step,
        "l0_ramp_steps": args.l0_ramp_steps,
        "polarization_lambda": args.polarization_lambda,
        "polarization_start_step": args.polarization_start_step,
        "polarization_ramp_steps": args.polarization_ramp_steps,
        "polarization_candidate_audit_only": (
            args.polarization_candidate_audit_only
        ),
        "polarization_candidate_audit_lambdas": list(
            args.polarization_candidate_audit_lambdas
        ),
        "gradient_clip_norm": args.gradient_clip_norm,
        "gate_forward_mode": args.gate_forward_mode,
        "st_topk_keep_ratios": list(args.st_topk_keep_ratios),
        "st_topk_assignment_policy": "rotating_context_round_robin",
        "master_seed": args.master_seed,
        "shuffle_contexts": args.shuffle_contexts,
        "checkpoint_layers": args.checkpoint_layers,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
    }
    if args.continuation_checkpoint is not None:
        signature["continuation_checkpoint"] = str(
            args.continuation_checkpoint.resolve()
        )
        signature["continuation_expected_step"] = (
            args.continuation_expected_step
        )
        signature["allow_manifest_change_on_continuation"] = (
            args.allow_manifest_change_on_continuation
        )
    if args.manifest_group is not None:
        signature["manifest_group_schedule"] = [
            [str(path.resolve()) for path in group]
            for group in args.manifest_groups
        ]
        signature["manifest_group_schedule_policy"] = "optimizer_step_modulo"
    return signature


def save_checkpoint(
    path: Path,
    *,
    signature: dict[str, Any],
    completed_optimizer_steps: int,
    log_alpha: torch.Tensor,
    keep_probability: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    hard_mask_threshold: float,
    continuation_provenance: dict[str, Any] | None = None,
    allow_overwrite: bool = True,
) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite milestone checkpoint: {path}")
    probability = keep_probability.detach().float().cpu()
    payload = {
        "schema_version": 3,
        "signature": signature,
        "completed_optimizer_steps": completed_optimizer_steps,
        "log_alpha": log_alpha.detach().float().cpu(),
        "keep_probability": probability,
        "hard_mask": probability.ge(hard_mask_threshold),
        "hard_mask_threshold": hard_mask_threshold,
        "optimizer_state": optimizer.state_dict(),
    }
    if continuation_provenance is not None:
        payload["continuation_provenance"] = continuation_provenance
    atomic_torch_save(payload, path)


def main(*, runner_factory: Callable[..., Any],
    runtime_validator: Callable[[], None],
    model_loader: Callable[..., Any] = AutoModelForCausalLM.from_pretrained,
    tokenizer_loader: Callable[..., Any] = AutoTokenizer.from_pretrained,
    layout_factory: Callable[[], BlockLayout] = BlockLayout,
    materializer_factory: Callable[..., Any] = InstructMaterializer,
    args: argparse.Namespace | None = None,
) -> None:
    args = parse_args() if args is None else args
    runtime_validator()
    if args.manifest_group is not None:
        args.manifest_groups = tuple(
            tuple(group) for group in args.manifest_group
        )
    else:
        args.manifest_groups = (
            tuple(args.manifest or ()),
        )
    args.manifests = tuple(
        path for group in args.manifest_groups for path in group
    )
    validate_args(args)

    layout = layout_factory()
    tokenizer = tokenizer_loader(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    materializer = materializer_factory(tokenizer, layout=layout)
    datasets = tuple(
        IndependentQueryDatasetCollection(
            group,
            tokenizer,
            layout=layout,
            materializer=materializer,
        )
        for group in args.manifest_groups
    )
    manifest_ids = tuple(
        formal_manifest_collection_id(
            tuple(child.plans for child in dataset.datasets)
        )
        for dataset in datasets
    )
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest groups must have unique collection identities")
    gate_index_groups_by_manifest_group = tuple(
        planned_gate_indices(dataset.plans) for dataset in datasets
    )
    coverage_by_manifest_group = tuple(
        manifest_distance_coverage(
            gate_index_groups,
            num_distances=layout.num_learnable_blocks,
        )
        for gate_index_groups in gate_index_groups_by_manifest_group
    )
    if any(
        int(coverage.min().item()) < 1
        for coverage in coverage_by_manifest_group
    ):
        raise ValueError(
            "every distributed optimizer-step manifest group requires complete "
            "distance coverage"
        )
    manifest_id = (
        manifest_ids[0]
        if len(manifest_ids) == 1
        else "schedule:" + "||".join(manifest_ids)
    )

    planned_world_size = args.dry_run_world_size if args.dry_run else int(
        os.environ.get("WORLD_SIZE", "0")
    )
    # Every group is a complete optimizer-step boundary. Multiple groups are
    # selected round-robin from the completed-step count, so checkpoint resume
    # requires no separate mutable schedule cursor.
    contexts_per_optimizer_step_by_manifest_group = tuple(
        len(dataset) for dataset in datasets
    )
    shards_by_manifest_group = tuple(
        build_context_shards(len(dataset), world_size=planned_world_size)
        for dataset in datasets
    )
    group_plans = [
        {
            "manifest_group_index": group_index,
            "manifests": [str(path.resolve()) for path in args.manifest_groups[group_index]],
            "manifest_id": manifest_ids[group_index],
            "num_contexts": len(dataset),
            "contexts_per_optimizer_step": len(dataset),
            "num_queries": sum(item.needle_count for item in dataset.plans),
            "coverage_min": int(coverage_by_manifest_group[group_index].min().item()),
            "coverage_max": int(coverage_by_manifest_group[group_index].max().item()),
            "contexts_per_rank": [
                len(shard.global_offsets)
                for shard in shards_by_manifest_group[group_index]
            ],
            "ddp_slots_per_rank": [
                len(shard.slot_offsets)
                for shard in shards_by_manifest_group[group_index]
            ],
            "padded_contexts_per_rank": [
                len(shard.slot_offsets) - len(shard.global_offsets)
                for shard in shards_by_manifest_group[group_index]
            ],
        }
        for group_index, dataset in enumerate(datasets)
    ]
    first_group_plan = group_plans[0]
    plan = {
        "purpose": (
            "fixed-parameter multi-lambda L0 audit"
            if args.l0_candidate_audit_only
            else (
                "Step-continuation polarization candidate audit"
                if args.polarization_candidate_audit_only
                else "official PyTorch DDP Q-head Hard Concrete training"
            )
        ),
        "manifests": [str(path.resolve()) for path in args.manifests],
        "manifest_id": manifest_id,
        "manifest_group_schedule_policy": "optimizer_step_modulo",
        "num_manifest_groups": len(datasets),
        "manifest_groups": group_plans,
        "num_contexts": first_group_plan["num_contexts"],
        "contexts_per_optimizer_step": first_group_plan[
            "contexts_per_optimizer_step"
        ],
        "num_queries": first_group_plan["num_queries"],
        "num_distances": layout.num_learnable_blocks,
        "coverage_min": min(item["coverage_min"] for item in group_plans),
        "coverage_max": max(item["coverage_max"] for item in group_plans),
        "world_size": planned_world_size,
        "contexts_per_rank": first_group_plan["contexts_per_rank"],
        "ddp_slots_per_rank": first_group_plan["ddp_slots_per_rank"],
        "padded_contexts_per_rank": first_group_plan[
            "padded_contexts_per_rank"
        ],
        "gradient_sync": "DistributedDataParallel average",
        "task_loss_scale": planned_world_size,
        "full_coverage_before_optimizer_step": True,
        "l0_in_final_synced_backward": True,
        "continuation": (
            {
                "source_checkpoint": str(
                    args.continuation_checkpoint.resolve()
                ),
                "expected_completed_optimizer_steps": (
                    args.continuation_expected_step
                ),
            }
            if args.continuation_checkpoint is not None
            else None
        ),
        "hyperparameters": {
            "initial_keep_mapping": args.initial_keep_mapping,
            "initial_keep_target_mean": args.initial_keep_target_mean,
            "initial_keep_score_slope": args.initial_keep_score_slope,
            "learning_rate": args.learning_rate,
            "l0_lambda": args.l0_lambda,
            "l0_candidate_audit_only": args.l0_candidate_audit_only,
            "l0_candidate_audit_lambdas": list(
                args.l0_candidate_audit_lambdas
            ),
            "max_optimizer_steps": args.max_optimizer_steps,
            "l0_start_step": args.l0_start_step,
            "l0_ramp_steps": args.l0_ramp_steps,
            "polarization_lambda": args.polarization_lambda,
            "polarization_start_step": args.polarization_start_step,
            "polarization_ramp_steps": args.polarization_ramp_steps,
            "polarization_candidate_audit_only": (
                args.polarization_candidate_audit_only
            ),
            "polarization_candidate_audit_lambdas": list(
                args.polarization_candidate_audit_lambdas
            ),
            "gradient_clip_norm": args.gradient_clip_norm,
            "gate_forward_mode": args.gate_forward_mode,
            "st_topk_keep_ratios": list(args.st_topk_keep_ratios),
            "st_topk_assignment_policy": "rotating_context_round_robin",
        },
    }
    if args.dry_run or int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    rank, world_size, local_rank, device = init_distributed(
        args.ddp_timeout_minutes
    )
    if world_size != planned_world_size:
        raise AssertionError("WORLD_SIZE changed during initialization")
    is_main = rank == 0
    set_reproducibility(args.master_seed)

    try:
        if is_main:
            print("stage=model_load", flush=True)
        model = model_loader(
            args.model_name_or_path,
            torch_dtype=resolve_dtype(args.torch_dtype),
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        model.config.use_cache = True
        runner = runner_factory(model, layout=layout)
        initialization = initialize_q_head_gates_from_retrieval(
            args.retrieval_checkpoint,
            metric=args.retrieval_metric,
            expected_shape=(
                runner.num_layers,
                runner.num_q_heads,
                layout.num_learnable_blocks,
            ),
            keep_floor=args.initial_keep_floor,
            keep_gain=args.initial_keep_gain,
            mapping=args.initial_keep_mapping,
            target_mean=args.initial_keep_target_mean,
            score_slope=args.initial_keep_score_slope,
            probability_min=args.initial_keep_probability_min,
            probability_max=args.initial_keep_probability_max,
            device=device,
        )
        base_gates = initialization.gates
        ddp_gates = DistributedDataParallel(
            base_gates,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            ddp_gates.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("suffix runner did not freeze model weights")

        signature = training_signature(
            args,
            manifest_id=manifest_id,
            world_size=world_size,
        )
        checkpoint_path = args.output_dir / "latest.pt"
        log_path = args.output_dir / "train_log.jsonl"
        if is_main:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        optimizer_step = 0
        continuation_provenance: dict[str, Any] | None = None
        if checkpoint_path.exists():
            if not args.resume:
                raise FileExistsError(
                    f"{checkpoint_path} exists; use --resume or a new output-dir"
                )
            state = torch.load(checkpoint_path, map_location="cpu")
            if int(state.get("schema_version", -1)) != 3:
                raise ValueError("unsupported DDP checkpoint schema")
            if state.get("signature") != signature:
                raise ValueError("checkpoint signature differs from this run")
            continuation_provenance = state.get("continuation_provenance")
            with torch.no_grad():
                base_gates.log_alpha.copy_(
                    torch.as_tensor(
                        state["log_alpha"],
                        dtype=torch.float32,
                        device=device,
                    )
                )
            optimizer.load_state_dict(state["optimizer_state"])
            optimizer_step = int(state["completed_optimizer_steps"])
            if not 0 <= optimizer_step <= args.max_optimizer_steps:
                raise ValueError("checkpoint completed step lies outside this plan")
            if is_main:
                print(f"stage=resume optimizer_step={optimizer_step}", flush=True)
        elif args.continuation_checkpoint is not None:
            state = torch.load(args.continuation_checkpoint, map_location="cpu")
            continuation_metadata = validate_continuation_checkpoint(
                state,
                target_signature=signature,
                expected_completed_optimizer_steps=(
                    args.continuation_expected_step
                ),
                allow_manifest_change=(
                    args.allow_manifest_change_on_continuation
                ),
            )
            continuation_provenance = {
                "source_checkpoint": str(
                    args.continuation_checkpoint.resolve()
                ),
                **continuation_metadata,
            }
            with torch.no_grad():
                base_gates.log_alpha.copy_(
                    torch.as_tensor(
                        state["log_alpha"],
                        dtype=torch.float32,
                        device=device,
                    )
                )
            optimizer.load_state_dict(state["optimizer_state"])
            optimizer_step = int(state["completed_optimizer_steps"])
            imported_keep_probability = base_gates.keep_probability().detach()
            if is_main:
                save_checkpoint(
                    checkpoint_path,
                    signature=signature,
                    completed_optimizer_steps=optimizer_step,
                    log_alpha=base_gates.log_alpha,
                    keep_probability=imported_keep_probability,
                    optimizer=optimizer,
                    hard_mask_threshold=args.hard_mask_threshold,
                    continuation_provenance=continuation_provenance,
                )
                if optimizer_step in set(args.milestone_steps):
                    save_checkpoint(
                        args.output_dir
                        / "milestones"
                        / f"step_{optimizer_step:04d}.pt",
                        signature=signature,
                        completed_optimizer_steps=optimizer_step,
                        log_alpha=base_gates.log_alpha,
                        keep_probability=imported_keep_probability,
                        optimizer=optimizer,
                        hard_mask_threshold=args.hard_mask_threshold,
                        continuation_provenance=continuation_provenance,
                        allow_overwrite=False,
                    )
                (args.output_dir / "continuation_import.json").write_text(
                    json.dumps(
                        continuation_provenance,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    "stage=continuation_import "
                    f"optimizer_step={optimizer_step} "
                    f"source={args.continuation_checkpoint.resolve()}",
                    flush=True,
                )

        run_limit = args.max_optimizer_steps
        if args.optimizer_steps_this_run is not None:
            run_limit = min(
                args.max_optimizer_steps,
                optimizer_step + args.optimizer_steps_this_run,
            )
        dist.barrier()

        audit_completed = False
        while optimizer_step < run_limit:
            manifest_group_index = manifest_group_index_for_step(
                optimizer_step,
                num_manifest_groups=len(datasets),
            )
            dataset = datasets[manifest_group_index]
            step_manifest_id = manifest_ids[manifest_group_index]
            gate_index_groups = gate_index_groups_by_manifest_group[
                manifest_group_index
            ]
            coverage = coverage_by_manifest_group[manifest_group_index]
            contexts_per_optimizer_step = (
                contexts_per_optimizer_step_by_manifest_group[
                    manifest_group_index
                ]
            )
            shards = shards_by_manifest_group[manifest_group_index]
            local_offsets = shards[rank].global_offsets
            local_slots = shards[rank].slot_offsets
            if not local_offsets:
                raise AssertionError("every DDP rank must own at least one context")
            if len({len(shard.slot_offsets) for shard in shards}) != 1:
                raise AssertionError(
                    "all ranks must use equal DDP synchronization slots"
                )
            step_started = time.perf_counter()
            step_progress = LogProgressBar(
                total=len(local_slots),
                label=f"train step {optimizer_step + 1}/{run_limit}",
                unit="DDP slots",
                enabled=is_main,
            )
            step_progress.start(
                detail=(
                    f"round group {manifest_group_index + 1}/{len(datasets)}; "
                    f"{len(dataset)} global contexts across {world_size} GPUs"
                )
            )
            occurrences = context_occurrences_for_step(
                num_contexts=len(dataset),
                contexts_per_optimizer_step=contexts_per_optimizer_step,
                optimizer_step=optimizer_step,
                master_seed=args.master_seed,
                shuffle=args.shuffle_contexts,
            )
            selected_gate_indices = tuple(
                gate_index_groups[item.dataset_index] for item in occurrences
            )
            global_branch_weights = normalized_step_branch_weights(
                selected_gate_indices,
                coverage_count=coverage,
            )
            global_weight_total = sum(
                float(weights.sum().item()) for weights in global_branch_weights
            )
            if not math.isclose(global_weight_total, 1.0, abs_tol=1e-6):
                raise AssertionError("global branch weights do not sum to one")

            optimizer.zero_grad(set_to_none=True)
            local_task_loss = 0.0
            local_weight = 0.0
            local_queries = 0
            scheduled_l0_lambda = scheduled_l0_coefficient(
                optimizer_step=optimizer_step,
                target_lambda=args.l0_lambda,
                start_step=args.l0_start_step,
                ramp_steps=args.l0_ramp_steps,
            )
            scheduled_polarization_lambda = scheduled_l0_coefficient(
                optimizer_step=optimizer_step,
                target_lambda=args.polarization_lambda,
                start_step=args.polarization_start_step,
                ramp_steps=args.polarization_ramp_steps,
            )
            # The continuation audit needs the pure task gradient at the
            # checkpoint parameters.  Candidate L0/polarization gradients are
            # added analytically below and simulated with the imported AdamW
            # moments, without mutating the checkpoint or optimizer.
            audit_no_regularization = args.polarization_candidate_audit_only
            l0_lambda = 0.0 if audit_no_regularization else scheduled_l0_lambda
            polarization_lambda = (
                0.0
                if audit_no_regularization
                else scheduled_polarization_lambda
            )
            keep_probability_before = base_gates.keep_probability()
            endpoint_mass_before = base_gates.endpoint_mass()
            endpoint_mask_before = endpoint_mass_before.one.gt(
                endpoint_mass_before.zero
            )
            expected_keep_ratio = keep_probability_before.mean()
            expected_interior_mass = endpoint_mass_before.interior.mean()
            regularization_loss = (
                expected_keep_ratio * l0_lambda
                + expected_interior_mass * polarization_lambda
            )
            l0_gradient = (
                l0_lambda * keep_probability_before.detach()
                * (1.0 - keep_probability_before.detach())
                / keep_probability_before.numel()
            )
            polarization_gradient = (
                polarization_lambda
                * (
                    keep_probability_before.detach()
                    * (1.0 - keep_probability_before.detach())
                    - endpoint_mass_before.one.detach()
                    * (1.0 - endpoint_mass_before.one.detach())
                )
                / keep_probability_before.numel()
            )
            regularization_active = (
                l0_lambda > 0.0 or polarization_lambda > 0.0
            )
            st_topk_masks: dict[float, torch.Tensor] = {}
            st_topk_context_counts: dict[str, int] = {}
            st_topk_task_weights: dict[str, float] = {}
            local_st_topk_loss_contributions: dict[str, float] = {}
            if args.gate_forward_mode == "st_topk":
                for keep_ratio in args.st_topk_keep_ratios:
                    ratio = float(keep_ratio)
                    st_topk_masks[ratio] = exact_topk_mask(
                        keep_probability_before.detach(),
                        ratio,
                    ).to(device=device)
                    ratio_key = f"{ratio:.8g}"
                    st_topk_context_counts[ratio_key] = 0
                    st_topk_task_weights[ratio_key] = 0.0
                    local_st_topk_loss_contributions[ratio_key] = 0.0
                for global_offset, weights in enumerate(global_branch_weights):
                    ratio = st_topk_keep_ratio_for_context(
                        args.st_topk_keep_ratios,
                        optimizer_step=optimizer_step,
                        position_in_step=global_offset,
                    )
                    ratio_key = f"{ratio:.8g}"
                    st_topk_context_counts[ratio_key] += 1
                    st_topk_task_weights[ratio_key] += float(weights.sum().item())

            for local_position, global_offset in enumerate(local_slots):
                final_local_context = local_position == len(local_slots) - 1
                if global_offset is None:
                    # Ranks with fewer real contexts still participate in the
                    # same final DDP collective.  This zero-loss branch adds
                    # no task gradient and avoids duplicating a real sample.
                    dummy_uniform = torch.full(
                        base_gates.shape,
                        0.5,
                        dtype=torch.float32,
                        device=device,
                    )
                    sync_context = (
                        nullcontext()
                        if final_local_context
                        else ddp_gates.no_sync()
                    )
                    with sync_context:
                        dummy_relaxed_gate = ddp_gates(dummy_uniform)
                        dummy_loss = dummy_relaxed_gate.sum() * 0.0
                        if final_local_context and regularization_active:
                            dummy_loss = dummy_loss + regularization_loss
                        dummy_loss.backward()
                    del dummy_uniform, dummy_relaxed_gate, dummy_loss
                    display_completed = min(
                        local_position + 1,
                        max(0, len(local_slots) - 1),
                    )
                    step_progress.update(
                        display_completed,
                        detail="waiting for final DDP synchronization"
                        if final_local_context
                        else None,
                    )
                    continue

                occurrence = occurrences[global_offset]
                example = dataset[occurrence.dataset_index]
                weights = global_branch_weights[global_offset]
                seed = q_head_training_gate_sample_seed(
                    args.master_seed,
                    manifest_id=step_manifest_id,
                    optimizer_step=optimizer_step,
                    position_in_step=global_offset,
                    sample_id=example.sample_id,
                )
                generator = torch.Generator(device=device).manual_seed(seed)
                uniform = torch.rand(
                    base_gates.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                prompt_cache = runner.dense_prefill(example.prompt_input_ids)
                binary_gate = None
                st_topk_ratio_key = None
                if args.gate_forward_mode == "st_topk":
                    ratio = st_topk_keep_ratio_for_context(
                        args.st_topk_keep_ratios,
                        optimizer_step=optimizer_step,
                        position_in_step=global_offset,
                    )
                    binary_gate = st_topk_masks[ratio]
                    st_topk_ratio_key = f"{ratio:.8g}"
                result = backward_distributed_q_head_branches(
                    runner,
                    prompt_cache,
                    ddp_gates,
                    tuple(branch.suffix_input_ids for branch in example.branches),
                    tuple(branch.suffix_labels for branch in example.branches),
                    branch_weights=weights,
                    uniform=uniform,
                    binary_gate=binary_gate,
                    synchronize_gradients=final_local_context,
                    final_regularization_loss=(
                        regularization_loss
                        if final_local_context and regularization_active
                        else None
                    ),
                    checkpoint_layers=args.checkpoint_layers,
                )
                local_task_loss += result.weighted_task_loss
                if st_topk_ratio_key is not None:
                    local_st_topk_loss_contributions[
                        st_topk_ratio_key
                    ] += result.weighted_task_loss
                local_weight += float(weights.sum().item())
                local_queries += len(example.branches)
                del prompt_cache, uniform, result
                display_completed = min(
                    local_position + 1,
                    max(0, len(local_slots) - 1),
                )
                estimated_global_contexts = min(
                    display_completed * world_size,
                    len(dataset),
                )
                step_progress.update(
                    display_completed,
                    detail=(
                        "waiting for final DDP synchronization"
                        if final_local_context
                        else f"≈{estimated_global_contexts}/{len(dataset)} global contexts"
                    ),
                )

            total_gradient = base_gates.log_alpha.grad
            if total_gradient is None or not torch.isfinite(total_gradient).all():
                raise FloatingPointError("DDP gate gradient is absent or non-finite")
            task_gradient = (
                total_gradient.detach()
                - l0_gradient
                - polarization_gradient
            )
            task_gradient_norm = float(task_gradient.float().norm().item())
            l0_gradient_norm = float(l0_gradient.float().norm().item())
            polarization_gradient_norm = float(
                polarization_gradient.float().norm().item()
            )
            total_gradient_norm = float(total_gradient.float().norm().item())
            gatewise_gradient_competition = gradient_competition_statistics(
                task_gradient,
                l0_gradient,
            )
            polarization_gradient_competition = gradient_competition_statistics(
                task_gradient + l0_gradient,
                polarization_gradient,
            )

            st_topk_ratio_keys = tuple(local_st_topk_loss_contributions)
            reduced_step_values = reduce_sum(
                [
                    local_task_loss,
                    local_weight,
                    float(local_queries),
                    float(len(local_offsets)),
                    *(
                        local_st_topk_loss_contributions[key]
                        for key in st_topk_ratio_keys
                    ),
                ],
                device,
            )
            (
                global_task_loss,
                global_weight,
                global_queries,
                global_contexts,
                *global_st_topk_loss_values,
            ) = reduced_step_values
            st_topk_loss_by_ratio = {
                key: {
                    "weighted_loss_contribution": contribution,
                    "normalized_task_loss": (
                        contribution / st_topk_task_weights[key]
                    ),
                }
                for key, contribution in zip(
                    st_topk_ratio_keys,
                    global_st_topk_loss_values,
                    strict=True,
                )
            }
            if not math.isclose(global_weight, 1.0, abs_tol=1e-5):
                raise AssertionError(f"global processed weight is {global_weight}")
            if int(global_contexts) != len(dataset):
                raise AssertionError("DDP step did not process every context exactly once")
            elapsed = torch.tensor(
                time.perf_counter() - step_started,
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

            if args.l0_candidate_audit_only:
                if l0_lambda != 0.0:
                    raise AssertionError("audit task gradient must be collected at lambda=0")
                unit_l0_gradient = (
                    keep_probability_before.detach()
                    * (1.0 - keep_probability_before.detach())
                    / keep_probability_before.numel()
                )
                if is_main:
                    candidate_statistics = l0_candidate_first_step_statistics(
                        task_gradient,
                        unit_l0_gradient,
                        base_gates.log_alpha.detach(),
                        candidates=args.l0_candidate_audit_lambdas,
                        learning_rate=args.learning_rate,
                        gradient_clip_norm=args.gradient_clip_norm,
                        hard_mask_threshold=args.hard_mask_threshold,
                        config=base_gates.config,
                    )
                    tensor_path = args.output_dir / "l0_candidate_audit.pt"
                    report_path = args.output_dir / "l0_candidate_audit.json"
                    report = {
                        "schema_version": 1,
                        "plan": plan,
                        "optimizer_step_called": False,
                        "world_size": world_size,
                        "contexts": int(global_contexts),
                        "queries": int(global_queries),
                        "weighted_task_loss": global_task_loss,
                        "elapsed_seconds_max_rank": float(elapsed.item()),
                        "candidate_statistics": candidate_statistics,
                    }
                    atomic_torch_save(
                        {
                            "schema_version": 1,
                            "plan": plan,
                            "task_gradient": task_gradient.float().cpu(),
                            "unit_l0_gradient": unit_l0_gradient.float().cpu(),
                            "log_alpha": base_gates.log_alpha.detach().float().cpu(),
                            "initial_keep_probability": (
                                keep_probability_before.detach().float().cpu()
                            ),
                            "l0_candidates": list(args.l0_candidate_audit_lambdas),
                            "learning_rate": args.learning_rate,
                            "gradient_clip_norm": args.gradient_clip_norm,
                            "hard_mask_threshold": args.hard_mask_threshold,
                            "weighted_task_loss": global_task_loss,
                        },
                        tensor_path,
                    )
                    report_path.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    summary = {
                        **plan,
                        "status": "completed",
                        "optimizer_step_called": False,
                        "audit_report": str(report_path.resolve()),
                        "audit_tensors": str(tensor_path.resolve()),
                    }
                    (args.output_dir / "completed_summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(json.dumps(report, ensure_ascii=False), flush=True)
                step_progress.finish(
                    detail=f"loss={global_task_loss:.6f} audit-only; no update"
                )
                audit_completed = True
                dist.barrier()
                break

            if args.polarization_candidate_audit_only:
                if l0_lambda != 0.0 or polarization_lambda != 0.0:
                    raise AssertionError(
                        "polarization audit must collect a pure task gradient"
                    )
                unit_l0_gradient = (
                    keep_probability_before.detach()
                    * (1.0 - keep_probability_before.detach())
                    / keep_probability_before.numel()
                )
                unit_polarization_gradient = (
                    keep_probability_before.detach()
                    * (1.0 - keep_probability_before.detach())
                    - endpoint_mass_before.one.detach()
                    * (1.0 - endpoint_mass_before.one.detach())
                ) / keep_probability_before.numel()
                adam_state = optimizer.state.get(base_gates.log_alpha)
                if not isinstance(adam_state, dict):
                    raise ValueError(
                        "continuation optimizer has no state for gate parameter"
                    )
                required_adam_state = ("step", "exp_avg", "exp_avg_sq")
                missing_adam_state = [
                    key for key in required_adam_state if key not in adam_state
                ]
                if missing_adam_state:
                    raise ValueError(
                        "continuation AdamW state is missing: "
                        + ", ".join(missing_adam_state)
                    )
                adam_step_value = adam_state["step"]
                adam_step = int(
                    adam_step_value.item()
                    if torch.is_tensor(adam_step_value)
                    else adam_step_value
                )
                if adam_step != optimizer_step:
                    raise ValueError(
                        "continuation AdamW step differs from checkpoint step: "
                        f"{adam_step} != {optimizer_step}"
                    )
                optimizer_group = optimizer.param_groups[0]
                beta1, beta2 = optimizer_group["betas"]
                if float(optimizer_group.get("weight_decay", 0.0)) != 0.0:
                    raise ValueError(
                        "polarization continuation audit requires zero weight decay"
                    )
                if is_main:
                    candidate_statistics = (
                        polarization_candidate_continuation_statistics(
                            task_gradient,
                            unit_l0_gradient,
                            unit_polarization_gradient,
                            base_gates.log_alpha.detach(),
                            adam_state["exp_avg"],
                            adam_state["exp_avg_sq"],
                            adam_step=adam_step,
                            candidates=(
                                args.polarization_candidate_audit_lambdas
                            ),
                            l0_lambda=scheduled_l0_lambda,
                            learning_rate=float(optimizer_group["lr"]),
                            beta1=float(beta1),
                            beta2=float(beta2),
                            gradient_clip_norm=args.gradient_clip_norm,
                            adam_epsilon=float(optimizer_group["eps"]),
                            config=base_gates.config,
                        )
                    )
                    tensor_path = (
                        args.output_dir / "polarization_candidate_audit.pt"
                    )
                    report_path = (
                        args.output_dir / "polarization_candidate_audit.json"
                    )
                    report = {
                        "schema_version": 1,
                        "plan": plan,
                        "optimizer_step_called": False,
                        "source_completed_optimizer_steps": optimizer_step,
                        "world_size": world_size,
                        "contexts": int(global_contexts),
                        "queries": int(global_queries),
                        "weighted_task_loss": global_task_loss,
                        "elapsed_seconds_max_rank": float(elapsed.item()),
                        "candidate_statistics": candidate_statistics,
                    }
                    atomic_torch_save(
                        {
                            "schema_version": 1,
                            "plan": plan,
                            "optimizer_step_called": False,
                            "source_completed_optimizer_steps": optimizer_step,
                            "task_gradient": task_gradient.float().cpu(),
                            "unit_l0_gradient": unit_l0_gradient.float().cpu(),
                            "unit_polarization_gradient": (
                                unit_polarization_gradient.float().cpu()
                            ),
                            "log_alpha": (
                                base_gates.log_alpha.detach().float().cpu()
                            ),
                            "adam_exp_avg": (
                                adam_state["exp_avg"].detach().float().cpu()
                            ),
                            "adam_exp_avg_sq": (
                                adam_state["exp_avg_sq"].detach().float().cpu()
                            ),
                            "adam_step": adam_step,
                            "l0_lambda": scheduled_l0_lambda,
                            "polarization_candidates": list(
                                args.polarization_candidate_audit_lambdas
                            ),
                            "weighted_task_loss": global_task_loss,
                        },
                        tensor_path,
                    )
                    report_path.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    summary = {
                        **plan,
                        "status": "completed",
                        "optimizer_step_called": False,
                        "audit_report": str(report_path.resolve()),
                        "audit_tensors": str(tensor_path.resolve()),
                    }
                    (args.output_dir / "completed_summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(json.dumps(report, ensure_ascii=False), flush=True)
                step_progress.finish(
                    detail=f"loss={global_task_loss:.6f} audit-only; no update"
                )
                audit_completed = True
                dist.barrier()
                break

            pre_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [base_gates.log_alpha],
                    args.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )

            before = base_gates.log_alpha.detach().clone()
            hard_before = base_gates.keep_probability().ge(args.hard_mask_threshold)
            optimizer.step()
            update = base_gates.log_alpha.detach() - before
            keep_probability = base_gates.keep_probability().detach()
            endpoint_mass_after = base_gates.endpoint_mass()
            endpoint_mask_after = endpoint_mass_after.one.gt(
                endpoint_mass_after.zero
            )
            hard_after = keep_probability.ge(args.hard_mask_threshold)
            max_rank_difference = replicated_parameter_max_difference(
                base_gates.log_alpha
            )
            if max_rank_difference > 1e-6:
                raise AssertionError(
                    f"DDP replicas diverged: max difference={max_rank_difference}"
                )

            completed_step = optimizer_step + 1
            quantiles = torch.quantile(
                keep_probability.float(),
                torch.tensor([0.1, 0.5, 0.9], device=device),
            )
            st_topk_mask_stability: dict[str, dict[str, float]] = {}
            if is_main and args.gate_forward_mode == "st_topk":
                for keep_ratio in args.st_topk_keep_ratios:
                    ratio = float(keep_ratio)
                    before_mask = st_topk_masks[ratio].detach().cpu()
                    after_mask = exact_topk_mask(keep_probability, ratio)
                    intersection = before_mask.logical_and(after_mask).sum()
                    st_topk_mask_stability[f"{ratio:.8g}"] = {
                        "actual_keep_ratio": float(after_mask.float().mean().item()),
                        "all_gate_change_ratio": float(
                            before_mask.ne(after_mask).float().mean().item()
                        ),
                        "retained_open_fraction": float(
                            intersection.float().div(before_mask.sum()).item()
                        ),
                    }
            record = {
                "optimizer_step": optimizer_step,
                "completed_optimizer_steps": completed_step,
                "manifest_group_index": manifest_group_index,
                "manifest_id": step_manifest_id,
                "manifests": [
                    str(path.resolve())
                    for path in args.manifest_groups[manifest_group_index]
                ],
                "world_size": world_size,
                "contexts": int(global_contexts),
                "queries": int(global_queries),
                "weighted_task_loss": global_task_loss,
                "task_gradient_norm": task_gradient_norm,
                "l0_lambda": l0_lambda,
                "scheduled_l0_lambda": scheduled_l0_lambda,
                "l0_gradient_norm": l0_gradient_norm,
                "polarization_lambda": polarization_lambda,
                "scheduled_polarization_lambda": (
                    scheduled_polarization_lambda
                ),
                "polarization_gradient_norm": polarization_gradient_norm,
                "total_gradient_norm": total_gradient_norm,
                "gatewise_gradient_competition": gatewise_gradient_competition,
                "polarization_gradient_competition": (
                    polarization_gradient_competition
                ),
                "pre_clip_gradient_norm": pre_clip_norm,
                "gate_forward_mode": args.gate_forward_mode,
                "st_topk_keep_ratios": list(args.st_topk_keep_ratios),
                "st_topk_context_counts": st_topk_context_counts,
                "st_topk_task_weights": st_topk_task_weights,
                "st_topk_loss_by_ratio": st_topk_loss_by_ratio,
                "st_topk_mask_stability": st_topk_mask_stability,
                "mean_abs_log_alpha_update": float(update.abs().mean().item()),
                "max_abs_log_alpha_update": float(update.abs().max().item()),
                "expected_keep_ratio_before_update": float(
                    expected_keep_ratio.detach().item()
                ),
                "hard_keep_ratio_after_update": float(hard_after.float().mean().item()),
                "hard_mask_change_ratio": float(
                    hard_after.ne(hard_before).float().mean().item()
                ),
                "endpoint_mass_before_update": {
                    "zero": float(endpoint_mass_before.zero.mean().item()),
                    "interior": float(
                        endpoint_mass_before.interior.mean().item()
                    ),
                    "one": float(endpoint_mass_before.one.mean().item()),
                },
                "endpoint_mass_after_update": {
                    "zero": float(endpoint_mass_after.zero.mean().item()),
                    "interior": float(
                        endpoint_mass_after.interior.mean().item()
                    ),
                    "one": float(endpoint_mass_after.one.mean().item()),
                },
                "endpoint_keep_ratio_before_update": float(
                    endpoint_mask_before.float().mean().item()
                ),
                "endpoint_keep_ratio_after_update": float(
                    endpoint_mask_after.float().mean().item()
                ),
                "endpoint_mask_change_ratio": float(
                    endpoint_mask_after.ne(endpoint_mask_before).float().mean().item()
                ),
                "interior_mass_change": float(
                    endpoint_mass_after.interior.mean()
                    .sub(endpoint_mass_before.interior.mean())
                    .item()
                ),
                "keep_probability_after_update": {
                    "min": float(keep_probability.min().item()),
                    "p10": float(quantiles[0].item()),
                    "median": float(quantiles[1].item()),
                    "mean": float(keep_probability.mean().item()),
                    "p90": float(quantiles[2].item()),
                    "max": float(keep_probability.max().item()),
                },
                "max_rank_parameter_difference": max_rank_difference,
                "elapsed_seconds_max_rank": float(elapsed.item()),
            }
            step_progress.finish(
                detail=(
                    f"loss={global_task_loss:.6f} "
                    f"keep={record['keep_probability_after_update']['mean']:.6f}"
                )
            )

            optimizer_step = completed_step
            if is_main:
                save_checkpoint(
                    checkpoint_path,
                    signature=signature,
                    completed_optimizer_steps=optimizer_step,
                    log_alpha=base_gates.log_alpha,
                    keep_probability=keep_probability,
                    optimizer=optimizer,
                    hard_mask_threshold=args.hard_mask_threshold,
                    continuation_provenance=continuation_provenance,
                )
                if optimizer_step in set(args.milestone_steps):
                    save_checkpoint(
                        args.output_dir / "milestones" / f"step_{optimizer_step:04d}.pt",
                        signature=signature,
                        completed_optimizer_steps=optimizer_step,
                        log_alpha=base_gates.log_alpha,
                        keep_probability=keep_probability,
                        optimizer=optimizer,
                        hard_mask_threshold=args.hard_mask_threshold,
                        continuation_provenance=continuation_provenance,
                        allow_overwrite=False,
                    )
                append_jsonl(log_path, record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            dist.barrier()

        if is_main and not audit_completed:
            status = "completed" if optimizer_step >= args.max_optimizer_steps else "paused"
            summary = {
                **plan,
                "status": status,
                "completed_optimizer_steps": optimizer_step,
                "checkpoint": str(checkpoint_path.resolve()),
                "train_log": str(log_path.resolve()),
                "continuation_provenance": continuation_provenance,
            }
            summary_path = args.output_dir / f"{status}_summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
