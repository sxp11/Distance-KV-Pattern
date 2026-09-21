#!/usr/bin/env python3
"""Collect formal 128-token retrieval scores for Hard Concrete initialization."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


METHOD_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from distance_kv_pattern import (
    BlockLayout,
    InstructMaterializer,
    LogProgressBar,
    read_manifest,
)
from distance_kv_pattern.training.retrieval_initialization.retrieval_init import (
    RAW_SCHEMA_VERSION,
    aggregate_raw_records,
    append_jsonl,
    read_jsonl,
    save_aggregate,
)
from distance_kv_pattern.core.randomness import set_reproducibility


@dataclass(frozen=True, slots=True)
class ContextPlan:
    manifest_path: str
    manifest_index: int
    sample_id: str
    value_type: str
    needle_count: int
    template_id: str
    coverage_round: int
    branch_ids: tuple[str, ...]
    gate_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--manifest-root",
        type=Path,
    )
    parser.add_argument("--manifest-paths", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--value-types", nargs="+", default=("num", "word"))
    parser.add_argument("--needle-count", type=int, default=4)
    parser.add_argument("--template-ids", nargs="+", default=("Q1", "Q3"))
    parser.add_argument("--coverage-round", type=int, default=0)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--top-q", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-contexts", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def branch_id(sample_id: str, needle_index: int) -> str:
    return f"{sample_id}::needle_{needle_index}"


def manifest_path(
    root: Path,
    *,
    value_type: str,
    needle_count: int,
    template_id: str,
    coverage_round: int,
) -> Path:
    return (
        root
        / value_type
        / f"n{needle_count}"
        / template_id.lower()
        / f"round_{coverage_round:03d}.jsonl"
    )


def build_plan(args: argparse.Namespace) -> tuple[ContextPlan, ...]:
    contexts: list[ContextPlan] = []
    paths = args.manifest_paths
    if paths is None:
        paths = [
            manifest_path(
                args.manifest_root,
                value_type=value_type,
                needle_count=args.needle_count,
                template_id=template_id,
                coverage_round=args.coverage_round,
            )
            for value_type in args.value_types
            for template_id in args.template_ids
        ]
    for path in paths:
        plans = tuple(read_manifest(path))
        for manifest_index, plan in enumerate(plans):
            contexts.append(
                ContextPlan(
                    manifest_path=str(path.resolve()),
                    manifest_index=manifest_index,
                    sample_id=plan.sample_id,
                    value_type=plan.value_type,
                    needle_count=plan.needle_count,
                    template_id=plan.template_id,
                    coverage_round=plan.coverage_round,
                    branch_ids=tuple(
                        branch_id(plan.sample_id, needle_index)
                        for needle_index in plan.query_order
                    ),
                    gate_indices=tuple(
                        plan.needles[needle_index].gate_index
                        for needle_index in plan.query_order
                    ),
                )
            )
    if args.limit_contexts is not None:
        if args.limit_contexts <= 0:
            raise ValueError("--limit-contexts must be positive")
        contexts = contexts[: args.limit_contexts]
    return tuple(contexts)


def plan_summary(
    contexts: Sequence[ContextPlan], layout: BlockLayout
) -> dict[str, Any]:
    coverage = torch.zeros(layout.num_learnable_blocks, dtype=torch.int64)
    for context in contexts:
        for gate_index in context.gate_indices:
            coverage[gate_index] += 1
    return {
        "num_contexts": len(contexts),
        "num_branches": sum(len(context.branch_ids) for context in contexts),
        "num_covered_distances": int(coverage.gt(0).sum().item()),
        "minimum_coverage": int(coverage.min().item()),
        "maximum_coverage": int(coverage.max().item()),
        "missing_gate_indices": torch.nonzero(
            coverage.eq(0), as_tuple=False
        ).flatten().tolist(),
    }


def completed_branches(records_path: Path) -> set[str]:
    return {
        str(record["branch_id"])
        for record in read_jsonl(records_path)
        if record.get("error") is None
    }


def write_plan(
    args: argparse.Namespace,
    contexts: Sequence[ContextPlan],
    layout: BlockLayout,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            "FlashAttention dense 128K prompt prefill; shared prompt cache; "
            "independent single-key teacher-forced suffixes; eager attention "
            "only for the five causal rows that predict value tokens"
        ),
        "schema_version": RAW_SCHEMA_VERSION,
        "master_seed": args.master_seed,
        "model_name_or_path": args.model_name_or_path,
        "layout": layout.as_dict(),
        "selectors": {
            "value_types": list(args.value_types),
            "needle_count": args.needle_count,
            "template_ids": list(args.template_ids),
            "coverage_round": args.coverage_round,
            "retrieval_k": args.retrieval_k,
            "top_q": args.top_q,
        },
        "summary": plan_summary(contexts, layout),
        "contexts": [asdict(context) for context in contexts],
    }
    (args.output_dir / "plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(score_branch_fn: Callable[..., dict[str, Any]]) -> None:
    args = parse_args()
    set_reproducibility(args.master_seed)
    if args.retrieval_k <= 0:
        raise ValueError("--retrieval-k must be positive")
    layout = BlockLayout()
    contexts = build_plan(args)
    if not contexts:
        raise ValueError("retrieval initialization plan is empty")
    write_plan(args, contexts, layout)
    summary = plan_summary(contexts, layout)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"plan={args.output_dir / 'plan.json'}", flush=True)
    if args.dry_run:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal 128K score collection")

    records_path = args.output_dir / "raw_branch_scores.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            f"raw results already exist: {records_path}; use --resume"
        )
    completed = completed_branches(records_path) if args.resume else set()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    materializer = InstructMaterializer(tokenizer, layout=layout)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.torch_dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    model.config.use_cache = True
    plan_cache: dict[str, tuple[Any, ...]] = {}
    pending_contexts = [
        context
        for context in contexts
        if any(item not in completed for item in context.branch_ids)
    ]
    device_name = (
        torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda")
        else args.device
    )
    print(
        f"completed_branches={len(completed)} "
        f"pending_contexts={len(pending_contexts)} "
        f"device={device_name}",
        flush=True,
    )
    progress = (
        LogProgressBar(
            total=len(pending_contexts),
            label="retrieval initialization",
            unit="contexts",
        )
        if pending_contexts
        else None
    )
    if progress is not None:
        progress.start(detail=f"already completed branches={len(completed)}")

    for ordinal, context in enumerate(pending_contexts, start=1):
        pending_ids = {
            item for item in context.branch_ids if item not in completed
        }
        plans = plan_cache.get(context.manifest_path)
        if plans is None:
            plans = tuple(read_manifest(Path(context.manifest_path)))
            plan_cache[context.manifest_path] = plans
        plan = plans[context.manifest_index]
        if plan.sample_id != context.sample_id:
            raise AssertionError("execution plan and manifest row disagree")
        sample = materializer.materialize_independent_queries(plan)
        prompt_ids = torch.tensor(
            sample.prompt_input_ids,
            dtype=torch.long,
            device=args.device,
        ).unsqueeze(0)
        torch.cuda.reset_peak_memory_stats()
        cache: DynamicCache | None = None
        prefill_seconds: float | None = None
        try:
            with torch.inference_mode():
                cache = DynamicCache()
                torch.cuda.synchronize()
                prefill_started = time.perf_counter()
                prompt_output = model.model(
                    input_ids=prompt_ids,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                cache = prompt_output.past_key_values
                del prompt_output
                torch.cuda.synchronize()
                prefill_seconds = time.perf_counter() - prefill_started
                if int(cache.get_seq_length()) != layout.context_tokens:
                    raise AssertionError("dense prompt cache has the wrong length")

                for suffix in sample.suffixes:
                    current_branch_id = branch_id(plan.sample_id, suffix.needle_index)
                    if current_branch_id not in pending_ids:
                        continue
                    needle = plan.needles[suffix.needle_index]
                    started = time.perf_counter()
                    base_record = {
                        "schema_version": RAW_SCHEMA_VERSION,
                        "branch_id": current_branch_id,
                        "sample_id": plan.sample_id,
                        "manifest_path": context.manifest_path,
                        "manifest_index": context.manifest_index,
                        "prompt_ids_sha256": sample.prompt_ids_sha256,
                        "value_type": plan.value_type,
                        "needle_count": plan.needle_count,
                        "template_id": plan.template_id,
                        "coverage_round": plan.coverage_round,
                        "query_position": suffix.query_position,
                        "needle_index": suffix.needle_index,
                        "key": suffix.key,
                        "value": suffix.value,
                        "absolute_block": needle.absolute_block,
                        "relative_distance": needle.relative_distance,
                        "gate_index": needle.gate_index,
                        "retrieval_k": args.retrieval_k,
                        "error": None,
                    }
                    try:
                        branch_metrics = score_branch_fn(
                            model=model,
                            tokenizer=tokenizer,
                            cache=cache,
                            prompt_length=layout.context_tokens,
                            suffix=suffix,
                            needle=needle,
                            layout=layout,
                            retrieval_k=args.retrieval_k,
                        )
                        record = {
                            **base_record,
                            "prefill_seconds": prefill_seconds,
                            "branch_seconds": time.perf_counter() - started,
                            **branch_metrics,
                        }
                        append_jsonl(records_path, record)
                        completed.add(current_branch_id)
                    except Exception as exc:
                        append_jsonl(
                            records_path,
                            {
                                **base_record,
                                "prefill_seconds": prefill_seconds,
                                "branch_seconds": time.perf_counter() - started,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        print(
                            f"  {current_branch_id} failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if not args.continue_on_error:
                            raise
                    finally:
                        cache.crop(layout.context_tokens)
                        if int(cache.get_seq_length()) != layout.context_tokens:
                            raise AssertionError("failed to restore pristine prompt cache")
        except Exception as exc:
            print(
                f"context {context.sample_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if not args.continue_on_error:
                raise
        finally:
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2
            del prompt_ids, cache
            torch.cuda.empty_cache()
            gc.collect()
            if progress is not None:
                prefill_text = "--" if prefill_seconds is None else f"{prefill_seconds:.1f}s"
                progress.update(
                    ordinal,
                    detail=(
                        f"branches={len(completed)}; last prefill={prefill_text}; "
                        f"peak={peak_mib:.0f}MiB"
                    ),
                )

    raw_records = read_jsonl(records_path)
    aggregate = aggregate_raw_records(
        raw_records,
        num_distances=layout.num_learnable_blocks,
        top_q=args.top_q,
    )
    successful_records = [
        record for record in raw_records if record.get("error") is None
    ]
    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "value_types": list(args.value_types),
        "needle_count": args.needle_count,
        "template_ids": list(args.template_ids),
        "coverage_round": args.coverage_round,
        "retrieval_k": args.retrieval_k,
        "q_to_kv_aggregation": f"top_{args.top_q}_mean_after_condition_average",
        "num_raw_records": len(raw_records),
        "num_successful_records": len(successful_records),
        "mean_teacher_token_accuracy": sum(
            float(record["teacher_token_accuracy"])
            for record in successful_records
        )
        / len(successful_records),
        "mean_teacher_nll": sum(
            float(record["teacher_mean_nll"])
            for record in successful_records
        )
        / len(successful_records),
    }
    save_aggregate(
        aggregate,
        output_dir=args.output_dir,
        metadata=metadata,
        min_distance=layout.min_learnable_distance,
    )
    print(
        f"raw={records_path}\n"
        f"tensor={args.output_dir / 'retrieval_scores.pt'}\n"
        f"csv={args.output_dir / 'kv_head_scores.csv'}\n"
        f"summary={args.output_dir / 'summary.json'}",
        flush=True,
    )
