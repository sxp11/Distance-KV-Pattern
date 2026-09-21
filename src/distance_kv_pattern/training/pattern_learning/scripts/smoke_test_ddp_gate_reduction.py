#!/usr/bin/env python3
"""Verify official DDP gate reduction and process communication.

CPU example:

    torchrun --standalone --nproc-per-node=2 scripts/smoke_test_ddp_gate_reduction.py

The test covers process/device binding, a moderate-size collective, the two
nontrivial reduction semantics used by formal training, and replica equality
after one optimizer step. It intentionally does not load an LLM or dataset.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


METHOD_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from distance_kv_pattern import (  # noqa: E402
    QHeadHardConcreteGates,
    build_context_shards,
    ddp_task_loss_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--expected-world-size", type=int, default=None)
    parser.add_argument("--collective-elements", type=int, default=1_048_576)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--nondiv-contexts", type=int, default=10)
    return parser.parse_args()


def sample_spec(
    rank: int,
    world_size: int,
) -> tuple[tuple[torch.Tensor, float, torch.Tensor], ...]:
    """Return two deterministic local samples with globally normalized weights."""

    rank_fraction = (rank + 1) / (world_size + 1)
    base = 0.05 + 0.25 * rank_fraction
    number_of_samples = 2 * world_size
    weight_denominator = number_of_samples * (number_of_samples + 1) / 2
    first_weight = (2 * rank + 1) / weight_denominator
    second_weight = (2 * rank + 2) / weight_denominator
    return (
        (
            torch.tensor([[[base, base + 0.21, base + 0.47]]]),
            first_weight,
            torch.tensor([[[0.7 + 0.03 * rank, -0.2, 0.5]]]),
        ),
        (
            torch.tensor([[[base + 0.08, base + 0.31, base + 0.59]]]),
            second_weight,
            torch.tensor([[[-0.3, 0.4 + 0.02 * rank, 0.8]]]),
        ),
    )


def main() -> None:
    args = parse_args()
    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        timeout=timedelta(minutes=5),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("this smoke test requires at least two torchrun processes")
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise ValueError(
            f"expected {args.expected_world_size} processes, received {world_size}"
        )

    if args.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
    else:
        device = torch.device("cpu")
        ddp_kwargs = {}

    # The default tensor is 4 MiB in fp32: enough to catch NCCL/setup failures,
    # but tiny compared with model training.
    collective = torch.full(
        (args.collective_elements,),
        float(rank + 1),
        device=device,
    )
    dist.all_reduce(collective, op=dist.ReduceOp.SUM)
    expected_collective = world_size * (world_size + 1) / 2
    if not bool(torch.all(collective == expected_collective).item()):
        raise AssertionError("distributed all_reduce produced an incorrect value")

    if args.backend == "nccl":
        binding = torch.tensor([local_rank, torch.cuda.current_device()], device=device)
        bindings = [torch.empty_like(binding) for _ in range(world_size)]
        dist.all_gather(bindings, binding)
        local_ranks = sorted(int(item[0].item()) for item in bindings)
        if local_ranks != list(range(world_size)):
            raise AssertionError(f"invalid local-rank/GPU binding: {local_ranks}")

    gates = QHeadHardConcreteGates(
        1,
        1,
        3,
        initial_keep_probability=0.7,
    ).to(device)
    ddp_gates = DistributedDataParallel(
        gates,
        broadcast_buffers=False,
        find_unused_parameters=False,
        **ddp_kwargs,
    )
    specs = sample_spec(rank, world_size)
    scale = ddp_task_loss_scale(world_size)
    l0_coefficient = 0.3
    regularization_loss = l0_coefficient * gates.expected_l0("mean")

    ddp_gates.zero_grad(set_to_none=True)
    for index, (uniform, weight, coefficient) in enumerate(specs):
        uniform = uniform.to(device)
        coefficient = coefficient.to(device)
        final = index == len(specs) - 1
        context = torch.enable_grad() if final else ddp_gates.no_sync()
        with context:
            gate = ddp_gates(uniform)
            loss = (gate * coefficient).sum() * weight * scale
            if final:
                loss = loss + regularization_loss
            loss.backward()
    distributed_gradient = gates.log_alpha.grad.detach().clone()

    reference = QHeadHardConcreteGates(
        1,
        1,
        3,
        initial_keep_probability=0.7,
    ).to(device)
    reference.zero_grad(set_to_none=True)
    reference_loss = torch.zeros((), device=device)
    for source_rank in range(world_size):
        for uniform, weight, coefficient in sample_spec(source_rank, world_size):
            reference_loss = reference_loss + (
                reference(uniform.to(device)) * coefficient.to(device)
            ).sum() * weight
    reference_loss = reference_loss + l0_coefficient * reference.expected_l0("mean")
    reference_loss.backward()
    expected_gradient = reference.log_alpha.grad.detach()

    if not torch.allclose(
        distributed_gradient,
        expected_gradient,
        atol=args.atol,
        rtol=args.rtol,
    ):
        raise AssertionError(
            "DDP gradient differs from single-process reference: "
            f"rank={rank}, distributed={distributed_gradient.cpu().tolist()}, "
            f"expected={expected_gradient.cpu().tolist()}"
        )

    maximum_error = (distributed_gradient - expected_gradient).abs().max()
    dist.all_reduce(maximum_error, op=dist.ReduceOp.MAX)

    optimizer = torch.optim.AdamW(
        ddp_gates.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )
    optimizer.step()
    rank_zero_parameter = gates.log_alpha.detach().clone()
    dist.broadcast(rank_zero_parameter, src=0)
    parameter_error = (gates.log_alpha.detach() - rank_zero_parameter).abs().max()
    dist.all_reduce(parameter_error, op=dist.ReduceOp.MAX)
    if float(parameter_error.item()) > args.atol:
        raise AssertionError(
            "DDP replicas diverged after optimizer step: "
            f"max_abs_error={float(parameter_error.item()):.3e}"
        )
    if rank == 0:
        print("stage=nondivisible_start", flush=True)

    # Regression test for the formal training loop: a non-divisible context
    # count must still produce the same number of DDP backward slots on every
    # rank.  Padded slots execute a zero-loss backward so the final collective
    # happens at the same loop position everywhere.
    if args.nondiv_contexts < world_size:
        raise ValueError("--nondiv-contexts must be >= world size")
    padded_gates = QHeadHardConcreteGates(
        1,
        1,
        3,
        initial_keep_probability=0.7,
    ).to(device)
    padded_ddp = DistributedDataParallel(
        padded_gates,
        broadcast_buffers=False,
        find_unused_parameters=False,
        **ddp_kwargs,
    )
    padded_optimizer = torch.optim.SGD(padded_ddp.parameters(), lr=1e-2)
    shard = build_context_shards(
        args.nondiv_contexts,
        world_size=world_size,
    )[rank]
    if len(shard.slot_offsets) != max(
        len(item.slot_offsets)
        for item in build_context_shards(
            args.nondiv_contexts,
            world_size=world_size,
        )
    ):
        raise AssertionError("non-divisible shards do not share slot count")
    padded_ddp.zero_grad(set_to_none=True)
    padded_scale = ddp_task_loss_scale(world_size)
    for slot, global_offset in enumerate(shard.slot_offsets):
        final_slot = slot == len(shard.slot_offsets) - 1
        context = nullcontext() if final_slot else padded_ddp.no_sync()
        with context:
            uniform = torch.full(
                padded_gates.shape,
                0.35 + 0.01 * (global_offset or 0),
                dtype=torch.float32,
                device=device,
            )
            gate = padded_ddp(uniform)
            if global_offset is None:
                slot_loss = gate.sum() * 0.0
            else:
                coefficient = torch.full_like(gate, float(global_offset + 1))
                slot_loss = (gate * coefficient).sum() * padded_scale
            slot_loss.backward()
        if rank == 0:
            print(
                f"stage=nondivisible_slot_done slot={slot} "
                f"padded={global_offset is None}",
                flush=True,
            )
    padded_gradient = padded_gates.log_alpha.grad.detach()
    if not torch.isfinite(padded_gradient).all():
        raise AssertionError("non-divisible padded gradient is non-finite")
    padded_optimizer.step()
    padded_parameter_error = (
        padded_gates.log_alpha.detach().clone()
    )
    dist.broadcast(padded_parameter_error, src=0)
    padded_parameter_error = (
        padded_gates.log_alpha.detach() - padded_parameter_error
    ).abs().max()
    dist.all_reduce(padded_parameter_error, op=dist.ReduceOp.MAX)
    if float(padded_parameter_error.item()) > args.atol:
        raise AssertionError(
            "non-divisible padded DDP replicas diverged: "
            f"max_abs_error={float(padded_parameter_error.item()):.3e}"
        )
    if rank == 0:
        print("stage=nondivisible_done", flush=True)

    if rank == 0:
        accelerator = (
            torch.cuda.get_device_name(0) if args.backend == "nccl" else "cpu"
        )
        print(
            "distributed_smoke=passed "
            f"backend={args.backend} world_size={world_size} "
            f"accelerator={accelerator!r} collective=passed "
            f"ddp_gradient=passed optimizer_sync=passed "
            f"gradient_max_abs_error={float(maximum_error.item()):.3e} "
            f"parameter_max_abs_error={float(parameter_error.item()):.3e} "
            f"nondivisible_slots=passed contexts={args.nondiv_contexts} "
            f"padded_parameter_max_abs_error={float(padded_parameter_error.item()):.3e}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
