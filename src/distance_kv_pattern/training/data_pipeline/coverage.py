"""Strict, balanced, without-replacement distance coverage."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ...core.randomness import derive_seed


@dataclass(frozen=True, slots=True)
class CoverageGroup:
    sample_index: int
    absolute_blocks: tuple[int, ...]


def build_coverage_groups(
    learnable_blocks: Iterable[int],
    needle_count: int,
    *,
    master_seed: int,
    namespace: tuple[object, ...],
    coverage_round: int,
) -> tuple[CoverageGroup, ...]:
    """Cover every block once, with the minimum balanced repeats if needed."""
    blocks = sorted(set(learnable_blocks))
    if not blocks:
        raise ValueError("learnable_blocks cannot be empty")
    if needle_count <= 0 or needle_count > len(blocks):
        raise ValueError("invalid needle_count")

    rng = random.Random(
        derive_seed(master_seed, *namespace, "coverage", coverage_round)
    )
    shuffled = blocks.copy()
    rng.shuffle(shuffled)

    num_samples = math.ceil(len(shuffled) / needle_count)
    full_prefix = (num_samples - 1) * needle_count
    groups = [
        shuffled[index : index + needle_count]
        for index in range(0, full_prefix, needle_count)
    ]
    final_group = shuffled[full_prefix:]
    missing = needle_count - len(final_group)
    if missing:
        # Filling the final group is unavoidable when the number of learnable
        # blocks is not divisible by needle_count.  Draw those repeats with an
        # independent deterministic RNG so they do not systematically favor
        # the farthest (lowest absolute-ID) blocks.
        repeat_rng = random.Random(
            derive_seed(
                master_seed,
                *namespace,
                "coverage_repeats",
                coverage_round,
            )
        )
        candidates = [block for block in blocks if block not in final_group]
        if len(candidates) < missing:
            raise AssertionError("not enough non-duplicate repeat candidates")
        final_group.extend(repeat_rng.sample(candidates, missing))
    groups.append(final_group)

    rng.shuffle(groups)
    result: list[CoverageGroup] = []
    for sample_index, group in enumerate(groups):
        rng.shuffle(group)
        if len(group) != needle_count or len(set(group)) != needle_count:
            raise AssertionError("coverage group has wrong size or duplicate blocks")
        result.append(CoverageGroup(sample_index, tuple(group)))

    validate_coverage(result, blocks, needle_count)
    return tuple(result)


def validate_coverage(
    groups: Iterable[CoverageGroup],
    expected_blocks: Iterable[int],
    needle_count: int,
) -> dict[str, int]:
    expected = set(expected_blocks)
    groups = tuple(groups)
    if not groups:
        raise ValueError("coverage groups cannot be empty")
    if any(len(group.absolute_blocks) != needle_count for group in groups):
        raise AssertionError("coverage group size mismatch")
    if any(len(set(group.absolute_blocks)) != needle_count for group in groups):
        raise AssertionError("a sample contains a duplicate target block")

    counts = Counter(block for group in groups for block in group.absolute_blocks)
    if set(counts) != expected:
        missing = sorted(expected - set(counts))
        extra = sorted(set(counts) - expected)
        raise AssertionError(f"coverage mismatch; missing={missing[:8]}, extra={extra[:8]}")
    minimum = min(counts.values())
    maximum = max(counts.values())
    if maximum - minimum > 1:
        raise AssertionError("coverage counts differ by more than one")
    return {
        "num_groups": len(groups),
        "num_unique_blocks": len(counts),
        "num_slots": sum(counts.values()),
        "min_count": minimum,
        "max_count": maximum,
        "extra_slots": sum(counts.values()) - len(expected),
    }
