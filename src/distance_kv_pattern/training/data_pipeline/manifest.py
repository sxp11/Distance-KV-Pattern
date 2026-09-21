"""Compact, deterministic manifests for strict distance coverage."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .coverage import CoverageGroup, build_coverage_groups, validate_coverage
from ...core.layout import BlockLayout
from ...core.randomness import derive_seed
from .templates import (
    ALL_TEMPLATE_IDS,
    KEY_POOL,
    VALUE_TYPES,
    generate_values,
    render_answer,
    render_needle,
    render_query,
)


SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class NeedlePlan:
    key: str
    value: str
    text: str
    absolute_block: int
    relative_distance: int
    gate_index: int
    token_start: int
    token_end: int
    token_length: int


@dataclass(frozen=True, slots=True)
class SamplePlan:
    schema_version: int
    sample_id: str
    split: str
    value_type: str
    needle_count: int
    template_id: str
    coverage_round: int
    sample_index: int
    sample_seed: int
    query: str
    answer: str
    query_order: tuple[int, ...]
    query_keys: tuple[str, ...]
    answer_values: tuple[str, ...]
    needles: tuple[NeedlePlan, ...]
    tokenizer_name_or_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "SamplePlan":
        return cls(
            schema_version=int(record["schema_version"]),
            sample_id=str(record["sample_id"]),
            split=str(record["split"]),
            value_type=str(record["value_type"]),
            needle_count=int(record["needle_count"]),
            template_id=str(record["template_id"]),
            coverage_round=int(record["coverage_round"]),
            sample_index=int(record["sample_index"]),
            sample_seed=int(record["sample_seed"]),
            query=str(record["query"]),
            answer=str(record["answer"]),
            query_order=tuple(int(value) for value in record["query_order"]),
            query_keys=tuple(str(value) for value in record["query_keys"]),
            answer_values=tuple(str(value) for value in record["answer_values"]),
            needles=tuple(
                NeedlePlan(
                    key=str(item["key"]),
                    value=str(item["value"]),
                    text=str(item["text"]),
                    absolute_block=int(item["absolute_block"]),
                    relative_distance=int(item["relative_distance"]),
                    gate_index=int(item["gate_index"]),
                    token_start=int(item["token_start"]),
                    token_end=int(item["token_end"]),
                    token_length=int(item["token_length"]),
                )
                for item in record["needles"]
            ),
            tokenizer_name_or_path=str(record["tokenizer_name_or_path"]),
        )


class ManifestPlanner:
    def __init__(
        self,
        tokenizer: Any,
        *,
        layout: BlockLayout,
        master_seed: int,
        split: str,
    ) -> None:
        if not split:
            raise ValueError("split cannot be empty")
        self.tokenizer = tokenizer
        self.layout = layout
        self.master_seed = master_seed
        self.split = split
        self.tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))

    def coverage_groups(
        self,
        *,
        value_type: str,
        needle_count: int,
        template_id: str,
        coverage_round: int,
    ) -> tuple[CoverageGroup, ...]:
        return build_coverage_groups(
            self.layout.learnable_blocks,
            needle_count,
            master_seed=self.master_seed,
            namespace=(self.split, value_type, needle_count, template_id),
            coverage_round=coverage_round,
        )

    def _sample_plan(
        self,
        *,
        value_type: str,
        needle_count: int,
        template_id: str,
        coverage_round: int,
        group: CoverageGroup,
    ) -> SamplePlan:
        sample_seed = derive_seed(
            self.master_seed,
            self.split,
            value_type,
            needle_count,
            template_id,
            coverage_round,
            group.sample_index,
        )
        key_rng = random.Random(derive_seed(sample_seed, "keys"))
        value_rng = random.Random(derive_seed(sample_seed, "values"))
        assignment_rng = random.Random(derive_seed(sample_seed, "assignment"))
        query_rng = random.Random(derive_seed(sample_seed, "query_order"))

        keys = key_rng.sample(KEY_POOL, needle_count)
        values = list(generate_values(value_type, needle_count, value_rng))
        pairs = list(zip(keys, values, strict=True))
        assignment_rng.shuffle(pairs)

        needles: list[NeedlePlan] = []
        for needle_index, ((key, value), absolute_block) in enumerate(
            zip(pairs, group.absolute_blocks, strict=True)
        ):
            text = render_needle(value_type, key, value)
            token_length = len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
            if token_length > self.layout.usable_needle_tokens:
                raise ValueError(
                    f"needle {needle_index} needs {token_length} tokens; "
                    f"maximum is {self.layout.usable_needle_tokens}"
                )
            block_start, block_end = self.layout.block_span(absolute_block)
            minimum_start = block_start + self.layout.needle_margin_tokens
            maximum_start = (
                block_end - self.layout.needle_margin_tokens - token_length
            )
            offset_rng = random.Random(
                derive_seed(sample_seed, "needle_offset", needle_index)
            )
            token_start = offset_rng.randint(minimum_start, maximum_start)
            distance = self.layout.distance_block_id(absolute_block)
            needles.append(
                NeedlePlan(
                    key=key,
                    value=value,
                    text=text,
                    absolute_block=absolute_block,
                    relative_distance=distance,
                    gate_index=self.layout.gate_index(distance),
                    token_start=token_start,
                    token_end=token_start + token_length,
                    token_length=token_length,
                )
            )

        query_order = list(range(needle_count))
        query_rng.shuffle(query_order)
        query_keys = tuple(needles[index].key for index in query_order)
        answer_values = tuple(needles[index].value for index in query_order)
        query = render_query(template_id, value_type, query_keys)
        answer = render_answer(answer_values)
        sample_id = (
            f"{self.split}-{value_type}-n{needle_count}-{template_id.lower()}-"
            f"r{coverage_round:03d}-s{group.sample_index:04d}"
        )
        return SamplePlan(
            schema_version=SCHEMA_VERSION,
            sample_id=sample_id,
            split=self.split,
            value_type=value_type,
            needle_count=needle_count,
            template_id=template_id,
            coverage_round=coverage_round,
            sample_index=group.sample_index,
            sample_seed=sample_seed,
            query=query,
            answer=answer,
            query_order=tuple(query_order),
            query_keys=query_keys,
            answer_values=answer_values,
            needles=tuple(needles),
            tokenizer_name_or_path=self.tokenizer_name,
        )

    def iter_plans(
        self,
        *,
        value_type: str,
        needle_count: int,
        template_id: str,
        coverage_round: int,
    ) -> Iterator[SamplePlan]:
        if value_type not in VALUE_TYPES:
            raise ValueError(f"unknown value type: {value_type}")
        if needle_count not in {1, 2, 4, 8}:
            raise ValueError(f"unsupported needle count: {needle_count}")
        if template_id not in ALL_TEMPLATE_IDS:
            raise ValueError(f"unknown template: {template_id}")
        for group in self.coverage_groups(
            value_type=value_type,
            needle_count=needle_count,
            template_id=template_id,
            coverage_round=coverage_round,
        ):
            yield self._sample_plan(
                value_type=value_type,
                needle_count=needle_count,
                template_id=template_id,
                coverage_round=coverage_round,
                group=group,
            )


def validate_plans(
    plans: Sequence[SamplePlan],
    *,
    layout: BlockLayout,
    value_type: str,
    needle_count: int,
    template_id: str,
    coverage_round: int,
) -> dict[str, int]:
    if not plans:
        raise ValueError("manifest contains no samples")
    if len({plan.sample_id for plan in plans}) != len(plans):
        raise AssertionError("sample IDs are not unique")
    for plan in plans:
        if (
            plan.value_type != value_type
            or plan.needle_count != needle_count
            or plan.template_id != template_id
            or plan.coverage_round != coverage_round
        ):
            raise AssertionError("manifest mixes incompatible configurations")
        if plan.schema_version != SCHEMA_VERSION:
            raise AssertionError("unsupported sample schema version")
        if len(plan.needles) != needle_count:
            raise AssertionError("Q=N contract is violated")
        if len(plan.query_keys) != needle_count or len(plan.answer_values) != needle_count:
            raise AssertionError("query/answer count differs from needle_count")
        if tuple(sorted(plan.query_order)) != tuple(range(needle_count)):
            raise AssertionError("query_order is not a permutation of needle indices")
        if tuple(plan.needles[index].key for index in plan.query_order) != plan.query_keys:
            raise AssertionError("query key order is inconsistent")
        if tuple(plan.needles[index].value for index in plan.query_order) != plan.answer_values:
            raise AssertionError("answer value order is inconsistent")
        if len(set(needle.key for needle in plan.needles)) != needle_count:
            raise AssertionError("sample contains duplicate keys")
        if len(set(needle.value for needle in plan.needles)) != needle_count:
            raise AssertionError("sample contains duplicate values")
        if plan.query != render_query(template_id, value_type, plan.query_keys):
            raise AssertionError("serialized query is inconsistent")
        if plan.answer != render_answer(plan.answer_values):
            raise AssertionError("serialized answer is inconsistent")
        blocks = [needle.absolute_block for needle in plan.needles]
        if len(set(blocks)) != needle_count:
            raise AssertionError("sample contains duplicate target blocks")
        for needle in plan.needles:
            if needle.absolute_block not in layout.learnable_blocks:
                raise AssertionError("needle lies outside the learnable block range")
            expected_distance = layout.distance_block_id(needle.absolute_block)
            if needle.relative_distance != expected_distance:
                raise AssertionError("needle relative distance is inconsistent")
            if needle.gate_index != layout.gate_index(expected_distance):
                raise AssertionError("needle gate index is inconsistent")
            if needle.text != render_needle(value_type, needle.key, needle.value):
                raise AssertionError("serialized needle text is inconsistent")
            if needle.token_end - needle.token_start != needle.token_length:
                raise AssertionError("needle token span width is inconsistent")
            block_start, block_end = layout.block_span(needle.absolute_block)
            if needle.token_start < block_start + layout.needle_margin_tokens:
                raise AssertionError("needle violates the left block margin")
            if needle.token_end > block_end - layout.needle_margin_tokens:
                raise AssertionError("needle violates the right block margin")

    groups = tuple(
        CoverageGroup(index, tuple(needle.absolute_block for needle in plan.needles))
        for index, plan in enumerate(plans)
    )
    return validate_coverage(groups, layout.learnable_blocks, needle_count)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output_path: Path,
    plans: Iterable[SamplePlan],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(
                json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            count += 1
    final_metadata = {
        "schema_version": SCHEMA_VERSION,
        **metadata,
        "num_samples": count,
        "jsonl_sha256": _sha256_file(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(final_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_metadata


def read_manifest(path: Path) -> Iterator[SamplePlan]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield SamplePlan.from_dict(json.loads(line))
            except Exception as error:
                raise ValueError(f"invalid manifest row {path}:{line_number}") from error
