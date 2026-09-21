"""Lazy manifest-backed dataset for post-prefill pattern training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ...core.layout import BlockLayout
from .manifest import SCHEMA_VERSION, SamplePlan, read_manifest
from .materialize import InstructMaterializer
from .contracts import tokenizer_names_compatible


@dataclass(frozen=True, slots=True)
class PatternTrainingExample:
    """One lazily materialized sample split at the fixed query anchor."""

    prompt_input_ids: torch.Tensor
    suffix_input_ids: torch.Tensor
    suffix_labels: torch.Tensor
    sample_id: str
    value_type: str
    needle_count: int
    template_id: str
    absolute_blocks: tuple[int, ...]
    relative_distances: tuple[int, ...]
    gate_indices: tuple[int, ...]
    query_order: tuple[int, ...]
    answer_values: tuple[str, ...]
    full_length: int
    suffix_length: int
    input_ids_sha256: str

    @property
    def suffix_loss_mask(self) -> torch.Tensor:
        """Derive the supervision mask from the single source of truth."""
        return self.suffix_labels.ne(-100)

    @property
    def num_supervised_tokens(self) -> int:
        return int(self.suffix_labels.ne(-100).sum().item())




@dataclass(frozen=True, slots=True)
class IndependentQueryBranchExample:
    """One single-key suffix derived from a shared multi-needle prompt."""

    suffix_input_ids: torch.Tensor
    suffix_labels: torch.Tensor
    query_position: int
    needle_index: int
    key: str
    value: str
    relative_distance: int
    gate_index: int
    suffix_length: int
    input_ids_sha256: str

    @property
    def suffix_loss_mask(self) -> torch.Tensor:
        return self.suffix_labels.ne(-100)

    @property
    def num_supervised_tokens(self) -> int:
        return int(self.suffix_labels.ne(-100).sum().item())


@dataclass(frozen=True, slots=True)
class IndependentQueryTrainingExample:
    """One shared prompt plus one independently supervised branch per needle."""

    prompt_input_ids: torch.Tensor
    branches: tuple[IndependentQueryBranchExample, ...]
    sample_id: str
    value_type: str
    needle_count: int
    template_id: str
    absolute_blocks: tuple[int, ...]
    relative_distances: tuple[int, ...]
    gate_indices: tuple[int, ...]
    query_order: tuple[int, ...]
    prompt_ids_sha256: str



    @property
    def num_supervised_tokens(self) -> int:
        return sum(branch.num_supervised_tokens for branch in self.branches)
class ManifestDataset(Dataset[PatternTrainingExample]):
    """Load compact plans and materialize full token arrays only on demand."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: Any,
        *,
        layout: BlockLayout | None = None,
        materializer: InstructMaterializer | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.layout = layout if layout is not None else BlockLayout()
        self.plans = tuple(read_manifest(self.manifest_path))
        if not self.plans:
            raise ValueError(f"manifest contains no samples: {self.manifest_path}")

        schema_versions = {plan.schema_version for plan in self.plans}
        if schema_versions != {SCHEMA_VERSION}:
            raise ValueError(
                "manifest schema differs from the active data contract: "
                f"{sorted(schema_versions)} != {[SCHEMA_VERSION]} "
                f"for {self.manifest_path}"
            )

        if materializer is None:
            materializer = InstructMaterializer(tokenizer, layout=self.layout)
        elif materializer.layout != self.layout:
            raise ValueError("materializer layout differs from dataset layout")
        self.materializer = materializer

        tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))
        mismatched = {
            plan.tokenizer_name_or_path
            for plan in self.plans
            if not tokenizer_names_compatible(
                plan.tokenizer_name_or_path, tokenizer_name
            )
        }
        if mismatched:
            raise ValueError(
                "manifest tokenizer is not compatible with dataset tokenizer: "
                f"{sorted(mismatched)!r} != {tokenizer_name!r}"
            )

    def __len__(self) -> int:
        return len(self.plans)

    def plan(self, index: int) -> SamplePlan:
        """Return a compact plan without materializing token arrays."""
        return self.plans[index]

    def __getitem__(self, index: int) -> PatternTrainingExample:
        sample = self.materializer.materialize(self.plans[index])
        anchor = self.layout.context_tokens
        if sample.query_start != anchor:
            raise AssertionError(
                f"query starts at {sample.query_start}, expected fixed anchor {anchor}"
            )

        prompt_ids = sample.input_ids[:anchor]
        suffix_ids = sample.input_ids[anchor:]
        suffix_labels = sample.labels[anchor:]
        expected_supervised = sum(
            span.answer_token_end - span.answer_token_start
            for span in sample.value_token_spans
        )

        if len(prompt_ids) != anchor:
            raise AssertionError("prompt split does not match the fixed anchor")
        if len(suffix_ids) != sample.suffix_length:
            raise AssertionError("suffix split length differs from materializer")
        if len(suffix_ids) > self.layout.suffix_capacity_tokens:
            raise AssertionError("suffix exceeds the configured capacity")
        num_supervised = sum(label != -100 for label in suffix_labels)
        if num_supervised != expected_supervised:
            raise AssertionError(
                "unexpected number of supervised suffix tokens: "
                f"{num_supervised} != {expected_supervised}"
            )
        if any(label != -100 for label in sample.labels[:anchor]):
            raise AssertionError("prompt unexpectedly contains supervised tokens")

        needles = sample.plan.needles
        return PatternTrainingExample(
            prompt_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
            suffix_input_ids=torch.tensor(suffix_ids, dtype=torch.long),
            suffix_labels=torch.tensor(suffix_labels, dtype=torch.long),
            sample_id=sample.plan.sample_id,
            value_type=sample.plan.value_type,
            needle_count=sample.plan.needle_count,
            template_id=sample.plan.template_id,
            absolute_blocks=tuple(needle.absolute_block for needle in needles),
            relative_distances=tuple(
                needle.relative_distance for needle in needles
            ),
            gate_indices=tuple(needle.gate_index for needle in needles),
            query_order=sample.plan.query_order,
            answer_values=sample.plan.answer_values,
            full_length=sample.full_length,
            suffix_length=sample.suffix_length,
            input_ids_sha256=sample.input_ids_sha256,
        )




class IndependentQueryDataset(Dataset[IndependentQueryTrainingExample]):
    """Materialize one shared prompt and N independently supervised suffixes."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: Any,
        *,
        layout: BlockLayout | None = None,
        materializer: InstructMaterializer | None = None,
    ) -> None:
        source = ManifestDataset(
            manifest_path,
            tokenizer,
            layout=layout,
            materializer=materializer,
        )
        self.manifest_path = source.manifest_path
        self.layout = source.layout
        self.plans = source.plans
        self.materializer = source.materializer

    def __len__(self) -> int:
        return len(self.plans)

    def plan(self, index: int) -> SamplePlan:
        return self.plans[index]

    def __getitem__(self, index: int) -> IndependentQueryTrainingExample:
        sample = self.materializer.materialize_independent_queries(
            self.plans[index]
        )
        if len(sample.prompt_input_ids) != self.layout.context_tokens:
            raise AssertionError("shared prompt does not match the fixed anchor")
        if len(sample.suffixes) != sample.plan.needle_count:
            raise AssertionError("branch count differs from needle count")

        branches: list[IndependentQueryBranchExample] = []
        for branch in sample.suffixes:
            if branch.suffix_length != len(branch.input_ids):
                raise AssertionError("branch suffix length is inconsistent")
            if branch.suffix_length > self.layout.suffix_capacity_tokens:
                raise AssertionError("branch suffix exceeds configured capacity")
            if len(branch.labels) != branch.suffix_length:
                raise AssertionError("branch labels and input IDs differ in length")
            if tuple(label != -100 for label in branch.labels) != branch.loss_mask:
                raise AssertionError("branch labels and loss mask disagree")
            if any(
                label != -100 and label != token_id
                for label, token_id in zip(
                    branch.labels,
                    branch.input_ids,
                    strict=True,
                )
            ):
                raise AssertionError("supervised label differs from suffix token")
            needle = sample.plan.needles[branch.needle_index]
            if (
                branch.key != needle.key
                or branch.value != needle.value
                or branch.relative_distance != needle.relative_distance
                or branch.gate_index != needle.gate_index
            ):
                raise AssertionError("branch metadata does not match its needle")
            branches.append(
                IndependentQueryBranchExample(
                    suffix_input_ids=torch.tensor(
                        branch.input_ids,
                        dtype=torch.long,
                    ),
                    suffix_labels=torch.tensor(
                        branch.labels,
                        dtype=torch.long,
                    ),
                    query_position=branch.query_position,
                    needle_index=branch.needle_index,
                    key=branch.key,
                    value=branch.value,
                    relative_distance=branch.relative_distance,
                    gate_index=branch.gate_index,
                    suffix_length=branch.suffix_length,
                    input_ids_sha256=branch.input_ids_sha256,
                )
            )

        needles = sample.plan.needles
        return IndependentQueryTrainingExample(
            prompt_input_ids=torch.tensor(
                sample.prompt_input_ids,
                dtype=torch.long,
            ),
            branches=tuple(branches),
            sample_id=sample.plan.sample_id,
            value_type=sample.plan.value_type,
            needle_count=sample.plan.needle_count,
            template_id=sample.plan.template_id,
            absolute_blocks=tuple(needle.absolute_block for needle in needles),
            relative_distances=tuple(
                needle.relative_distance for needle in needles
            ),
            gate_indices=tuple(needle.gate_index for needle in needles),
            query_order=sample.plan.query_order,
            prompt_ids_sha256=sample.prompt_ids_sha256,
        )


class IndependentQueryDatasetCollection(
    Dataset[IndependentQueryTrainingExample]
):
    """Lazy concatenation of homogeneous formal manifests.

    Each child manifest keeps its own compact plans and materializer, while
    the collection presents one deterministic context stream to the formal
    trainer.  This lets an optimizer step cover (for example) num+word and
    Q1+Q3 without materializing all 128K prompts in memory.
    """

    def __init__(
        self,
        manifest_paths: Sequence[str | Path],
        tokenizer: Any,
        *,
        layout: BlockLayout | None = None,
        materializer: InstructMaterializer | None = None,
    ) -> None:
        paths = tuple(Path(path) for path in manifest_paths)
        if not paths:
            raise ValueError("at least one manifest path is required")
        shared_layout = layout if layout is not None else BlockLayout()
        self.datasets = tuple(
            IndependentQueryDataset(
                path,
                tokenizer,
                layout=shared_layout,
                materializer=materializer,
            )
            for path in paths
        )
        self.manifest_paths = paths
        self.layout = shared_layout
        self.offsets: tuple[int, ...] = tuple(
            sum(len(dataset) for dataset in self.datasets[:index])
            for index in range(len(self.datasets) + 1)
        )
        self.plans = tuple(
            plan for dataset in self.datasets for plan in dataset.plans
        )

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for dataset_index in range(len(self.datasets)):
            start, stop = self.offsets[dataset_index : dataset_index + 2]
            if start <= index < stop:
                return dataset_index, index - start
        raise AssertionError("collection offset lookup failed")

    def plan(self, index: int) -> SamplePlan:
        dataset_index, local_index = self._locate(index)
        return self.datasets[dataset_index].plan(local_index)

    def __getitem__(self, index: int) -> IndependentQueryTrainingExample:
        dataset_index, local_index = self._locate(index)
        return self.datasets[dataset_index][local_index]


def single_independent_example_collate(
    examples: Sequence[IndependentQueryTrainingExample],
) -> IndependentQueryTrainingExample:
    """Return one shared-prompt sample and reject accidental larger batches."""
    if len(examples) != 1:
        raise ValueError(
            "the first 128K implementation requires batch_size=1; "
            f"received {len(examples)} samples"
        )
    return examples[0]
def single_example_collate(
    examples: Sequence[PatternTrainingExample],
) -> PatternTrainingExample:
    """Return one sample unchanged and reject accidental larger batches."""
    if len(examples) != 1:
        raise ValueError(
            "the first 128K implementation requires batch_size=1; "
            f"received {len(examples)} samples"
        )
    return examples[0]
