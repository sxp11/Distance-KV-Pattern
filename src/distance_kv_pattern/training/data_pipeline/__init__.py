"""Deterministic compact-manifest training data pipeline."""

from .contracts import validate_tokenizer_contract
from .coverage import CoverageGroup, build_coverage_groups, validate_coverage
from .dataset import (
    IndependentQueryBranchExample,
    IndependentQueryDataset,
    IndependentQueryDatasetCollection,
    IndependentQueryTrainingExample,
    ManifestDataset,
    PatternTrainingExample,
    single_example_collate,
    single_independent_example_collate,
)
from .manifest import (
    ManifestPlanner,
    NeedlePlan,
    SamplePlan,
    read_manifest,
    validate_plans,
    write_manifest,
)
from .materialize import (
    IndependentMaterializedSample,
    IndependentQuerySuffix,
    InstructMaterializer,
    MaterializedSample,
    ValueTokenSpan,
)
from .templates import (
    ALL_TEMPLATE_IDS,
    HELDOUT_TEMPLATE_IDS,
    TRAIN_TEMPLATE_IDS,
    VALUE_TYPES,
)

__all__ = [
    "ALL_TEMPLATE_IDS",
    "CoverageGroup",
    "HELDOUT_TEMPLATE_IDS",
    "IndependentMaterializedSample",
    "IndependentQueryBranchExample",
    "IndependentQueryDataset",
    "IndependentQueryDatasetCollection",
    "IndependentQuerySuffix",
    "IndependentQueryTrainingExample",
    "InstructMaterializer",
    "ManifestDataset",
    "ManifestPlanner",
    "MaterializedSample",
    "NeedlePlan",
    "PatternTrainingExample",
    "SamplePlan",
    "TRAIN_TEMPLATE_IDS",
    "VALUE_TYPES",
    "ValueTokenSpan",
    "build_coverage_groups",
    "read_manifest",
    "single_example_collate",
    "single_independent_example_collate",
    "validate_coverage",
    "validate_plans",
    "validate_tokenizer_contract",
    "write_manifest",
]
