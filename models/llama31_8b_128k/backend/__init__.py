"""Llama 3.1 model-specific training backend."""

from .suffix_runner import (
    SUPPORTED_TRANSFORMERS_VERSION,
    GatedSuffixRunner,
    KVHeadGatedSuffixRunner,
    QHeadGatedSuffixRunner,
    validate_transformers_runtime,
)

__all__ = [
    "SUPPORTED_TRANSFORMERS_VERSION",
    "GatedSuffixRunner",
    "KVHeadGatedSuffixRunner",
    "QHeadGatedSuffixRunner",
    "validate_transformers_runtime",
]
