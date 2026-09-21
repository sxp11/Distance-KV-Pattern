#!/usr/bin/env python3
"""Generate Llama2-32K manifests with its tokenizer adapters."""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD_ROOT / "src"))
sys.path.insert(0, str(MODEL_ROOT))

from backend.data_adapter import validate_llama2_tokenizer_contract
from backend.model_loading import load_llama2_tokenizer
from distance_kv_pattern.training.data_pipeline.scripts.generate_manifests import (
    main,
)

if __name__ == "__main__":
    main(
        tokenizer_loader=load_llama2_tokenizer,
        contract_validator=validate_llama2_tokenizer_contract,
    )
