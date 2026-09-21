#!/usr/bin/env python3
"""Llama2-32K entrypoint for the common distributed trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD_ROOT / "src"))
sys.path.insert(0, str(MODEL_ROOT))

from backend.data_adapter import (
    Llama2InstructMaterializer,
    build_llama2_layout,
)
from backend.model_loading import (
    load_llama2_model,
    load_llama2_tokenizer,
)
from backend.suffix_runner import (
    QHeadGatedSuffixRunner,
    validate_transformers_runtime,
)
from distance_kv_pattern.training.pattern_learning.scripts.train_q_head_pattern_distributed import (
    args_from_config,
    main,
)


def parse_entry_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    entry_args = parse_entry_args()
    training_args = args_from_config(entry_args.config)
    training_args.dry_run = entry_args.dry_run
    main(
        runner_factory=QHeadGatedSuffixRunner,
        runtime_validator=validate_transformers_runtime,
        model_loader=load_llama2_model,
        tokenizer_loader=load_llama2_tokenizer,
        layout_factory=build_llama2_layout,
        materializer_factory=Llama2InstructMaterializer,
        args=training_args,
    )
