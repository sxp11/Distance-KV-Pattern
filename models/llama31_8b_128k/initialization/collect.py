#!/usr/bin/env python3
"""Collect Llama 3.1 retrieval-initialization scores on multiple GPUs."""

from __future__ import annotations

import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD_ROOT / "src"))
sys.path.insert(0, str(MODEL_ROOT))

from backend.retrieval_attention import score_branch
from distance_kv_pattern.training.retrieval_initialization.scripts.collect_retrieval_initialization_distributed import (
    main,
)


if __name__ == "__main__":
    main(score_branch_fn=score_branch)
