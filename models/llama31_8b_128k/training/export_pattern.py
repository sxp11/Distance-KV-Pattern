#!/usr/bin/env python3
"""Export the one fixed-budget pattern owned by a formal STE+HC config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD_ROOT / "src"))

from distance_kv_pattern.inference.static_q_head.export_topk_pattern import (  # noqa: E402
    export_topk_pattern,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    keep_ratio = float(config["st_topk_keep_ratios"][0])
    budget = round(keep_ratio * 100)
    checkpoint = Path(config["output_dir"]) / "milestones/step_0044.pt"
    output = MODEL_ROOT / "patterns" / f"budget{budget}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    export_topk_pattern(str(checkpoint), str(output), keep_ratio)
