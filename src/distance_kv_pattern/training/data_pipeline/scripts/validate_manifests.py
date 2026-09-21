#!/usr/bin/env python3
"""Validate compact manifests, strict distance coverage and metadata hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from distance_kv_pattern.core import BlockLayout  # noqa: E402
from distance_kv_pattern.training.data_pipeline import (  # noqa: E402
    read_manifest,
    validate_plans,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest_root",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.manifest_root.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL manifests under {args.manifest_root}")
    total_samples = 0
    for path in paths:
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(path) != metadata["jsonl_sha256"]:
            raise AssertionError(f"manifest hash mismatch: {path}")
        layout_fields = {
            key: metadata["layout"][key]
            for key in (
                "total_tokens",
                "context_tokens",
                "block_size",
                "sink_blocks",
                "recent_blocks",
                "needle_margin_tokens",
            )
        }
        layout = BlockLayout(**layout_fields)
        plans = tuple(read_manifest(path))
        coverage = validate_plans(
            plans,
            layout=layout,
            value_type=metadata["value_type"],
            needle_count=int(metadata["needle_count"]),
            template_id=metadata["template_id"],
            coverage_round=int(metadata["coverage_round"]),
        )
        if coverage != metadata["coverage"]:
            raise AssertionError(f"coverage metadata mismatch: {path}")
        if len(plans) != int(metadata["num_samples"]):
            raise AssertionError(f"sample count metadata mismatch: {path}")
        total_samples += len(plans)
        print(f"ok {path}: {len(plans)} samples, {coverage}")
    print(f"validated {len(paths)} manifests and {total_samples} samples")


if __name__ == "__main__":
    main()
