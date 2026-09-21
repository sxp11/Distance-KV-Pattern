#!/usr/bin/env python3
"""Print the public metadata and tensor shape of a learned pattern."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.pattern, map_location="cpu", weights_only=True)
    if set(payload) != {"pattern", "keep_ratio"}:
        raise ValueError("unexpected pattern payload")

    pattern = payload["pattern"]
    if pattern.ndim != 3:
        raise ValueError("pattern must have shape [layers, query_heads, distances]")

    print(
        json.dumps(
            {
                "path": str(args.pattern),
                "shape": list(pattern.shape),
                "dtype": str(pattern.dtype),
                "keep_ratio": float(payload["keep_ratio"]),
                "kept_entries": int(pattern.sum().item()),
                "total_entries": pattern.numel(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
