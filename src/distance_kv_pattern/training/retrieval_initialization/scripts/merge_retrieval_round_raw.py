#!/usr/bin/env python3
"""Deterministically merge raw retrieval JSONL files from distinct rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_jsonl",
        nargs="+",
        type=Path,
        help="Input files in the desired deterministic round order.",
    )
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def validate_file(
    path: Path,
    *,
    branch_ids: set[str],
    stats: dict[str, Any],
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            branch_id = str(record["branch_id"])
            if branch_id in branch_ids:
                raise ValueError(f"duplicate branch_id across inputs: {branch_id}")
            branch_ids.add(branch_id)
            stats["num_records"] += 1
            if record.get("error") is not None:
                stats["num_failed_records"] += 1
            else:
                stats["num_successful_records"] += 1
            stats["coverage_rounds"].add(int(record["coverage_round"]))


def main() -> None:
    args = parse_args()
    inputs = tuple(path.resolve() for path in args.raw_jsonl)
    output = args.output.resolve()
    if len(set(inputs)) != len(inputs):
        raise ValueError("input files must be distinct")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if output in inputs:
        raise ValueError("output must differ from every input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    branch_ids: set[str] = set()
    stats: dict[str, Any] = {
        "num_records": 0,
        "num_successful_records": 0,
        "num_failed_records": 0,
        "coverage_rounds": set(),
    }
    for path in inputs:
        validate_file(path, branch_ids=branch_ids, stats=stats)
    if stats["num_failed_records"]:
        raise RuntimeError(
            f"refusing to merge {stats['num_failed_records']} failed records"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for path in inputs:
                with path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            destination.write(line)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "inputs": [str(path) for path in inputs],
                "coverage_rounds": sorted(stats["coverage_rounds"]),
                "num_records": stats["num_records"],
                "num_successful_records": stats["num_successful_records"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
