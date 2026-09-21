"""Convert checkpoint gate scores to a fixed global Top-K 0/1 pattern."""

from __future__ import annotations

import argparse

import torch


def export_topk_pattern(
    input_checkpoint: str,
    output_pattern: str,
    keep_ratio: float,
) -> None:
    checkpoint = torch.load(
        input_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    scores = checkpoint["keep_probability"].float()
    num_keep = round(scores.numel() * keep_ratio)

    indices = torch.argsort(
        scores.view(-1),
        descending=True,
        stable=True,
    )[:num_keep]

    pattern = torch.zeros_like(
        scores,
        dtype=torch.uint8,
    )
    pattern.view(-1)[indices] = 1

    torch.save(
        {
            "pattern": pattern,
            "keep_ratio": keep_ratio,
            "source_checkpoint": input_checkpoint,
        },
        output_pattern,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint")
    parser.add_argument("output_pattern")
    parser.add_argument("--keep-ratio", type=float, required=True)

    args = parser.parse_args()

    export_topk_pattern(
        args.input_checkpoint,
        args.output_pattern,
        args.keep_ratio,
    )
