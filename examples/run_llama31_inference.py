#!/usr/bin/env python3
"""Minimal Llama 3.1 generation with a Distance-KV pattern."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from distance_kv_pattern.inference.static_q_head import (
    Llama31QHeadCache,
    install_llama31_q_head_attention,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN = (
    REPOSITORY_ROOT
    / "models"
    / "llama31_8b_128k"
    / "patterns"
    / "budget20.pt"
)
MAX_CONTEXT_TOKENS = 130_816
MAX_TOTAL_TOKENS = 131_072
BLOCK_SIZE = 128
RECENT_BLOCKS = 8


def materialize_pattern(
    distance_pattern: torch.Tensor,
    context_length: int,
) -> torch.Tensor:
    """Map a distance-block pattern to a token-level Q-head keep mask."""
    if distance_pattern.ndim != 3:
        raise ValueError("distance_pattern must have shape [layers, heads, blocks]")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    distance_pattern = distance_pattern.bool()
    recent_tokens = RECENT_BLOCKS * BLOCK_SIZE
    token_pattern = torch.zeros(
        distance_pattern.shape[0],
        distance_pattern.shape[1],
        context_length,
        dtype=torch.bool,
        device=distance_pattern.device,
    )
    token_index = torch.arange(context_length, device=distance_pattern.device)
    learnable = (token_index >= BLOCK_SIZE) & (
        token_index < context_length - recent_tokens
    )

    if learnable.any():
        gate_index = (
            torch.div(
                context_length - 1 - token_index[learnable],
                BLOCK_SIZE,
                rounding_mode="floor",
            )
            - RECENT_BLOCKS
        )
        gate_min = int(gate_index.min())
        gate_max = int(gate_index.max())
        if gate_min < 0 or gate_max >= distance_pattern.shape[2]:
            raise ValueError("context length is outside the supplied pattern layout")
        token_pattern[:, :, learnable] = distance_pattern.index_select(2, gate_index)

    token_pattern[:, :, :BLOCK_SIZE] = True
    token_pattern[:, :, -recent_tokens:] = True
    return token_pattern.unsqueeze(1)


def get_query_start(
    prompt: str,
    query_char_start: int,
    input_ids: list[int],
    tokenizer,
) -> int:
    prefix_ids = tokenizer(
        prompt[:query_char_start],
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    query_start = 0
    for full_token, prefix_token in zip(input_ids, prefix_ids):
        if full_token != prefix_token:
            break
        query_start += 1
    return query_start


def build_inputs(tokenizer, context: str, query: str, device: torch.device):
    query = query.strip()
    if not context.strip() or not query:
        raise ValueError("context and query must both be non-empty")

    query_anchor = "\n\nQuestion:\n"
    user_content = f"{context.rstrip()}{query_anchor}{query}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    query_char_start = prompt.rfind(query_anchor)
    if query_char_start < 0:
        raise ValueError("query boundary was not found in the rendered prompt")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    query_start = get_query_start(
        prompt,
        query_char_start,
        inputs["input_ids"][0].tolist(),
        tokenizer,
    )
    if not 0 < query_start < inputs["input_ids"].shape[1]:
        raise ValueError("invalid tokenized query boundary")
    return {name: value.to(device) for name, value in inputs.items()}, query_start


@torch.inference_mode()
def distance_kv_generate(
    model,
    inputs: dict[str, torch.Tensor],
    query_start: int,
    distance_pattern: torch.Tensor,
    **generation_kwargs,
) -> torch.Tensor:
    dense_cache = DynamicCache()
    context_outputs = model(
        input_ids=inputs["input_ids"][:, :query_start],
        attention_mask=inputs["attention_mask"][:, :query_start],
        past_key_values=dense_cache,
        use_cache=True,
        num_logits_to_keep=1,
    )
    if context_outputs.past_key_values is not dense_cache:
        raise RuntimeError("model did not populate the requested DynamicCache")

    compact_cache = Llama31QHeadCache.from_dynamic_cache(
        dense_cache,
        keep_mask=materialize_pattern(distance_pattern, query_start),
        logical_context_length=query_start,
    )
    return model.generate(
        **inputs,
        past_key_values=compact_cache,
        **generation_kwargs,
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--model-name-or-path",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the static Q-head inference path requires a CUDA device")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=args.local_files_only,
    ).to(device).eval()
    install_llama31_q_head_attention(model)

    payload = torch.load(args.pattern, map_location=device, weights_only=True)
    if set(payload) != {"pattern", "keep_ratio"}:
        raise ValueError("pattern file must contain only 'pattern' and 'keep_ratio'")
    distance_pattern = payload["pattern"].bool()
    if tuple(distance_pattern.shape) != (32, 32, 1013):
        raise ValueError("expected a Llama 3.1 pattern with shape [32, 32, 1013]")

    context = args.context_file.read_text(encoding="utf-8")
    inputs, query_start = build_inputs(tokenizer, context, args.query, device)
    input_length = inputs["input_ids"].shape[1]
    if query_start > MAX_CONTEXT_TOKENS:
        raise ValueError(f"query boundary exceeds {MAX_CONTEXT_TOKENS} context tokens")
    if input_length + args.max_new_tokens > MAX_TOTAL_TOKENS:
        raise ValueError(f"prompt plus generation exceeds {MAX_TOTAL_TOKENS} tokens")

    output_ids = distance_kv_generate(
        model,
        inputs,
        query_start,
        distance_pattern,
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[input_length:]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
