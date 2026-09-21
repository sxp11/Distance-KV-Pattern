"""Load Qwen2.5-7B-Instruct with the official 128K YaRN configuration."""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_qwen25_model(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool,
    local_files_only: bool,
    low_cpu_mem_usage: bool,
):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    config.rope_scaling = {"factor": 4.0, "type": "yarn"}
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
