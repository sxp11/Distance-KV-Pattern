"""Load the local Llama-2-7B-32K-Instruct model and tokenizer."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST]\\n' + message['content'] + '\\n[/INST]\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_llama2_tokenizer(model_name_or_path: str, **kwargs: Any):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    return tokenizer


def load_llama2_model(model_name_or_path: str, **kwargs: Any):
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
