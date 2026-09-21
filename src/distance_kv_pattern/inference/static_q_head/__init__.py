"""Query-boundary static Q-head KV compaction."""

from .llama2_q_head_inference import (
    Llama2QHeadCache,
    install_llama2_q_head_attention,
)
from .llama31_q_head_inference import (
    Llama31QHeadCache,
    install_llama31_q_head_attention,
)
from .qwen25_q_head_inference import (
    Qwen25QHeadCache,
    install_qwen25_q_head_attention,
)

__all__ = [
    "Llama2QHeadCache",
    "Llama31QHeadCache",
    "Qwen25QHeadCache",
    "install_llama2_q_head_attention",
    "install_llama31_q_head_attention",
    "install_qwen25_q_head_attention",
]
