"""Shared packed-KV append primitive with no model or attention semantics."""

from __future__ import annotations

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def resolve_q_head_packed_update():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "q_head_packed_update.cu"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    extension_name = f"distance_kv_pattern_q_head_packed_update_{source_hash}"
    extension_root = Path(
        os.environ.get(
            "DISTANCE_KV_PATTERN_TORCH_EXTENSIONS_DIR",
            str(Path(tempfile.gettempdir()) / "distance_kv_pattern_torch_extensions"),
        )
    )
    build_directory = extension_root / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    extension = load(
        name=extension_name,
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )

    def update(packed, states, head_lengths, cu_seqlens):
        updated = extension.update_flatten_klenN_view(
            packed.squeeze(1),
            states,
            head_lengths.reshape(-1),
            cu_seqlens,
        )
        return updated.unsqueeze(1)

    return update
