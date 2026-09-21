"""Stable, namespace-separated random seeds and process reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed every process-wide RNG used by project experiments.

    Must be called before model construction/DataLoader creation.  This helper
    only seeds process RNGs and does not alter backend/kernel selection.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

import hashlib


def derive_seed(master_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
