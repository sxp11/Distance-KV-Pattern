import sys
from pathlib import Path

import torch

import distance_kv_pattern

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.run_llama31_inference import materialize_pattern


EXPECTED_PATTERN_SHAPES = {
    "llama2_7b_32k": (32, 32, 245),
    "llama31_8b_128k": (32, 32, 1013),
    "qwen25_7b_128k": (28, 28, 1013),
}


def test_package_imports() -> None:
    assert distance_kv_pattern.BlockLayout is not None


def test_public_patterns() -> None:
    for model, expected_shape in EXPECTED_PATTERN_SHAPES.items():
        path = REPOSITORY_ROOT / "models" / model / "patterns" / "budget20.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert set(payload) == {"pattern", "keep_ratio"}
        assert tuple(payload["pattern"].shape) == expected_shape
        assert payload["pattern"].dtype == torch.uint8
        assert float(payload["keep_ratio"]) == 0.2


def test_materialize_pattern_uses_relative_distance_blocks() -> None:
    distance_pattern = torch.tensor([[[True, False]]])
    mask = materialize_pattern(distance_pattern, context_length=1408)

    assert tuple(mask.shape) == (1, 1, 1, 1408)
    assert mask[0, 0, 0].tolist() == (
        [True] * 128
        + [False] * 128
        + [True] * 128
        + [True] * 1024
    )
