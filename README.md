# Distance-KV-Pattern

Distance-KV-Pattern learns static
`layer × query-head × relative-distance-block` patterns for retaining KV cache
entries at a query boundary.

This repository contains the shared training implementation, model-specific
backends, static Q-head inference components, and learned patterns for:

- Llama 3.1 8B Instruct (128K)
- Qwen2.5 7B Instruct (128K)
- Llama 2 7B 32K Instruct

## Installation

Python 3.10 or later is required. Install a CUDA-enabled PyTorch build and a
compatible FlashAttention build for your system first, then install this
package:

```bash
pip install -e .
```

The implementation is fixed to `transformers==4.45.0` and
`tokenizers==0.20.1`.

## Patterns

The provided patterns are stored at:

```text
models/<model>/patterns/budget20.pt
```

Each file contains a `pattern` tensor and its `keep_ratio`. Inspect a pattern
without loading a model:

```bash
python examples/inspect_pattern.py models/llama31_8b_128k/patterns/budget20.pt
```

The model-specific static cache implementations are in
`src/distance_kv_pattern/inference/static_q_head/`. They accept boolean
Q-head keep masks constructed at the benchmark-specific query boundary.

## Minimal inference

The following example runs Llama 3.1 with the provided pattern. It requires a
CUDA environment with FlashAttention and access to the model weights.

```bash
python examples/run_llama31_inference.py \
  --context-file context.txt \
  --query "Summarize the context."
```

The script tokenizes the complete prompt once, finds the query boundary,
prefills the context densely, compacts its KV cache with the static Q-head
pattern, and continues with greedy generation.

## Training

Model-independent training code is under
`src/distance_kv_pattern/training/`. Model-specific backends, configurations,
initialization entry points, and training entry points are under
`models/<model>/`.

Generate the training manifests before running initialization or training:

```bash
python -m distance_kv_pattern.training.data_pipeline.scripts.generate_manifests \
  --config models/llama31_8b_128k/configs/data.json \
  --output-dir models/llama31_8b_128k/data/manifests \
  --needle-counts 4 --coverage-rounds 0 1 2 3 --no-local-files-only

python -m distance_kv_pattern.training.data_pipeline.scripts.generate_manifests \
  --config models/qwen25_7b_128k/configs/data.json \
  --output-dir models/qwen25_7b_128k/data/manifests \
  --needle-counts 4 --coverage-rounds 0 1 2 3 --no-local-files-only

python models/llama2_7b_32k/data/generate.py \
  --config models/llama2_7b_32k/configs/data.json \
  --output-dir models/llama2_7b_32k/data/manifests \
  --needle-counts 4 --coverage-rounds 0 1 2 3 --no-local-files-only
```

The initialization and training configurations use repository-relative paths.
Model weights are downloaded separately from their respective providers.

## Verification

```bash
pip install -e '.[test]'
pytest -q
```

## License

The project is released under the Apache License 2.0. The packed KV update
CUDA kernel is adapted from
[DefensiveKV](https://github.com/FFY0/DefensiveKV) and remains under the MIT
License; see `NOTICE` and
`src/distance_kv_pattern/inference/static_q_head/csrc/LICENSE`.
