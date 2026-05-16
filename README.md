# turboquant-mlx

TurboQuant KV cache compression for large language models, implemented with Apple MLX and Metal for Apple Silicon (M5).

Based on Google Research's [TurboQuant paper](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — achieves 3-bit KV cache quantization with no accuracy loss and no fine-tuning.

## Algorithm Overview

TurboQuant combines three techniques:

1. **PolarQuant** — converts attention K vectors from Cartesian to polar coordinates, exploiting fixed-norm structure for efficient quantization
2. **QJL (Quantized Johnson-Lindenstrauss)** — projects residual error to a single sign bit using random projections, zero memory overhead
3. **TurboQuant** — random rotation → PolarQuant → QJL residual correction

## Project Structure

```
turboquant-mlx/
├── turboquant/
│   ├── polar_quant.py     # PolarQuant: polar coordinate quantization
│   ├── qjl.py             # QJL: single-bit residual correction
│   ├── turbo_quant.py     # TurboQuant: combined two-stage pipeline
│   └── kv_cache.py        # Drop-in KV cache replacement for MLX LLMs
├── metal/
│   └── kernels.metal      # Fused Metal GPU kernels (MSL)
├── benchmarks/
│   ├── benchmark_memory.py
│   └── benchmark_speed.py
├── examples/
│   └── llama_example.py
└── tests/
    ├── test_polar_quant.py
    └── test_qjl.py
```

## Requirements

- macOS 15+ with Apple Silicon (M1–M5)
- Python 3.10+
- Xcode (for Metal kernels)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from turboquant import TurboQuantKVCache

# Drop-in replacement for standard KV cache in any MLX LLM
cache = TurboQuantKVCache(bits=3, head_dim=128)
cache.update(keys, values)
k, v = cache.get()
```

## Benchmarks

Run memory and speed benchmarks on your machine:

```bash
python benchmarks/benchmark_memory.py
python benchmarks/benchmark_speed.py
```

## Status

- [x] Project scaffold
- [ ] PolarQuant MLX implementation
- [ ] QJL MLX implementation
- [ ] TurboQuant pipeline
- [ ] KV cache integration (mlx-lm compatible)
- [ ] Metal fused kernels
- [ ] Benchmarks vs baseline
