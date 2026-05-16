"""
Compare peak memory usage: standard KV cache vs TurboQuantKVCache
across increasing sequence lengths.
"""
import time
import resource
import mlx.core as mx
from turboquant import TurboQuantKVCache


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run(batch: int, heads: int, head_dim: int, seq_len: int, use_turbo: bool):
    if use_turbo:
        cache = TurboQuantKVCache(bits=3, head_dim=head_dim)
    else:
        all_keys, all_values = [], []

    before = peak_mb()
    for step in range(seq_len):
        k = mx.random.normal(shape=(batch, heads, 1, head_dim)).astype(mx.bfloat16)
        v = mx.random.normal(shape=(batch, heads, 1, head_dim)).astype(mx.bfloat16)
        if use_turbo:
            cache.update(k, v)
        else:
            all_keys.append(k)
            all_values.append(v)
        mx.eval(k, v)

    after = peak_mb()
    return after - before


if __name__ == "__main__":
    cfg = dict(batch=1, heads=8, head_dim=128)
    seq_lengths = [512, 1024, 2048, 4096]

    print(f"{'SeqLen':>8}  {'Baseline (MB)':>15}  {'TurboQuant (MB)':>16}  {'Reduction':>10}")
    print("-" * 58)
    for sl in seq_lengths:
        base = run(**cfg, seq_len=sl, use_turbo=False)
        turbo = run(**cfg, seq_len=sl, use_turbo=True)
        ratio = base / turbo if turbo > 0 else float("inf")
        print(f"{sl:>8}  {base:>15.1f}  {turbo:>16.1f}  {ratio:>9.2f}x")
