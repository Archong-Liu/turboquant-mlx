"""
Speed benchmark: KVCache (fp16) vs TurboQuantKVCache (3-bit).

Measures two phases separately:
  Decode  — S=1 per step (the latency-critical path during generation)
  Prefill — S=full prompt (one big forward pass)

Usage:
    cd turboquant-mlx
    python benchmarks/benchmark_speed.py
"""
import sys
import time
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from turboquant import TurboQuantKVCache

BATCH    = 1
N_HEADS  = 8
HEAD_DIM = 128
N_WARMUP = 20
N_STEPS  = 200


def bench_decode(cache_factory, n_steps: int = N_STEPS, n_warmup: int = N_WARMUP) -> float:
    """Returns tokens/sec for the decode phase (one token at a time)."""
    cache = cache_factory()
    k1 = mx.random.normal(shape=(BATCH, N_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    v1 = mx.random.normal(shape=(BATCH, N_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(k1, v1)

    # Warmup — fills cache so timing is stable
    for _ in range(n_warmup):
        k_out, v_out = cache.update_and_fetch(k1, v1)
        mx.eval(k_out, v_out)

    # Fresh cache for timing (avoid O(n²) decompression artifact)
    cache = cache_factory()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        k_out, v_out = cache.update_and_fetch(k1, v1)
        mx.eval(k_out, v_out)
    elapsed = time.perf_counter() - t0
    return n_steps / elapsed


def bench_prefill(cache_factory, seq_len: int) -> float:
    """Returns latency in ms for a single prefill call of seq_len tokens."""
    cache = cache_factory()
    k = mx.random.normal(shape=(BATCH, N_HEADS, seq_len, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal(shape=(BATCH, N_HEADS, seq_len, HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(k, v)  # materialise before timing

    t0 = time.perf_counter()
    k_out, v_out = cache.update_and_fetch(k, v)
    mx.eval(k_out, v_out)
    return (time.perf_counter() - t0) * 1000  # ms


def main():
    cache_configs = [
        ("KVCache fp16",      lambda: KVCache()),
        ("TurboQuant 3-bit",  lambda: TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)),
        ("TurboQuant 4-bit",  lambda: TurboQuantKVCache(bits=4, head_dim=HEAD_DIM)),
    ]

    print(f"\nConfig: batch={BATCH}, heads={N_HEADS}, head_dim={HEAD_DIM}")
    print(f"Decode: {N_WARMUP} warmup steps, {N_STEPS} timed steps\n")

    # --- Decode throughput ---
    print("=" * 55)
    print("Decode throughput (S=1 per step)")
    print("=" * 55)
    print(f"{'Cache Type':<22}  {'tokens/sec':>12}  {'overhead':>10}")
    print("-" * 48)

    baseline_tps = None
    for label, factory in cache_configs:
        try:
            tps = bench_decode(factory)
        except Exception as e:
            print(f"{label:<22}  ERROR: {e}")
            continue
        if baseline_tps is None:
            baseline_tps = tps
            overhead_str = "baseline"
        else:
            overhead_str = f"{(baseline_tps / tps - 1) * 100:+.1f}%"
        print(f"{label:<22}  {tps:>12,.0f}  {overhead_str:>10}")

    # --- Prefill latency ---
    print()
    print("=" * 55)
    print("Prefill latency (single call, full sequence)")
    print("=" * 55)
    seq_lengths = [256, 512, 1024, 2048]
    header = f"{'Cache Type':<22}" + "".join(f"  {sl:>6}tok" for sl in seq_lengths)
    print(header)
    print("-" * (22 + 10 * len(seq_lengths)))

    for label, factory in cache_configs:
        row = f"{label:<22}"
        for sl in seq_lengths:
            try:
                ms = bench_prefill(factory, sl)
                row += f"  {ms:>7.1f}ms"
            except Exception:
                row += f"  {'ERR':>9}"
        print(row)

    print()
    print("Interpretation:")
    print("  Decode overhead = extra latency per generated token from compress+decompress.")
    print("  Prefill latency = time to cache the input prompt.")
    print("  TurboQuant trades some speed for a 3-6× smaller memory footprint.")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
