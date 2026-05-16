"""
Compare compress/decompress throughput: TurboQuant vs no-op baseline.
Measures tokens/sec for KV cache operations.
"""
import time
import mlx.core as mx
from turboquant import TurboQuantKVCache


def bench(cache, keys, values, n_steps: int) -> float:
    cache.reset()
    start = time.perf_counter()
    for _ in range(n_steps):
        cache.update(keys, values)
        k, v = cache.get()
        mx.eval(k, v)
    elapsed = time.perf_counter() - start
    total_tokens = n_steps * keys.shape[2]
    return total_tokens / elapsed


if __name__ == "__main__":
    batch, heads, seq, head_dim = 1, 8, 1, 128
    n_steps = 200

    keys = mx.random.normal(shape=(batch, heads, seq, head_dim)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(batch, heads, seq, head_dim)).astype(mx.bfloat16)

    cache_3bit = TurboQuantKVCache(bits=3, head_dim=head_dim)
    cache_4bit = TurboQuantKVCache(bits=4, head_dim=head_dim)

    tps_3 = bench(cache_3bit, keys, values, n_steps)
    tps_4 = bench(cache_4bit, keys, values, n_steps)

    print(f"TurboQuant 3-bit: {tps_3:.0f} tokens/sec")
    print(f"TurboQuant 4-bit: {tps_4:.0f} tokens/sec")
