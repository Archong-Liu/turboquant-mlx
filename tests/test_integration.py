"""
Integration tests using a mock model — no model download required.

Tests that TurboQuantKVCache integrates correctly with an mlx-lm-style
forward pass: cache.update_and_fetch is called per layer, offset grows,
output shape is correct.
"""
import sys
sys.path.insert(0, ".")
import mlx.core as mx
from turboquant.kv_cache import TurboQuantKVCache

BATCH, HEADS, HEAD_DIM = 1, 4, 64
N_LAYERS = 3


class MockAttentionLayer:
    """Minimal attention layer that calls cache.update_and_fetch like mlx-lm does."""

    def __call__(self, x: mx.array, cache: TurboQuantKVCache) -> mx.array:
        B, S, _ = x.shape
        k = mx.random.normal(shape=(B, HEADS, S, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal(shape=(B, HEADS, S, HEAD_DIM)).astype(mx.bfloat16)
        k_full, v_full = cache.update_and_fetch(k, v)
        # Dummy output — just return x unchanged
        return x


class MockModel:
    def __init__(self):
        self.layers = [MockAttentionLayer() for _ in range(N_LAYERS)]

    def __call__(self, x: mx.array, cache: list) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i])
        return x


def make_caches():
    return [TurboQuantKVCache(bits=3, head_dim=HEAD_DIM, seed=i) for i in range(N_LAYERS)]


def test_forward_pass_runs():
    model = MockModel()
    cache = make_caches()
    x = mx.random.normal(shape=(BATCH, 8, HEAD_DIM * HEADS)).astype(mx.bfloat16)
    out = model(x, cache)
    mx.eval(out)
    assert out.shape == x.shape


def test_each_layer_has_independent_offset():
    model = MockModel()
    cache = make_caches()
    x = mx.random.normal(shape=(BATCH, 5, HEAD_DIM * HEADS)).astype(mx.bfloat16)
    model(x, cache)
    for c in cache:
        assert c.offset == 5, f"Expected offset 5, got {c.offset}"


def test_offset_grows_across_steps():
    model = MockModel()
    cache = make_caches()

    # Prefill S=6
    x6 = mx.random.normal(shape=(BATCH, 6, HEAD_DIM * HEADS)).astype(mx.bfloat16)
    model(x6, cache)
    assert cache[0].offset == 6

    # Decode S=1
    x1 = mx.random.normal(shape=(BATCH, 1, HEAD_DIM * HEADS)).astype(mx.bfloat16)
    model(x1, cache)
    assert cache[0].offset == 7

    # Another decode
    model(x1, cache)
    assert cache[0].offset == 8


def test_output_seq_dim_grows_monotonically():
    """The keys returned by update_and_fetch should grow by 1 per decode step."""
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    seq_dims = []
    for step in range(5):
        k = mx.random.normal(shape=(BATCH, HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal(shape=(BATCH, HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
        k_out, _ = cache.update_and_fetch(k, v)
        mx.eval(k_out)
        seq_dims.append(k_out.shape[2])

    assert seq_dims == list(range(1, 6)), f"Unexpected seq dims: {seq_dims}"


def test_reset_allows_reuse():
    model = MockModel()
    cache = make_caches()

    x = mx.random.normal(shape=(BATCH, 4, HEAD_DIM * HEADS)).astype(mx.bfloat16)
    model(x, cache)
    assert cache[0].offset == 4

    for c in cache:
        c.reset()
    assert cache[0].offset == 0

    model(x, cache)
    assert cache[0].offset == 4


if __name__ == "__main__":
    test_forward_pass_runs()
    print("PASS: forward pass runs")
    test_each_layer_has_independent_offset()
    print("PASS: per-layer offset")
    test_offset_grows_across_steps()
    print("PASS: offset grows across steps")
    test_output_seq_dim_grows_monotonically()
    print("PASS: seq dim grows monotonically")
    test_reset_allows_reuse()
    print("PASS: reset allows reuse")
    print("\nAll integration tests passed.")
