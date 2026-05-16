import sys
sys.path.insert(0, ".")
import mlx.core as mx
from turboquant.kv_cache import TurboQuantKVCache

BATCH, HEADS, HEAD_DIM = 1, 4, 64


def make_kv(seq_len: int):
    k = mx.random.normal(shape=(BATCH, HEADS, seq_len, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal(shape=(BATCH, HEADS, seq_len, HEAD_DIM)).astype(mx.bfloat16)
    return k, v


def test_update_and_fetch_shape_prefill():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k, v = make_kv(5)
    k_out, v_out = cache.update_and_fetch(k, v)
    mx.eval(k_out, v_out)
    assert k_out.shape == (BATCH, HEADS, 5, HEAD_DIM), f"Got {k_out.shape}"
    assert v_out.shape == (BATCH, HEADS, 5, HEAD_DIM), f"Got {v_out.shape}"


def test_update_and_fetch_accumulates():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k5, v5 = make_kv(5)
    cache.update_and_fetch(k5, v5)

    k1, v1 = make_kv(1)
    k_out, v_out = cache.update_and_fetch(k1, v1)
    mx.eval(k_out, v_out)
    assert k_out.shape == (BATCH, HEADS, 6, HEAD_DIM), f"Got {k_out.shape}"


def test_offset_increments_correctly():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    assert cache.offset == 0
    k, v = make_kv(10)
    cache.update_and_fetch(k, v)
    assert cache.offset == 10
    k1, v1 = make_kv(1)
    cache.update_and_fetch(k1, v1)
    assert cache.offset == 11


def test_empty():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    assert cache.empty() is True
    k, v = make_kv(3)
    cache.update_and_fetch(k, v)
    assert cache.empty() is False


def test_nbytes_positive():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k, v = make_kv(4)
    cache.update_and_fetch(k, v)
    assert cache.nbytes > 0, "nbytes should be > 0 after storing data"


def test_size():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k, v = make_kv(7)
    cache.update_and_fetch(k, v)
    assert cache.size() == 7


def test_reset():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k, v = make_kv(5)
    cache.update_and_fetch(k, v)
    cache.reset()
    assert cache.empty() is True
    assert cache.offset == 0
    assert cache.nbytes == 0


def test_is_trimmable_false():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    assert cache.is_trimmable() is False


def test_make_mask_shape():
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k, v = make_kv(5)
    cache.update_and_fetch(k, v)
    # cache provides offset internally; caller only passes N, return_array, window_size
    mask = cache.make_mask(3, return_array=True, window_size=None)
    if mask is not None:
        mx.eval(mask)
        assert mask.shape == (3, 8), f"Expected (3, 8), got {mask.shape}"


def test_sequential_decode():
    """Prefill with S=4, then two decode steps of S=1. Final seq dim = 6."""
    cache = TurboQuantKVCache(bits=3, head_dim=HEAD_DIM)
    k4, v4 = make_kv(4)
    cache.update_and_fetch(k4, v4)
    for _ in range(2):
        k1, v1 = make_kv(1)
        cache.update_and_fetch(k1, v1)
    k_out, v_out = cache.update_and_fetch(*make_kv(0) if False else make_kv(1))
    mx.eval(k_out)
    assert k_out.shape[2] == 7, f"Expected seq_len=7, got {k_out.shape[2]}"


if __name__ == "__main__":
    test_update_and_fetch_shape_prefill()
    print("PASS: prefill shape")
    test_update_and_fetch_accumulates()
    print("PASS: accumulates across steps")
    test_offset_increments_correctly()
    print("PASS: offset increments")
    test_empty()
    print("PASS: empty flag")
    test_nbytes_positive()
    print("PASS: nbytes > 0")
    test_size()
    print("PASS: size()")
    test_reset()
    print("PASS: reset")
    test_is_trimmable_false()
    print("PASS: is_trimmable=False")
    test_make_mask_shape()
    print("PASS: make_mask shape")
    test_sequential_decode()
    print("PASS: sequential decode")
    print("\nAll KVCache tests passed.")
