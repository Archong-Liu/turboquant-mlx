import mlx.core as mx
import sys
sys.path.insert(0, ".")
from turboquant.polar_quant import PolarQuant


def test_round_trip_shape():
    pq = PolarQuant(bits=3)
    x = mx.random.normal(shape=(4, 128)).astype(mx.bfloat16)
    codes, norms = pq.compress(x)
    x_hat = pq.decompress(codes, norms)
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"


def test_code_range():
    pq = PolarQuant(bits=3)
    x = mx.random.normal(shape=(16, 64)).astype(mx.float32)
    codes, _ = pq.compress(x)
    mx.eval(codes)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 2**3


def test_reconstruction_cosine_similarity():
    pq = PolarQuant(bits=3)
    x = mx.random.normal(shape=(32, 128)).astype(mx.float32)
    codes, norms = pq.compress(x)
    x_hat = pq.decompress(codes, norms).astype(mx.float32)

    dot = (x * x_hat).sum(axis=-1)
    norm_x = mx.linalg.norm(x, axis=-1)
    norm_hat = mx.linalg.norm(x_hat, axis=-1)
    cos_sim = dot / (norm_x * norm_hat + 1e-8)
    mx.eval(cos_sim)
    mean_sim = float(cos_sim.mean())
    print(f"Mean cosine similarity (3-bit PolarQuant): {mean_sim:.4f}")
    assert mean_sim > 0.8, f"Cosine similarity too low: {mean_sim}"


if __name__ == "__main__":
    test_round_trip_shape()
    print("PASS: round-trip shape")
    test_code_range()
    print("PASS: code range")
    test_reconstruction_cosine_similarity()
    print("PASS: cosine similarity")
    print("\nAll PolarQuant tests passed.")
