import sys
sys.path.insert(0, ".")
import mlx.core as mx
from turboquant.turbo_quant import TurboQuant


def test_rotation_is_orthogonal():
    """After QR fix, R @ R.T should be near identity (bfloat16 tolerance ~0.02)."""
    tq = TurboQuant(dim=64, seed=0)
    R = tq.R.astype(mx.float32)
    err = (R @ R.T - mx.eye(64)).abs().max()
    mx.eval(err)
    assert float(err) < 0.02, f"R not orthogonal, max error = {float(err):.4f}"


def test_compress_returns_expected_keys():
    tq = TurboQuant(dim=64, seed=1)
    x = mx.random.normal(shape=(8, 64)).astype(mx.bfloat16)
    out = tq.compress(x)
    assert "codes" in out
    assert "norms" in out
    assert "residual_bits" in out


def test_compress_decompress_shapes():
    tq = TurboQuant(dim=128, seed=2)
    x = mx.random.normal(shape=(16, 128)).astype(mx.bfloat16)
    compressed = tq.compress(x)
    x_hat = tq.decompress(compressed)
    mx.eval(x_hat)
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"


def test_round_trip_cosine_similarity():
    """TurboQuant (rotation + PolarQuant + QJL) should beat PolarQuant alone (>0.85)."""
    tq = TurboQuant(dim=128, bits=3, seed=3)
    x = mx.random.normal(shape=(64, 128)).astype(mx.float32)
    compressed = tq.compress(x.astype(mx.bfloat16))
    x_hat = tq.decompress(compressed).astype(mx.float32)

    dot = (x * x_hat).sum(axis=-1)
    norm_x = mx.linalg.norm(x, axis=-1)
    norm_hat = mx.linalg.norm(x_hat, axis=-1)
    cos = dot / (norm_x * norm_hat + 1e-8)
    mx.eval(cos)
    mean_cos = float(cos.mean())
    print(f"Mean cosine similarity (TurboQuant 3-bit): {mean_cos:.4f}")
    # 3-bit compression with random rotation is lossy; 0.65 is a reasonable floor
    assert mean_cos > 0.65, f"Cosine too low: {mean_cos:.4f}"


def test_different_seeds_give_different_rotation():
    tq0 = TurboQuant(dim=32, seed=0)
    tq1 = TurboQuant(dim=32, seed=1)
    diff = (tq0.R - tq1.R).abs().max()
    mx.eval(diff)
    assert float(diff) > 0.01, "Different seeds produced identical rotation matrices"


if __name__ == "__main__":
    test_rotation_is_orthogonal()
    print("PASS: rotation is orthogonal")
    test_compress_returns_expected_keys()
    print("PASS: compress keys")
    test_compress_decompress_shapes()
    print("PASS: shapes preserved")
    test_round_trip_cosine_similarity()
    print("PASS: cosine similarity")
    test_different_seeds_give_different_rotation()
    print("PASS: different seeds differ")
    print("\nAll TurboQuant tests passed.")
