import mlx.core as mx
import sys
sys.path.insert(0, ".")
from turboquant.qjl import QJL


def test_output_shape():
    qjl = QJL(input_dim=128, output_dim=128)
    x = mx.random.normal(shape=(8, 128)).astype(mx.bfloat16)
    bits = qjl.compress(x)
    assert bits.shape == (8, 128), f"Unexpected shape: {bits.shape}"


def test_bits_are_boolean():
    qjl = QJL(input_dim=64, output_dim=64)
    x = mx.random.normal(shape=(4, 64)).astype(mx.bfloat16)
    bits = qjl.compress(x)
    mx.eval(bits)
    unique = set(bits.flatten().tolist())
    assert unique <= {True, False}, f"Unexpected values: {unique}"


def test_reconstruction_preserves_sign():
    """QJL reconstruction should preserve the sign pattern of the original."""
    qjl = QJL(input_dim=128, output_dim=128)
    x = mx.random.normal(shape=(32, 128)).astype(mx.float32)
    bits = qjl.compress(x.astype(mx.bfloat16))
    x_hat = qjl.decompress(bits).astype(mx.float32)

    # Dot product between original and reconstruction should be mostly positive
    dot = (x * x_hat).sum(axis=-1)
    mx.eval(dot)
    positive_frac = float((dot > 0).astype(mx.float32).mean())
    print(f"Fraction of vectors with positive dot product: {positive_frac:.4f}")
    assert positive_frac > 0.6, f"Too many sign flips: {positive_frac}"


if __name__ == "__main__":
    test_output_shape()
    print("PASS: output shape")
    test_bits_are_boolean()
    print("PASS: bits are boolean")
    test_reconstruction_preserves_sign()
    print("PASS: reconstruction preserves sign")
    print("\nAll QJL tests passed.")
