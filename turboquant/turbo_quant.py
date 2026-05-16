import mlx.core as mx
from .polar_quant import PolarQuant
from .qjl import QJL


class TurboQuant:
    """
    Two-stage KV compression pipeline:
      Stage 1: random rotation + PolarQuant (3-bit angular codes + norms)
      Stage 2: QJL single-bit correction on the residual error
    """

    def __init__(self, dim: int, bits: int = 3, qjl_output_dim: int = None, seed: int = 42):
        self.dim = dim
        self.polar = PolarQuant(bits=bits)
        self.qjl = QJL(
            input_dim=dim,
            output_dim=qjl_output_dim or dim,
            seed=seed,
        )
        # Fixed random rotation matrix (orthogonal-ish via QR of random normal)
        key = mx.random.key(seed + 1)
        raw = mx.random.normal(shape=(dim, dim), key=key)
        # Approximate orthogonal matrix via normalization of columns
        self.R = (raw / (mx.linalg.norm(raw, axis=0, keepdims=True) + 1e-8)).astype(mx.bfloat16)

    def compress(self, x: mx.array) -> dict:
        """
        x: (..., dim)
        Returns a dict of compressed components.
        """
        x = x.astype(mx.bfloat16)

        # Stage 1: rotate then polar-quantize
        x_rot = x @ self.R
        codes, norms = self.polar.compress(x_rot)

        # Compute reconstruction to get residual
        x_approx = self.polar.decompress(codes, norms)
        residual = x_rot - x_approx

        # Stage 2: QJL on residual
        residual_bits = self.qjl.compress(residual)

        return {"codes": codes, "norms": norms, "residual_bits": residual_bits}

    def decompress(self, compressed: dict) -> mx.array:
        """
        Reconstruct approximate vectors from compressed dict.
        """
        x_approx = self.polar.decompress(compressed["codes"], compressed["norms"])
        residual_approx = self.qjl.decompress(compressed["residual_bits"])
        x_rot = x_approx + residual_approx
        # Undo rotation: R is approximate orthogonal, so R^T ≈ R^{-1}
        return (x_rot @ self.R.T).astype(mx.bfloat16)
