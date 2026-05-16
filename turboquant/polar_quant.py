import mlx.core as mx


class PolarQuant:
    """
    Quantizes vectors by converting to polar coordinates.
    Exploits the fixed-norm structure of attention keys:
    angles are quantized to `bits` bits; the sign of the norm is stored separately.
    """

    def __init__(self, bits: int = 3):
        self.bits = bits
        self.n_bins = 2**bits  # angular bins

    def compress(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        x: (..., D) float16/bfloat16
        Returns (angle_codes, norms) where angle_codes are uint8 angular bin indices.
        """
        # Normalize to unit sphere — polar quantization acts on direction
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        x_unit = x / (norms + 1e-8)

        # Encode each dimension as an angular bin in [-1, 1]
        # Values are in [-1, 1] after normalization; map to [0, n_bins-1]
        codes = mx.floor((x_unit + 1.0) * (self.n_bins / 2.0)).astype(mx.uint8)
        codes = mx.clip(codes, 0, self.n_bins - 1)

        return codes, norms.squeeze(-1)

    def decompress(self, codes: mx.array, norms: mx.array) -> mx.array:
        """
        Reconstruct approximate vectors from codes and norms.
        """
        # Map bin indices back to [-1, 1]
        x_unit = (codes.astype(mx.float32) + 0.5) / (self.n_bins / 2.0) - 1.0
        # Re-apply norm
        return x_unit * norms[..., None]
