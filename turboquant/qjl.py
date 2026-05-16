import mlx.core as mx


class QJL:
    """
    Quantized Johnson-Lindenstrauss transform.
    Projects residual error vectors down to single sign bits using a random matrix.
    Adds zero memory overhead while correcting quantization bias.
    """

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Fixed random projection matrix drawn once; kept as bfloat16
        key = mx.random.key(seed)
        raw = mx.random.normal(shape=(input_dim, output_dim), key=key)
        self.R = (raw / mx.sqrt(mx.array(float(output_dim)))).astype(mx.bfloat16)

    def compress(self, residual: mx.array) -> mx.array:
        """
        residual: (..., input_dim)
        Returns sign bits packed as bool array (..., output_dim).
        """
        projected = residual.astype(mx.bfloat16) @ self.R  # (..., output_dim)
        return projected > 0  # single bit per projection

    def decompress(self, bits: mx.array) -> mx.array:
        """
        Reconstruct an approximate residual from sign bits.
        bits: (..., output_dim) bool
        Returns (..., input_dim) approximate residual.
        """
        signs = (bits.astype(mx.bfloat16) * 2.0 - 1.0)  # {-1, +1}
        # Transpose projection: R^T maps back to input space
        return signs @ self.R.T
