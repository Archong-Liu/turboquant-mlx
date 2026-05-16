import mlx.core as mx
from .turbo_quant import TurboQuant


class TurboQuantKVCache:
    """
    Drop-in KV cache replacement for mlx-lm models.
    Compresses keys and values on insert using TurboQuant.

    Usage:
        cache = TurboQuantKVCache(bits=3, head_dim=128)
        cache.update(keys, values)   # keys/values: (batch, heads, seq, dim)
        k, v = cache.get()
    """

    def __init__(self, bits: int = 3, head_dim: int = 128, seed: int = 42):
        self.bits = bits
        self.head_dim = head_dim
        self._key_compressor = TurboQuant(dim=head_dim, bits=bits, seed=seed)
        self._val_compressor = TurboQuant(dim=head_dim, bits=bits, seed=seed + 100)
        self._keys: list[dict] = []
        self._values: list[dict] = []
        # Track offset for positional bookkeeping (mlx-lm compat)
        self.offset = 0

    def update(self, keys: mx.array, values: mx.array):
        """
        Compress and store a new KV slice.
        keys, values: (batch, heads, seq, head_dim)
        """
        B, H, S, D = keys.shape
        # Flatten to (B*H*S, D) for compression, then store per-step
        k_flat = keys.reshape(-1, D)
        v_flat = values.reshape(-1, D)
        self._keys.append({
            "compressed": self._key_compressor.compress(k_flat),
            "shape": (B, H, S, D),
        })
        self._values.append({
            "compressed": self._val_compressor.compress(v_flat),
            "shape": (B, H, S, D),
        })
        self.offset += S

    def get(self) -> tuple[mx.array, mx.array]:
        """
        Decompress and return full cached K, V tensors.
        Returns (keys, values) each of shape (batch, heads, total_seq, head_dim).
        """
        keys = mx.concatenate([
            self._key_compressor.decompress(e["compressed"]).reshape(e["shape"])
            for e in self._keys
        ], axis=2)
        values = mx.concatenate([
            self._val_compressor.decompress(e["compressed"]).reshape(e["shape"])
            for e in self._values
        ], axis=2)
        return keys, values

    def reset(self):
        self._keys.clear()
        self._values.clear()
        self.offset = 0

    def __len__(self):
        return self.offset
