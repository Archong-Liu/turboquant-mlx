import mlx.core as mx
from .turbo_quant import TurboQuant

try:
    from mlx_lm.models.cache import _BaseCache, create_attention_mask
except ImportError:
    _BaseCache = object
    create_attention_mask = None


class TurboQuantKVCache(_BaseCache):
    """
    KV cache compatible with mlx-lm's update_and_fetch API.

    Compresses keys and values on every update using TurboQuant (3-bit polar
    codes + 1-bit QJL residual). Pass one instance per transformer layer via
    the prompt_cache argument of mlx_lm.generate / generate_step.

    Usage:
        from mlx_lm import load, generate
        from turboquant import TurboQuantKVCache

        model, tokenizer = load("mlx-community/Llama-3.2-1B-4bit")
        n_layers = model.args.num_hidden_layers
        head_dim  = model.args.hidden_size // model.args.num_attention_heads
        cache = [TurboQuantKVCache(bits=3, head_dim=head_dim, seed=i)
                 for i in range(n_layers)]
        text = generate(model, tokenizer, prompt="...", prompt_cache=cache)
    """

    def __init__(self, bits: int = 3, head_dim: int = 128, seed: int = 42):
        self.bits = bits
        self.head_dim = head_dim
        self._key_compressor = TurboQuant(dim=head_dim, bits=bits, seed=seed)
        self._val_compressor = TurboQuant(dim=head_dim, bits=bits, seed=seed + 100)
        self._keys: list[dict] = []
        self._values: list[dict] = []
        self.offset = 0

    # ------------------------------------------------------------------
    # Core mlx-lm cache contract
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Compress and store new keys/values; return the full accumulated sequence.

        Input shape:  (B, n_kv_heads, S, head_dim)
        Output shape: (B, n_kv_heads, total_seq, head_dim)
        """
        B, H, S, D = keys.shape

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

        all_k = mx.concatenate([
            self._key_compressor.decompress(e["compressed"]).reshape(e["shape"])
            for e in self._keys
        ], axis=2)
        all_v = mx.concatenate([
            self._val_compressor.decompress(e["compressed"]).reshape(e["shape"])
            for e in self._values
        ], axis=2)
        return all_k, all_v

    def empty(self) -> bool:
        return len(self._keys) == 0

    @property
    def nbytes(self) -> int:
        """
        Theoretical compressed size in bytes, assuming bit-packing.

        MLX stores uint8 codes and bool bits as full bytes internally, but the
        information content is much smaller:
          - Angular codes: `bits` bits per element  (e.g. 3 bits for bits=3)
          - Norms: 16 bits (bfloat16) per vector
          - QJL residual bits: 1 bit per element

        This property reports the bit-packed byte count — the size a hand-written
        C/Metal kernel would achieve with proper packing. This is the fair
        comparison against fp16 (2 bytes/element).
        """
        total = 0
        for e in self._keys + self._values:
            B, H, S, D = e["shape"]
            N = B * H * S
            total += (N * D * self.bits + 7) // 8   # packed angular codes
            total += N * 2                           # norms: bfloat16
            total += (N * D + 7) // 8               # packed QJL bits (1 bit each)
        return total

    @property
    def nbytes_actual(self) -> int:
        """Actual MLX array bytes (no bit-packing — uint8 + bool = 1 byte each)."""
        total = 0
        for e in self._keys + self._values:
            B, H, S, D = e["shape"]
            N = B * H * S
            total += N * D      # codes: uint8
            total += N * 2      # norms: bfloat16
            total += N * D      # residual bits: bool
        return total

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return False

    def make_mask(self, *args, **kwargs):
        if create_attention_mask is not None:
            return create_attention_mask(*args, offset=self.offset, **kwargs)
        return None

    def reset(self):
        self._keys.clear()
        self._values.clear()
        self.offset = 0

    def __len__(self):
        return self.offset


def generate_with_turbo_cache(
    model_path: str,
    prompt: str,
    max_tokens: int = 200,
    bits: int = 3,
    verbose: bool = False,
) -> tuple[str, list]:
    """
    Load an mlx-lm model and generate text using TurboQuantKVCache for every layer.

    Returns (generated_text, list_of_TurboQuantKVCache).
    The cache list can be inspected after generation for memory stats via cache[i].nbytes.
    """
    from mlx_lm import load, generate

    model, tokenizer = load(model_path)

    n_layers = model.args.num_hidden_layers
    head_dim = getattr(
        model.args,
        "head_dim",
        model.args.hidden_size // model.args.num_attention_heads,
    )

    cache = [
        TurboQuantKVCache(bits=bits, head_dim=head_dim, seed=i)
        for i in range(n_layers)
    ]

    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        prompt_cache=cache,
        verbose=verbose,
    )
    return text, cache
