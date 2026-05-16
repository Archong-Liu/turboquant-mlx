"""
End-to-end demo: standard KVCache vs TurboQuantKVCache side by side.

Downloads a small Llama model on first run (~700 MB) via HuggingFace.

Usage:
    python examples/llama_example.py
    python examples/llama_example.py --model mlx-community/Llama-3.2-1B-4bit
    python examples/llama_example.py --bits 3 --max-tokens 150
"""
import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from turboquant import TurboQuantKVCache


def rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def peak_metal_mb() -> float:
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def run_standard(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    cache = make_prompt_cache(model)
    mx.metal.clear_cache()
    mem_before = rss_mb()
    peak_before = peak_metal_mb()
    t0 = time.perf_counter()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                    prompt_cache=cache, verbose=False)
    elapsed = time.perf_counter() - t0
    return {
        "text": text,
        "rss_delta_mb": rss_mb() - mem_before,
        "peak_metal_mb": peak_metal_mb() - peak_before,
        "time_s": elapsed,
        "compressed_bytes": sum(c.nbytes for c in cache if hasattr(c, "nbytes")),
        "cache": cache,
    }


def run_turbo(model, tokenizer, prompt: str, max_tokens: int, bits: int,
              n_layers: int, head_dim: int) -> dict:
    cache = [TurboQuantKVCache(bits=bits, head_dim=head_dim, seed=i)
             for i in range(n_layers)]
    mx.metal.clear_cache()
    mem_before = rss_mb()
    peak_before = peak_metal_mb()
    t0 = time.perf_counter()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                    prompt_cache=cache, verbose=False)
    elapsed = time.perf_counter() - t0
    compressed_bytes = sum(c.nbytes for c in cache)
    return {
        "text": text,
        "rss_delta_mb": rss_mb() - mem_before,
        "peak_metal_mb": peak_metal_mb() - peak_before,
        "time_s": elapsed,
        "compressed_bytes": compressed_bytes,
        "cache": cache,
    }


def main():
    parser = argparse.ArgumentParser(description="TurboQuant vs standard KV cache demo")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-4bit",
                        help="HuggingFace model repo (MLX format)")
    parser.add_argument("--prompt", default=(
        "Explain in detail why key-value cache compression is important "
        "for running large language models on devices with limited memory, "
        "such as a MacBook Pro with Apple Silicon."
    ))
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    n_layers = model.args.num_hidden_layers
    head_dim = getattr(model.args, "head_dim",
                       model.args.hidden_size // model.args.num_attention_heads)

    print(f"Model: {n_layers} layers, head_dim={head_dim}")
    print(f"Prompt: {args.prompt[:80]}...")
    print()

    # --- Standard KVCache (baseline) ---
    print("=" * 60)
    print("Standard KVCache (fp16 baseline)")
    print("=" * 60)
    std = run_standard(model, tokenizer, args.prompt, args.max_tokens)
    print(std["text"][:400])
    print(f"\nTime:         {std['time_s']:.2f}s")
    print(f"RSS delta:    {std['rss_delta_mb']:.1f} MB")
    print(f"Peak Metal:   {std['peak_metal_mb']:.1f} MB")

    # --- TurboQuantKVCache ---
    print()
    print("=" * 60)
    print(f"TurboQuantKVCache ({args.bits}-bit)")
    print("=" * 60)
    tq = run_turbo(model, tokenizer, args.prompt, args.max_tokens,
                   args.bits, n_layers, head_dim)
    print(tq["text"][:400])
    print(f"\nTime:             {tq['time_s']:.2f}s")
    print(f"RSS delta:        {tq['rss_delta_mb']:.1f} MB")
    print(f"Peak Metal:       {tq['peak_metal_mb']:.1f} MB")
    print(f"Compressed bytes: {tq['compressed_bytes'] / 1024:.1f} KB")

    # --- Comparison ---
    print()
    print("=" * 60)
    print("Compression Summary")
    print("=" * 60)
    tokens = args.max_tokens
    std_kb_per_token = std["rss_delta_mb"] * 1024 / max(tokens, 1)
    tq_kb_per_token  = tq["compressed_bytes"] / 1024 / max(tokens, 1)

    if tq["rss_delta_mb"] > 0 and std["rss_delta_mb"] > 0:
        mem_ratio = std["rss_delta_mb"] / tq["rss_delta_mb"]
        print(f"Memory reduction (RSS):   {mem_ratio:.2f}x")

    if tq["compressed_bytes"] > 0:
        # Theoretical baseline: fp16 KV = 2 bytes * 2 (K+V) * n_layers * heads * head_dim per token
        # Use compressed_bytes as numerator to compute bits/token
        bits_per_elem = (tq["compressed_bytes"] * 8) / max(
            sum(e["shape"][1] * e["shape"][2] * e["shape"][3]
                for c in tq["cache"] for e in c._keys), 1
        )
        print(f"Effective bits/element:   {bits_per_elem:.2f} (target ~{args.bits + 1})")

    speed_overhead = (tq["time_s"] / std["time_s"] - 1) * 100
    print(f"Speed overhead:           {speed_overhead:+.1f}%")
    print()
    print("Projected max context at 16 GB (8 GB free after model weights):")
    fp16_bytes_per_tok = 2 * 2 * n_layers * head_dim  # K+V, 2 bytes, all layers
    turbo_bytes_per_tok = tq["compressed_bytes"] / max(tokens, 1)
    free_bytes = 8 * 1024**3
    ctx_std   = int(free_bytes / max(fp16_bytes_per_tok, 1))
    ctx_turbo = int(free_bytes / max(turbo_bytes_per_tok, 1))
    print(f"  Standard fp16:     {ctx_std // 1000:>6}K tokens")
    print(f"  TurboQuant {args.bits}-bit:  {ctx_turbo // 1000:>6}K tokens")


if __name__ == "__main__":
    main()
