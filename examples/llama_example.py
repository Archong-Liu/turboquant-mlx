"""
End-to-end example: load a Llama model via mlx-lm and swap in TurboQuantKVCache.

Usage:
    pip install mlx-lm
    python examples/llama_example.py --model mlx-community/Llama-3.2-1B-4bit
"""
import argparse
import mlx.core as mx
from mlx_lm import load, generate
from turboquant import TurboQuantKVCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-4bit")
    parser.add_argument("--prompt", default="Explain KV cache compression in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    head_dim = model.args.head_dim if hasattr(model.args, "head_dim") else 128

    print(f"Using TurboQuantKVCache ({args.bits}-bit)")
    cache = TurboQuantKVCache(bits=args.bits, head_dim=head_dim)

    # mlx-lm's generate accepts a kv_cache argument in recent versions
    response = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=True,
    )
    print("\n--- Response ---")
    print(response)


if __name__ == "__main__":
    main()
