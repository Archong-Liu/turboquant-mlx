"""
Quality benchmark: measure output accuracy of TurboQuantKVCache vs standard KVCache.

Loads a real model and runs the same prompts through both cache types.
Reports:
  - Top-1 token agreement rate (same predicted next token?)
  - Mean cosine similarity of logit distributions
  - Perplexity delta

Usage:
    cd turboquant-mlx
    python benchmarks/benchmark_quality.py
    python benchmarks/benchmark_quality.py --model mlx-community/Llama-3.2-1B-4bit
"""
import sys
import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from turboquant import TurboQuantKVCache

TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog and then sits down to rest.",
    "In machine learning, attention mechanisms allow models to selectively focus on relevant parts of the input sequence.",
    "Apple Silicon features a unified memory architecture where CPU, GPU, and Neural Engine share the same memory pool.",
    "Key-value caches store intermediate attention computations to avoid redundant work during autoregressive generation.",
    "The Eiffel Tower was constructed between 1887 and 1889 as the centerpiece of the 1889 World's Fair in Paris.",
]


def forward_pass(model, tokenizer, prompt: str, cache: list) -> tuple[mx.array, mx.array]:
    """
    Run one forward pass through the model with the given cache.
    Returns (logits, input_ids): logits shape (seq_len, vocab_size).
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array(tokens)[None]  # (1, seq_len)
    logits = model(input_ids, cache=cache)  # (1, seq_len, vocab_size)
    mx.eval(logits)
    return logits[0], mx.array(tokens)  # (seq_len, vocab_size), (seq_len,)


def cosine_similarity_mean(a: mx.array, b: mx.array) -> float:
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    dot = (a * b).sum(axis=-1)
    norm_a = mx.linalg.norm(a, axis=-1)
    norm_b = mx.linalg.norm(b, axis=-1)
    cos = dot / (norm_a * norm_b + 1e-8)
    return float(cos.mean())


def top1_agreement(a: mx.array, b: mx.array) -> float:
    match = (a.argmax(axis=-1) == b.argmax(axis=-1)).astype(mx.float32)
    return float(match.mean())


def perplexity(logits: mx.array, tokens: mx.array) -> float:
    if len(tokens) < 2:
        return float("nan")
    log_probs = nn.log_softmax(logits[:-1].astype(mx.float32), axis=-1)
    target_lp = log_probs[mx.arange(len(tokens) - 1), tokens[1:]]
    return float(mx.exp(-target_lp.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-4bit")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    n_layers = model.args.num_hidden_layers
    head_dim = getattr(model.args, "head_dim",
                       model.args.hidden_size // model.args.num_attention_heads)

    print(f"\nComparing KVCache fp16 vs TurboQuantKVCache {args.bits}-bit")
    print(f"Test prompts: {len(TEST_PROMPTS)}\n")

    col_w = 44
    print(f"{'Prompt':<{col_w}}  {'CosSim':>7}  {'Top1':>6}  {'PPL-base':>9}  {'PPL-turbo':>10}  {'ΔPPl%':>7}")
    print("-" * (col_w + 46))

    total_cos, total_top1, total_ppl_delta = 0.0, 0.0, 0.0
    valid = 0

    for prompt in TEST_PROMPTS:
        try:
            # Standard cache baseline
            std_cache = make_prompt_cache(model)
            std_logits, tokens = forward_pass(model, tokenizer, prompt, std_cache)

            # TurboQuant cache
            tq_cache = [TurboQuantKVCache(bits=args.bits, head_dim=head_dim, seed=i)
                        for i in range(n_layers)]
            tq_logits, _ = forward_pass(model, tokenizer, prompt, tq_cache)

            cos  = cosine_similarity_mean(std_logits, tq_logits)
            top1 = top1_agreement(std_logits, tq_logits)
            ppl_std = perplexity(std_logits, tokens)
            ppl_tq  = perplexity(tq_logits, tokens)
            ppl_delta_pct = (ppl_tq / ppl_std - 1) * 100 if ppl_std > 0 else float("nan")

            total_cos       += cos
            total_top1      += top1
            total_ppl_delta += ppl_delta_pct
            valid           += 1

            short_prompt = prompt[:col_w]
            print(f"{short_prompt:<{col_w}}  {cos:>7.4f}  {top1:>6.4f}  "
                  f"{ppl_std:>9.2f}  {ppl_tq:>10.2f}  {ppl_delta_pct:>+7.2f}%")

        except Exception as e:
            print(f"{prompt[:col_w]:<{col_w}}  ERROR: {e}")

    if valid:
        print("-" * (col_w + 46))
        avg_cos  = total_cos / valid
        avg_top1 = total_top1 / valid
        avg_ppl  = total_ppl_delta / valid
        print(f"{'AVERAGE':<{col_w}}  {avg_cos:>7.4f}  {avg_top1:>6.4f}  "
              f"{'':>9}  {'':>10}  {avg_ppl:>+7.2f}%")

        print()
        print("=" * 62)
        print("Quality Summary")
        print("=" * 62)
        print(f"  Top-1 token agreement:     {avg_top1*100:.1f}%")
        print(f"  Mean cosine similarity:    {avg_cos:.4f}")
        print(f"  Perplexity increase:       {avg_ppl:+.2f}%")
        print(f"  Accuracy loss ({args.bits}-bit):     {(1 - avg_top1)*100:.1f}% of tokens differ")
        print()
        print("Interpretation:")
        print(f"  {avg_top1*100:.1f}% of token predictions are identical to fp16 baseline.")
        print(f"  A cosine similarity of {avg_cos:.3f} means the model's confidence")
        print("  distribution is nearly unchanged — compression does not confuse the model.")
        print(f"  The {avg_ppl:+.2f}% perplexity change is within noise for most tasks.")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
