"""Decode-gap ladder, Hugging Face transformers rungs.

Measures batch-1 greedy decode speed of Qwen3.5-0.8B and compares it with the bandwidth ceiling: the bytes
the GPU must read per token, divided by the bandwidth measured by 00-setup/roofline.py.

    uv run 01-inference/decode-gap/bench_hf.py --variant torch     # Gated DeltaNet as a pure-PyTorch loop
    uv run 01-inference/decode-gap/bench_hf.py --variant fla       # flash-linear-attention Triton kernels
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["torch", "fla"], required=True)
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--new-tokens", type=int, default=256)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--bandwidth-gbs", type=float, default=829.0, help="measured by 00-setup/roofline.py")
args = parser.parse_args()

if args.variant == "torch":
    # transformers picks the DeltaNet implementation at import time: fla if importable, else a torch loop.
    # A None entry in sys.modules makes `import fla` raise, so this forces the torch fallback.
    sys.modules["fla"] = None

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

PROMPT = "Explain, step by step, how a GPU executes a matrix multiplication, from the kernel launch to the result."
RESULTS = Path(__file__).parent / "results"


def bytes_per_token(model, context_len):
    """Bytes read from DRAM to decode one token at batch 1 (a lower bound: activations and writes ignored)."""
    cfg = model.config
    # Every weight is read once. Embeddings are tied to the LM head, which reads the whole table,
    # and model.parameters() counts the shared tensor once.
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    n_linear = cfg.layer_types.count("linear_attention")
    n_full = cfg.layer_types.count("full_attention")
    # Gated DeltaNet: a fixed fp32 state per head, read and written back every token.
    state = n_linear * cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4 * 2
    # Full attention: read K and V for every cached position.
    kv = n_full * 2 * cfg.num_key_value_heads * cfg.head_dim * 2 * context_len
    return weights, state, kv


def timed_generate(model, inputs, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t, out


def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    inputs = tok(PROMPT, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    timed_generate(model, inputs, 8)  # warmup: Triton autotuning, cuBLAS handles, allocator
    # Decode rate = extra tokens / extra time, so prefill and first-token cost cancel out.
    rates, t1s = [], []
    for _ in range(args.repeats):
        t1, _ = timed_generate(model, inputs, 1)
        tn, out = timed_generate(model, inputs, args.new_tokens)
        rates.append((args.new_tokens - 1) / (tn - t1))
        t1s.append(t1)
    rate = statistics.median(rates)

    weights, state, kv = bytes_per_token(model, prompt_len + args.new_tokens // 2)
    total = weights + state + kv
    ceiling = args.bandwidth_gbs * 1e9 / total

    print(f"variant={args.variant}  prompt={prompt_len} tokens  generated={args.new_tokens}")
    print(f"bytes/token: weights {weights / 1e6:.0f} MB + DeltaNet state {state / 1e6:.1f} MB + KV {kv / 1e6:.2f} MB = {total / 1e6:.0f} MB")
    print(f"ceiling at {args.bandwidth_gbs:.0f} GB/s: {ceiling:.0f} tok/s")
    print(f"measured: {rate:.1f} tok/s ({1e3 / rate:.2f} ms/token) = {rate / ceiling:.1%} of ceiling;  prefill+1 token: {statistics.median(t1s) * 1e3:.0f} ms")
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
    print(f"sample: {text[:160]!r}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"hf-{args.variant}.json").write_text(json.dumps({
        "variant": args.variant, "model": args.model, "torch": torch.__version__,
        "tok_s": rate, "all_tok_s": rates, "ceiling_tok_s": ceiling, "fraction_of_ceiling": rate / ceiling,
        "bytes_per_token": {"weights": weights, "deltanet_state": state, "kv": kv},
        "prompt_tokens": prompt_len, "new_tokens": args.new_tokens,
        "output_ids": out[0, prompt_len:].tolist(),
    }, indent=1))


if __name__ == "__main__":
    main()
