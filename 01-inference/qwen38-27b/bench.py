"""Qwen3.8-27B on the RTX 3090: what fits, how fast it decodes, and how that compares with the bandwidth ceiling.

Runs in the vLLM environment:

    uv run --project 01-inference/envs/vllm 01-inference/qwen38-27b/bench.py              # plain decode
    uv run --project 01-inference/envs/vllm 01-inference/qwen38-27b/bench.py --mtp 2      # + MTP speculative decoding

Part 1 needs no GPU: it reads the checkpoint's tensor headers and counts the bytes a decode step must stream.
Part 2 loads the model in vLLM and measures memory, prefill, decode at short and long context, and batched decode.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

RESULTS = Path(__file__).parent / "results"
BANDWIDTH = 829e9  # measured, 00-setup/roofline.py
PROMPT = "Explain, step by step, how a GPU executes a matrix multiplication, from the kernel launch to the result."


def tensor_bytes(model_dir):
    """Bytes per tensor, read from the safetensors headers (no weights loaded)."""
    sizes = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                start, end = meta["data_offsets"]
                sizes[name] = (end - start, meta["dtype"])
    return sizes


def category(name):
    if "visual" in name:
        return "vision tower (unused for text)"
    if name.startswith("mtp"):
        return "MTP head (only read when drafting)"
    if "embed_tokens" in name:
        return "embedding table (one row read per token)"
    if "lm_head" in name:
        return "LM head (bf16, read in full every token)"
    if "linear_attn" in name:
        return "DeltaNet layers"
    if "self_attn" in name:
        return "full-attention layers"
    if ".mlp." in name:
        return "MLPs"
    return "norms and other"


def byte_budget(model_dir, cfg):
    sizes = tensor_bytes(model_dir)
    cats = defaultdict(lambda: defaultdict(int))
    for name, (b, dtype) in sizes.items():
        cats[category(name)][dtype] += b
    table = {c: dict(d) for c, d in cats.items()}
    total = {c: sum(d.values()) for c, d in cats.items()}

    t = cfg["text_config"]
    n_full = t["layer_types"].count("full_attention")
    n_lin = t["layer_types"].count("linear_attention")
    kv_per_token = n_full * 2 * t["num_key_value_heads"] * t["head_dim"] * 2  # K and V, bf16
    state = n_lin * t["linear_num_value_heads"] * t["linear_key_head_dim"] * t["linear_value_head_dim"] * 4  # fp32
    row = t["hidden_size"] * 2

    streamed = sum(v for c, v in total.items() if not c.startswith(("vision", "MTP", "embedding"))) + row
    per_token = lambda ctx: streamed + 2 * state + kv_per_token * ctx  # state is read and written each step
    return {
        "by_category": table,
        "category_bytes": total,
        "full_attention_layers": n_full,
        "deltanet_layers": n_lin,
        "kv_bytes_per_token": kv_per_token,
        "deltanet_state_bytes": state,
        "streamed_weight_bytes": streamed,
        "ceiling_tok_s": {ctx: BANDWIDTH / per_token(ctx) for ctx in (0, 16384, 32768)},
    }


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def filler(n_words, seed):
    """Unique long text, so prefix caching can't reuse another prompt's prefill."""
    words = [f"record {seed}-{i}: the {['north', 'south', 'east', 'west'][i % 4]} depot shipped {i * 7 % 101} crates"
             for i in range(n_words // 8)]
    return ". ".join(words) + ".\n\nIn one sentence, what is this document about?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cyankiwi/Qwen3.8-27B-AWQ-INT4")
    parser.add_argument("--ctx", type=int, default=32768, help="max_model_len")
    parser.add_argument("--util", type=float, default=0.935, help="gpu_memory_utilization (the desktop holds ~1.2 GiB)")
    parser.add_argument("--chunk", type=int, default=2048, help="max_num_batched_tokens (prefill chunk)")
    parser.add_argument("--seqs", type=int, default=8, help="max_num_seqs")
    parser.add_argument("--mtp", type=int, default=0, help="MTP draft tokens (0 = off)")
    parser.add_argument("--new-tokens", type=int, default=256)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    model_dir = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    budget = byte_budget(model_dir, cfg)
    print("== bytes on disk by category (GB)")
    for c, b in sorted(budget["category_bytes"].items(), key=lambda x: -x[1]):
        dtypes = ", ".join(f"{d} {v / 1e9:.2f}" for d, v in budget["by_category"][c].items())
        print(f"  {c:45s} {b / 1e9:6.2f}   ({dtypes})")
    print(f"  streamed per decode step: {budget['streamed_weight_bytes'] / 1e9:.2f} GB of weights"
          f" + {2 * budget['deltanet_state_bytes'] / 1e6:.0f} MB DeltaNet state (read+write)"
          f" + {budget['kv_bytes_per_token'] / 1024:.0f} KB of KV per context token")
    for ctx, c in budget["ceiling_tok_s"].items():
        print(f"  ceiling at {ctx:>6} tokens of context: {c:.1f} tok/s ({1e3 / c:.1f} ms/token)")

    from vllm import LLM, SamplingParams

    idle = gpu_used_mib()
    spec = {"method": "mtp", "num_speculative_tokens": args.mtp} if args.mtp else None
    t0 = time.perf_counter()
    llm = LLM(model=args.model, max_model_len=args.ctx, gpu_memory_utilization=args.util, max_num_seqs=args.seqs,
              max_num_batched_tokens=args.chunk, limit_mm_per_prompt={"image": 0, "video": 0},
              hf_overrides={"language_model_only": True}, speculative_config=spec)
    load_s = time.perf_counter() - t0
    loaded = gpu_used_mib()
    cache = llm.llm_engine.vllm_config.cache_config
    kv_blocks = getattr(cache, "num_gpu_blocks", None)
    print(f"\n== loaded in {load_s:.0f} s; GPU memory {idle} -> {loaded} MiB; KV blocks {kv_blocks} x {cache.block_size}")

    tok = llm.get_tokenizer()

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def gen(prompts, n, ignore_eos=True):
        p = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n if ignore_eos else 0, ignore_eos=ignore_eos)
        t = time.perf_counter()
        out = llm.generate(prompts, p, use_tqdm=False)
        return time.perf_counter() - t, out

    def decode_rate(prompt, repeats=3):
        gen([prompt], 8)  # warm up, and put the prompt in the prefix cache
        rates = []
        for _ in range(repeats):
            t1, _ = gen([prompt], 1)
            tn, out = gen([prompt], args.new_tokens)
            rates.append((args.new_tokens - 1) / (tn - t1))
        return statistics.median(rates), out[0].outputs[0].text

    res = {"model": args.model, "mtp": args.mtp, "ctx": args.ctx, "util": args.util, "chunk": args.chunk,
           "seqs": args.seqs, "load_s": load_s,
           "gpu_idle_mib": idle, "gpu_loaded_mib": loaded, "kv_blocks": kv_blocks, "block_size": cache.block_size,
           "budget": {k: v for k, v in budget.items() if k != "by_category"}}

    rate, sample = decode_rate(chat(PROMPT))
    res["decode_short_tok_s"] = rate
    print(f"== decode, short prompt: {rate:.1f} tok/s ({1e3 / rate:.1f} ms/token)"
          f" = {rate / budget['ceiling_tok_s'][0]:.0%} of ceiling;  sample: {sample[:100]!r}")

    n_words = int(args.ctx * 0.8 * 8 / 14)  # a filler record is ~14 tokens per 8 words; fill ~80% of the window
    long_prompt = chat(filler(n_words, seed=0))
    n_long = len(tok(long_prompt).input_ids)
    t_pf, _ = gen([chat(filler(n_words, seed=1))], 1)  # a fresh prompt of the same length: prefill only
    res["prefill_tokens"], res["prefill_s"] = n_long, t_pf
    print(f"== prefill: {n_long} tokens in {t_pf:.2f} s = {n_long / t_pf:.0f} tok/s")
    rate_long, sample = decode_rate(long_prompt)
    res["decode_long_tok_s"] = rate_long
    print(f"== decode after {n_long} tokens of context: {rate_long:.1f} tok/s ({1e3 / rate_long:.1f} ms/token);"
          f"  sample: {sample[:100]!r}")

    res["batch"] = {}
    for b in (1, 4, args.seqs):
        prompts = [chat(f"{PROMPT} (Variant {i}.)") for i in range(b)]
        gen(prompts, 8)
        t, _ = gen(prompts, args.new_tokens)
        res["batch"][b] = b * args.new_tokens / t
        print(f"== batch {b:>2}: {res['batch'][b]:.0f} tok/s total, {res['batch'][b] / b:.1f} per stream")

    _, out = gen([chat("A farmer has 17 sheep. All but 9 run away. How many are left? Answer with one number.")],
                 64, ignore_eos=False)
    res["sanity"] = out[0].outputs[0].text
    print(f"== sanity: {res['sanity']!r}")

    name = f"vllm-awq{'-mtp' + str(args.mtp) if args.mtp else ''}"
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
