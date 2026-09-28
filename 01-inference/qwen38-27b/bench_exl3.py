"""Qwen3.8-27B on the RTX 3090 in ExLlamaV3 (EXL3): the same questions as bench.py, for the build that claims 262K
context on 24 GB.

Runs in its own environment (the prebuilt wheel pins torch 2.13 + CUDA 13.2):

    uv run --project 01-inference/envs/exl3 01-inference/qwen38-27b/bench_exl3.py --ctx 262144

Loads the model with its MTP head, allocates an int4 KV cache of --ctx tokens, then measures prefill, decode at short
and long context, batched decode, and decode with MTP drafting on.
"""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from exllamav3 import Cache, Config, Generator, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ArgmaxSampler

RESULTS = Path(__file__).parent / "results"
PROMPT = "Explain, step by step, how a GPU executes a matrix multiplication, from the kernel launch to the result."


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def chat(text):
    """Qwen chat format, thinking off (the same prompt bench.py builds with the HF template)."""
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def filler(n_words, seed):
    words = [f"record {seed}-{i}: the {['north', 'south', 'east', 'west'][i % 4]} depot shipped {i * 7 % 101} crates"
             for i in range(n_words // 8)]
    return ". ".join(words) + ".\n\nIn one sentence, what is this document about?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="turboderp/Qwen3.8-27B-exl3")
    parser.add_argument("--revision", default="4.00bpw")
    parser.add_argument("--ctx", type=int, default=262144, help="KV cache size in tokens, shared by all requests")
    parser.add_argument("--cache-bits", type=int, default=4)
    parser.add_argument("--long", type=int, default=32768, help="length of the long-context prompt, in tokens")
    parser.add_argument("--slots", type=int, default=1,
                        help="concurrent sequences; each reserves its own DeltaNet state (x draft+1 snapshots)")
    parser.add_argument("--draft", type=int, default=4, help="MTP draft tokens per step (0 = skip the MTP test)")
    parser.add_argument("--new-tokens", type=int, default=256)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    model_dir = snapshot_download(args.model, revision=args.revision)
    idle = gpu_used_mib()
    t0 = time.perf_counter()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    mtp = Model.from_config(config, component="mtp") if args.draft else None
    # EXL3 reserves the cache first, then fits the weights around it
    cache = Cache(model, max_num_tokens=args.ctx, layer_type=CacheLayer_quant,
                  k_bits=args.cache_bits, v_bits=args.cache_bits, max_history=args.draft,
                  max_batch_size=args.slots)
    mtp_cache = Cache(mtp, max_num_tokens=args.ctx, layer_type=CacheLayer_quant,
                      k_bits=args.cache_bits, v_bits=args.cache_bits, max_batch_size=args.slots) if mtp else None
    if mtp:
        mtp.load(progressbar=True)
    model.load(progressbar=True)
    load_s = time.perf_counter() - t0
    loaded = gpu_used_mib()
    cache_mib = sum(t.numel() * t.element_size() for c in (cache, mtp_cache) if c for t in c.get_all_tensors()) / 2**20
    state_mib = sum(t.numel() * t.element_size() for c in (cache, mtp_cache) if c for layer in c.recurrent_layers.values()
                    for t in vars(layer).values() if torch.is_tensor(t)) / 2**20
    tokenizer = Tokenizer.from_config(config)
    plain = Generator(model, cache, tokenizer, max_batch_size=args.slots)
    storage = model.get_storage_info()
    print(f"== loaded in {load_s:.0f} s; GPU memory {idle} -> {loaded} MiB, of which {cache_mib:.0f} MiB is the"
          f" {args.ctx}-token int{args.cache_bits} KV cache and {state_mib:.0f} MiB is DeltaNet state for {args.slots}"
          f" slot(s);  bpw (layers, head): {storage[:2]}")

    res = {"model": args.model, "revision": args.revision, "ctx": args.ctx, "cache_bits": args.cache_bits,
           "load_s": load_s, "gpu_idle_mib": idle, "gpu_loaded_mib": loaded,
           "slots": args.slots, "draft": args.draft,
           "kv_cache_mib": cache_mib, "deltanet_state_mib": state_mib, "bpw_layers": storage[0], "bpw_head": storage[1]}

    def gen(generator, prompts, n, ignore_eos=True):
        t = time.perf_counter()
        out = generator.generate(prompt=prompts, max_new_tokens=n, min_new_tokens=n if ignore_eos else 0,
                                 sampler=ArgmaxSampler(), completion_only=True, return_last_results=True,
                                 stop_conditions=None if ignore_eos else ["<|im_end|>", tokenizer.single_id("<|im_end|>")])
        return time.perf_counter() - t, out

    def decode_rate(generator, prompt, repeats=3):
        """(n-1) tokens over (time for n) - (time for 1): the prompt is in the page cache, so prefill cancels out."""
        gen(generator, prompt, 8)
        rates, stats = [], None
        for _ in range(repeats):
            t1, _ = gen(generator, prompt, 1)
            tn, (text, stats) = gen(generator, prompt, args.new_tokens)
            rates.append((args.new_tokens - 1) / (tn - t1))
        return statistics.median(rates), text, stats

    rate, sample, _ = decode_rate(plain, chat(PROMPT))
    res["decode_short_tok_s"] = rate
    print(f"== decode, short prompt: {rate:.1f} tok/s ({1e3 / rate:.1f} ms/token);  sample: {sample[:100]!r}")

    n_words = int(args.long * 8 / 14)
    long_prompt = chat(filler(n_words, seed=0))
    n_long = tokenizer.encode(long_prompt).shape[-1]
    t_pf, _ = gen(plain, long_prompt, 1)  # first pass: nothing cached yet, so this is prefill
    res["prefill_tokens"], res["prefill_s"] = n_long, t_pf
    print(f"== prefill: {n_long} tokens in {t_pf:.2f} s = {n_long / t_pf:.0f} tok/s")
    rate_long, sample, _ = decode_rate(plain, long_prompt)
    res["decode_long_tok_s"] = rate_long
    print(f"== decode after {n_long} tokens of context: {rate_long:.1f} tok/s;  sample: {sample[:100]!r}")

    res["batch"] = {}
    for b in sorted({1, min(4, args.slots), args.slots}):
        prompts = [chat(f"{PROMPT} (Variant {i}.)") for i in range(b)]
        gen(plain, prompts, 8)
        t, _ = gen(plain, prompts, args.new_tokens)
        res["batch"][b] = b * args.new_tokens / t
        print(f"== batch {b:>2}: {res['batch'][b]:.0f} tok/s total, {res['batch'][b] / b:.1f} per stream")

    _, (answer, _) = gen(plain, chat("A farmer has 17 sheep. All but 9 run away. How many are left? "
                                     "Answer with one number."), 64, ignore_eos=False)
    res["sanity"] = answer
    print(f"== sanity: {answer!r}")

    if args.draft:
        # MTP drafting: the model's own extra head guesses the next few tokens, one full pass checks them all
        del plain
        torch.cuda.empty_cache()
        drafted = Generator(model, cache, tokenizer, draft_model=mtp, draft_cache=mtp_cache, num_draft_tokens=args.draft, max_batch_size=args.slots)
        res["mtp"] = {}
        for label, prompt in (("short", chat(PROMPT)), ("long", long_prompt)):
            rate_mtp, sample, stats = decode_rate(drafted, prompt)
            acc, rej = stats.get("accepted_draft_tokens", 0), stats.get("rejected_draft_tokens", 0)
            res["mtp"][label] = {"tok_s": rate_mtp, "accepted": acc, "rejected": rej}
            print(f"== MTP x{args.draft}, {label} prompt: {rate_mtp:.1f} tok/s; drafts accepted {acc}/{acc + rej};"
                  f"  sample: {sample[:100]!r}")

    (RESULTS / f"exl3-{args.revision}-ctx{args.ctx}-slots{args.slots}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
