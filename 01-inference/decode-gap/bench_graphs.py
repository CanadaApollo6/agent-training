"""Decode-gap ladder, CUDA graphs rung: the same HF model and kernels, but launched by graph replay.

One decode step (forward, argmax, position update, token append) is captured once into a CUDA graph. Generating
N tokens is then N calls to graph.replay(): no Python, no dispatcher and no per-kernel launch cost between tokens.
Nothing is fused, so the kernels are the ones profile_hf.py counted. This isolates launch overhead.

    uv run 01-inference/decode-gap/bench_graphs.py --variant fla
"""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["torch", "fla"], required=True)
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--new-tokens", type=int, default=256)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
if args.variant == "torch":
    sys.modules["fla"] = None  # force the pure-PyTorch DeltaNet path, as in bench_hf.py

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, StaticCache  # noqa: E402

from bench_hf import PROMPT, RESULTS  # noqa: E402


def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    P, N = ids.shape[1], args.new_tokens
    cache = StaticCache(config=model.config, max_cache_len=P + N + 16)

    # Static buffers: the graph reads and writes these exact addresses on every replay.
    cur_tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    cur_pos = torch.zeros(1, dtype=torch.long, device="cuda")
    step = torch.zeros(1, dtype=torch.long, device="cuda")
    out_toks = torch.zeros(N + 16, dtype=torch.long, device="cuda")

    @torch.inference_mode()
    def prefill():
        cache.reset()
        logits = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(P, device="cuda")).logits
        cur_tok.copy_(logits[:, -1].argmax(-1, keepdim=True))
        cur_pos.fill_(P)
        step.zero_()
        out_toks.zero_()
        out_toks[0] = cur_tok[0, 0]

    @torch.inference_mode()
    def decode_step():
        logits = model(input_ids=cur_tok, past_key_values=cache, cache_position=cur_pos).logits
        cur_tok.copy_(logits[:, -1].argmax(-1, keepdim=True))
        cur_pos.add_(1)
        step.add_(1)
        out_toks.index_copy_(0, step, cur_tok[0])

    # Warm up on a side stream (Triton autotuning, cuBLAS workspaces), then capture one step.
    prefill()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            decode_step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_step()

    rates = []
    for _ in range(args.repeats):
        prefill()  # reset() zeroes the cache in place, so the graph's addresses stay valid
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N - 1):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        rates.append((N - 1) / (start.elapsed_time(end) / 1e3))
    rate = sorted(rates)[len(rates) // 2]

    generated = out_toks[:N].tolist()
    ref = json.loads((RESULTS / f"hf-{args.variant}.json").read_text())
    ceiling = ref["ceiling_tok_s"]
    agree = next((i for i, (a, b) in enumerate(zip(generated, ref["output_ids"])) if a != b), N)
    print(f"variant={args.variant} + CUDA graph, {N} tokens")
    print(f"measured: {rate:.1f} tok/s ({1e3 / rate:.2f} ms/token) = {rate / ceiling:.1%} of the {ceiling:.0f} tok/s ceiling")
    print(f"vs eager HF generate: {rate / ref['tok_s']:.2f}x;  first {agree} of {N} tokens identical to eager")
    print(f"sample: {tok.decode(generated, skip_special_tokens=True)[:160]!r}")

    (RESULTS / f"graphs-{args.variant}.json").write_text(json.dumps({
        "variant": args.variant, "graphs": True, "tok_s": rate, "all_tok_s": rates, "ceiling_tok_s": ceiling,
        "fraction_of_ceiling": rate / ceiling, "tokens_identical_to_eager": agree, "output_ids": generated,
    }, indent=1))


if __name__ == "__main__":
    main()
