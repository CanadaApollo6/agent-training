"""Decode-gap ladder, vLLM rungs: an inference engine's fused kernels, with and without CUDA graphs.

Same prompt, greedy decode and timing method as bench_hf.py (decode rate = extra tokens / extra time).
Runs in its own environment because vLLM pins its own torch:

    uv run --project 01-inference/envs/vllm 01-inference/decode-gap/bench_vllm.py            # CUDA graphs (default)
    uv run --project 01-inference/envs/vllm 01-inference/decode-gap/bench_vllm.py --eager    # enforce_eager: no graphs
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPT = "Explain, step by step, how a GPU executes a matrix multiplication, from the kernel launch to the result."
RESULTS = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--eager", action="store_true", help="disable CUDA graphs")
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        enforce_eager=args.eager,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        limit_mm_per_prompt={"image": 0, "video": 0},  # text only: skip the vision encoder
    )

    def run(n):
        params = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True)
        t = time.perf_counter()
        out = llm.generate([PROMPT], params, use_tqdm=False)
        return time.perf_counter() - t, out[0].outputs[0]

    run(8)  # warmup
    rates = []
    for _ in range(args.repeats):
        t1, _ = run(1)
        tn, out = run(args.new_tokens)
        rates.append((args.new_tokens - 1) / (tn - t1))
    rate = statistics.median(rates)

    ref = json.loads((RESULTS / "hf-fla.json").read_text())
    ceiling = ref["ceiling_tok_s"]
    agree = next((i for i, (a, b) in enumerate(zip(out.token_ids, ref["output_ids"])) if a != b), args.new_tokens)
    name = "vllm-eager" if args.eager else "vllm-graphs"
    print(f"{name}: {rate:.1f} tok/s ({1e3 / rate:.2f} ms/token) = {rate / ceiling:.1%} of the {ceiling:.0f} tok/s ceiling")
    print(f"first {agree} of {args.new_tokens} tokens identical to HF eager;  sample: {out.text[:120]!r}")
    (RESULTS / f"{name}.json").write_text(json.dumps({
        "variant": name, "tok_s": rate, "all_tok_s": rates, "ceiling_tok_s": ceiling,
        "fraction_of_ceiling": rate / ceiling, "tokens_identical_to_hf": agree, "output_ids": list(out.token_ids),
    }, indent=1))


if __name__ == "__main__":
    main()
