"""Segmented qmm launches vs one launch a weight, in-process and interleaved (each with its own captured graphs):
greedy.py's prompts, 1,024 tokens. Outputs must match; reports decode seconds, rounds and verify widths.

Usage: python seg_ab.py MODEL_DIR [--tokens 1024] [--reps 2]"""

import argparse
import collections
import os
import time
from pathlib import Path

import torch

from tensorfold.cuda import capacity
from tensorfold.cuda.kernels import qmm

PROMPTS = [
    ("chat", "Prove that there are infinitely many primes, then list the first 20 of them."),
    ("chat", "Write a Python function that parses ISO 8601 durations like P3DT4H5M, with tests."),
    ("completion", "The history of the transistor begins"),
    ("chat", "A train leaves at 3:40 pm going 72 km/h; another leaves the same station at 4:05 pm going 90 km/h. "
             "When does the second catch the first? Think step by step."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--context", type=int, default=32768)
    a = ap.parse_args()
    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    from tokenizers import Tokenizer

    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    raw = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    tok = TemplateTokens(raw, ChatTemplate(Path(a.model)))
    prompts = [list(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            enable_thinking=False)) if k == "chat"
               else raw.encode(t, add_special_tokens=False).ids for k, t in PROMPTS]
    e = Qwen36Engine(a.model, context=a.context, context_explicit=True)
    variants = {"separate": 0, "segmented": 16}
    stash = {v: ({}, {}) for v in variants}
    agg = collections.defaultdict(lambda: [0.0, 0, 0])
    outs = collections.defaultdict(dict)

    def run(v, p):
        qmm.SEG_ROWS = variants[v]
        e.graphs.target, e.graphs.mtp = stash[v]
        e.cache.entries.clear()
        out = []
        t0 = time.perf_counter()
        s = e.generate(prompts[p], a.tokens, None, lambda t: out.extend(t) and False, stop_eos=False)
        return s, out, time.perf_counter() - t0

    for v in variants:                                   # warm: every width these prompts take, captured
        for p in range(len(prompts)):
            run(v, p)
    for rep in range(a.reps):
        for p in range(len(prompts)):
            for v in (list(variants) if rep % 2 == 0 else list(variants)[::-1]):
                s, out, wall = run(v, p)
                g = agg[(v, p)]
                g[0] += s["decode_s"]; g[1] += s["rounds"]; g[2] += wall
                outs[p].setdefault(v, out)
                assert outs[p][v] == out
    print("outputs identical:", all(len({tuple(x) for x in d.values()}) == 1 for d in outs.values()))
    for p in range(len(prompts)):
        line = f"prompt {p}:"
        for v in variants:
            sec, r, wall = agg[(v, p)]
            line += f"  {v} {a.reps * a.tokens / sec:6.1f} tok/s ({1000 * sec / r:5.2f} ms/round, wall {wall / a.reps:.2f} s)"
        print(line, flush=True)


if __name__ == "__main__":
    main()
