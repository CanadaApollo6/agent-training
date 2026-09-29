"""Draft vocabularies A/B, in-process and interleaved: the engine drafts with each candidate head in turn on the same
prompts; tok/s, tokens a round and ms a round per head, and greedy outputs must match (drafts never change output).

Usage: python ab.py MODEL_DIR [--tokens 512] [--reps 2] [--sizes 16384,24576,32768,49152]"""

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda import capacity

HERE = Path(__file__).parent
TRACES = Path.home() / "Projects/Personal/Research/agent-training/06-agents/harness-evals/outputs/primeintellect"
PROMPTS = [
    ("chat", "Prove that there are infinitely many primes, then list the first 20 of them."),
    ("chat", "Write a Python function that parses ISO 8601 durations like P3DT4H5M, with tests."),
    ("completion", "The history of the transistor begins"),
    ("chat", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."),
]


def agent_prompts(n=2):
    """Task prompts of terminal-bench tasks held out of coverage.py's ranking."""
    held = json.load(open(HERE / "held_out_tasks.json"))
    names = {}
    for line in open(TRACES / "terminal-bench-2--ornith35b-q8--pi--0467b767/traces.jsonl"):
        d = json.loads(line)["task"]["data"]
        names[d["name"]] = d["prompt"]
    return [("chat", names[t]) for t in held if t in names][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sizes", default="16384,24576,32768,49152")
    ap.add_argument("--context", type=int, default=32768)
    a = ap.parse_args()
    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    from tokenizers import Tokenizer

    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine
    from tensorfold.families.qwen3_5_moe.cuda.mtp import Head
    from tensorfold.families.qwen4_exp.cuda.weights import draft_token_ids

    raw = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    tok = TemplateTokens(raw, ChatTemplate(Path(a.model)))
    prompts = []
    for kind, text in PROMPTS + agent_prompts():
        if kind == "chat":
            prompts.append(list(tok.apply_chat_template([{"role": "user", "content": text}],
                                                        add_generation_prompt=True, enable_thinking=False)))
        else:
            prompts.append(raw.encode(text, add_special_tokens=False).ids)
    e = Qwen36Engine(a.model, context=a.context, context_explicit=True)
    default = draft_token_ids("default")
    ranked = np.load(HERE / "ranked.npy")
    seen = int(np.load(HERE / "seen.npy")) if (HERE / "seen.npy").exists() else 0
    # trace-seen ids by frequency, then the default list's ids, then the rest
    order = np.concatenate([ranked[:seen], [t for t in default if t not in set(ranked[:seen].tolist())]])
    heads = {"default": e.head}
    for n in map(int, a.sizes.split(",")):
        heads[f"top{n // 1024}K"] = Head(e.w, e.head.m, np.sort(order[:n]))
    stash = {k: {} for k in heads}
    samplings = {"t0": None, "t1": Sampling(seed=1234, temperature=1.0)}
    agg = collections.defaultdict(lambda: [0.0, 0, 0, 0, 0])   # seconds, tokens, rounds, drafted, accepted
    greedy = collections.defaultdict(dict)

    def run(name, pi, sname):
        h = heads[name]
        e.head = e.graphs.head = h
        e.graphs.mtp = stash[name]
        e.cache.entries.clear()
        out = []
        s = e.generate(prompts[pi], a.tokens, samplings[sname], lambda t: out.extend(t) and False, stop_eos=False)
        return s, out

    for name in heads:                                   # warm: each head's graphs captured
        run(name, 0, "t0")
    for rep in range(a.reps):
        for pi in range(len(prompts)):
            for sname in samplings:
                for name in (list(heads) if rep % 2 == 0 else list(heads)[::-1]):
                    s, out = run(name, pi, sname)
                    g = agg[(name, sname)]
                    g[0] += s["decode_s"]; g[1] += len(out) - 1; g[2] += s["rounds"]
                    g[3] += s["drafted"]; g[4] += s["accepted"]
                    if sname == "t0":
                        greedy[pi].setdefault(name, out)
        print(f"rep {rep} done", flush=True)
    same = all(len({tuple(v) for v in d.values()}) == 1 for d in greedy.values())
    print(f"greedy outputs identical across heads: {same}")
    base = {s: agg[("default", s)][1] / agg[("default", s)][0] for s in samplings}
    for sname in samplings:
        for name in heads:
            sec, n, r, dr, ac = agg[(name, sname)]
            print(f"{sname} {name:8s} vocab {heads[name].ids.numel() if heads[name].ids is not None else 0:6d}: "
                  f"{n / sec:6.1f} tok/s ({100 * (n / sec / base[sname] - 1):+5.1f}%)  {n / r:4.2f} tokens/round  "
                  f"{1000 * sec / r:5.2f} ms/round  accepted {ac / max(dr, 1):.3f} of drafts", flush=True)


if __name__ == "__main__":
    main()
