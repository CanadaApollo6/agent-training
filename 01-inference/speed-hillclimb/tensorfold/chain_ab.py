"""Greedy drafting step by step (a host wait for each draft's pick) vs chained on the device in one graph
(``Graphs.chain``, one wait a chain), and that plus the accepted path's commit as one graph (``Graphs.commit``), not ~60
eager launches; in-process and interleaved: seg_ab's prompts plus two held-out agent tasks, 1,024
tokens. Outputs must match (round counts too, but for the full-chain verify). ``--burn N`` runs N busy processes alongside to load the host, as a
desktop with other work open does.

Usage: python chain_ab.py MODEL_DIR [--tokens 1024] [--reps 2] [--burn 0]"""

import argparse
import collections
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "draftvocab"))

from tensorfold.cuda import capacity  # noqa: E402


def spin(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100000):
            x = (x * 1103515245 + 12345) & 0xffffffff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--burn", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0, help="> 0: keyed sampling (only the commit graph applies)")
    a = ap.parse_args()
    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    from tokenizers import Tokenizer

    from ab import agent_prompts
    from seg_ab import PROMPTS
    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe.cuda import decode
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    raw = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    tok = TemplateTokens(raw, ChatTemplate(Path(a.model)))
    prompts = [list(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            enable_thinking=False)) if k == "chat"
               else raw.encode(t, add_special_tokens=False).ids for k, t in PROMPTS + agent_prompts()]
    e = Qwen36Engine(a.model, context=32768, context_explicit=True)
    variants = {"step by step": (False, False, False), "chained + commit graph": (True, True, False),
                "+ verify follows": (True, True, True)}
    agg = collections.defaultdict(lambda: [0.0, 0])
    outs = collections.defaultdict(dict)

    def run(v, p):
        decode.CHAIN, decode.COMMIT_GRAPH, decode.FOLLOW = variants[v]
        e.cache.entries.clear()
        out = []
        sampling = Sampling(seed=1234, temperature=a.temperature) if a.temperature > 0 else None
        s = e.generate(prompts[p], a.tokens, sampling, lambda t: out.extend(t) and False, stop_eos=False)
        return s, out

    for v in variants:                                   # warm: every graph these prompts take, captured
        for p in range(len(prompts)):
            run(v, p)
    stop = mp.Event()
    burners = [mp.Process(target=spin, args=(stop,), daemon=True) for _ in range(a.burn)]
    for b in burners:
        b.start()
    try:
        for rep in range(a.reps):
            for p in range(len(prompts)):
                for v in (list(variants) if rep % 2 == 0 else list(variants)[::-1]):
                    s, out = run(v, p)
                    g = agg[(v, p)]
                    g[0] += s["decode_s"]; g[1] += s["rounds"]
                    outs[p].setdefault(v, out)
                    assert outs[p][v] == out
    finally:
        stop.set()
        for b in burners:
            b.join()
    print(f"burners {a.burn}; outputs identical:", all(len({tuple(o) for o in d.values()}) == 1 for d in outs.values()))
    tot = collections.defaultdict(float)
    for p in range(len(prompts)):
        line = f"prompt {p}:"
        for v in variants:
            sec, r = agg[(v, p)]
            tot[v] += sec
            line += f"  {v} {a.reps * a.tokens / sec:6.1f} tok/s ({1000 * sec / r:5.2f} ms/round)"
        print(line, flush=True)
    n = a.reps * a.tokens * len(prompts)
    base = n / tot["step by step"]
    print("all: " + ", ".join(f"{v} {n / tot[v]:.1f} tok/s ({100 * (n / tot[v] / base - 1):+.1f}%)" for v in variants))


if __name__ == "__main__":
    main()
