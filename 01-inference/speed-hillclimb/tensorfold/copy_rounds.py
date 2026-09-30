"""Per verify round: copied continuation or MTP drafts, its width, tokens kept, and wall time. R2 in-process, the
seg_ab prompts plus two held-out agent tasks, 1,024 tokens, greedy and temperature 1. With several --copy-rows, the
copy window sizes interleaved in-process (greedy outputs must match), plus how far each copy would have gone unbounded.

Usage: python copy_rounds.py MODEL_DIR [--tokens 1024] [--copy-rows 16,32] [--reps 2]"""

import argparse
import collections
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "draftvocab"))

from tensorfold.cuda import capacity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--copy-rows", default="16")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--stop-eos", action="store_true", help="stop at the answer's end (the bench runs past it)")
    ap.add_argument("--greedy-only", action="store_true")
    a = ap.parse_args()
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(0.3 * capacity.GIB))
    from tokenizers import Tokenizer

    from ab import agent_prompts
    from seg_ab import PROMPTS
    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5.cuda import decode as dense_decode
    from tensorfold.families.qwen3_5_moe.cuda import decode
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    raw = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    tok = TemplateTokens(raw, ChatTemplate(Path(a.model)))
    prompts = [list(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            enable_thinking=False)) if k == "chat"
               else raw.encode(t, add_special_tokens=False).ids for k, t in PROMPTS + agent_prompts()]
    rounds = []                                    # (kind, width, kept, seconds)
    state = {"copy": False, "t": None, "context": None}
    runs = []                                      # (context position, unbounded copy) a copy round
    propose = dense_decode.CopyIndex.propose

    def spy_propose(self, context, max_nodes=127):
        out = propose(self, context, max_nodes)
        state["copy"] = bool(out)
        if out:
            state["context"] = context
            runs.append((len(context), propose(self, context, 4096)))
        return out

    accept = decode.accept

    def spy_accept(tokens, parents, sampled, *rest):
        path, terminal = accept(tokens, parents, sampled, *rest)
        now = time.perf_counter()
        if state["t"] is not None:
            rounds.append(("copy" if state["copy"] else "draft", len(tokens), len(path), now - state["t"]))
        state["t"] = now
        return path, terminal

    dense_decode.CopyIndex.propose = spy_propose
    decode.CopyIndex = dense_decode.CopyIndex
    decode.accept = spy_accept
    e = Qwen36Engine(a.model, context=32768, context_explicit=True)
    widths = [int(r) for r in a.copy_rows.split(",")]

    def run(p, sampling):
        state["t"], state["context"] = None, None
        runs.clear()
        e.cache.entries.clear()
        got = []
        t0 = time.perf_counter()
        e.generate(p, a.tokens, sampling, lambda t: got.extend(t) and False, stop_eos=a.stop_eos)
        wall = time.perf_counter() - t0
        seq = state["context"]
        spans = [] if seq is None else [next((i for i, (x, y) in enumerate(zip(c, seq[pos:])) if x != y),
                                            min(len(c), len(seq) - pos)) for pos, c in runs]
        return wall, spans, got

    samplings = [("greedy", None), ("temperature 1", Sampling(seed=1234, temperature=1.0))]
    for name, sampling in samplings[:1] if a.greedy_only else samplings:
        for r in widths:                           # warm: graphs for every width
            decode.COPY_ROWS = r
            for p in prompts:
                e.generate(p, a.tokens, sampling, lambda t: False, stop_eos=a.stop_eos)
        per = {r: {"rounds": [], "wall": 0.0, "spans": [], "texts": []} for r in widths}
        for _ in range(a.reps):
            for r in widths:
                decode.COPY_ROWS = r
                for p in prompts:
                    rounds.clear()
                    wall, spans, seq = run(p, sampling)
                    per[r]["rounds"] += rounds
                    per[r]["wall"] += wall
                    per[r]["spans"] += spans
                    per[r]["texts"].append(seq)
        for r in widths:
            report(f"{name}, copy window {r} rows", per[r], sum(map(len, per[r]["texts"])))
        if sampling is None and len(widths) > 1:
            same = all(per[r]["texts"] == per[widths[0]]["texts"] for r in widths)
            print(f"  greedy outputs identical across windows: {same}")


def report(title, res, n):
    rounds = res["rounds"]
    print(f"== {title}: {n} tokens, {n / res['wall']:.1f} tok/s over {res['wall']:.2f} s", flush=True)
    by = collections.defaultdict(list)
    for r in rounds:
        by[(r[0], r[1])].append(r)
    tot_t = sum(r[3] for r in rounds)
    for key in sorted(by):
        rs = by[key]
        if len(rs) < 5:
            continue
        kept = [r[2] for r in rs]
        ms = 1000 * sum(r[3] for r in rs) / len(rs)
        hist = collections.Counter(kept)
        print(f"  {key[0]:5s} width {key[1]:2d}: {len(rs):4d} rounds ({100 * sum(r[3] for r in rs) / tot_t:4.1f}% of "
              f"time)  kept {sum(kept) / len(rs):5.2f} tokens  {ms:5.2f} ms  {sum(kept) / (ms * len(rs)):.3f} "
              f"tokens/ms  kept hist {dict(sorted(hist.items()))}", flush=True)
    spans = res["spans"]
    if spans:
        cuts = (0, 1, 4, 8, 15, 16, 24, 32, 48, 64, 128, 4096)
        hist = {f"{lo + 1}-{hi}": sum(lo < s <= hi for s in spans) for lo, hi in zip(cuts, cuts[1:])}
        print(f"  copies matching unbounded ({len(spans)}): mean {sum(spans) / len(spans):.1f} tokens  {hist}")

if __name__ == "__main__":
    main()
