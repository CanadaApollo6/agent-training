"""Draft-head attention window A/B at long agent contexts: TensorFold's Qwen3.5-MoE engine in-process, the draft head
seeing every committed key (window 0) or only the last N (``mtp.DRAFT_WINDOW``), interleaved.

The contexts are real prime_agent transcripts from TB2 runs (the longest call path of a rollout, rendered through the
chat template with its tools), cut after a tool result at about ``--context`` tokens, so the next assistant turn is
what gets decoded. Sampled as agents run (T 1, top_k 20, top_p 0.95). The window only changes drafts, so every
variant's output must be identical; it reports decode tok/s, ms a round, and drafts accepted.

Usage: python draft_window_ab.py MODEL_DIR TRACES.jsonl [--context 64000] [--tokens 1024] [--reps 2]
       [--windows 0,8192,4096,2048] [--prompts 2]
"""

import argparse
import collections
import json
import os
from pathlib import Path

from tensorfold.cuda import capacity


def transcripts(path: str):
    """Each rollout's longest call path: (task, tools, messages)."""

    for line in open(path):
        r = json.loads(line)
        for t in r.get("traces") or []:
            nodes, best = t["nodes"], []
            for c in t["calls"]:
                i, chain = c.get("node"), []
                while i is not None:
                    chain.append(i)
                    i = nodes[i].get("parent")
                best = max(best, chain[::-1], key=len)
            if best:
                yield r["task"]["data"]["name"], nodes[best[-1]].get("tools"), [nodes[i]["message"] for i in best]


def clean(m: dict) -> dict:
    m = {k: v for k, v in m.items() if v is not None}
    calls = []
    for c in m.get("tool_calls") or []:
        f = dict(c.get("function", c))
        if isinstance(f.get("arguments"), str):
            try:
                f["arguments"] = json.loads(f["arguments"])
            except json.JSONDecodeError:
                f["arguments"] = {"raw": f["arguments"]}
        calls.append({"type": "function", "function": f})
    if calls:
        m["tool_calls"] = calls
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("traces")
    ap.add_argument("--context", type=int, default=64000)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--windows", default="0,8192,4096,2048")
    ap.add_argument("--prompts", type=int, default=2)
    a = ap.parse_args()
    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    from tokenizers import Tokenizer

    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe.cuda import mtp
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    tok = TemplateTokens(Tokenizer.from_file(str(Path(a.model) / "tokenizer.json")), ChatTemplate(Path(a.model)))
    prompts, names = [], []
    for task, tools, messages in transcripts(a.traces):
        if task in names:
            continue
        messages = [clean(m) for m in messages]
        cut = None                              # the longest prefix ending on a tool result under the target
        for n in range(2, len(messages) + 1):
            if messages[n - 1].get("role") != "tool":
                continue
            ids = tok.apply_chat_template(messages[:n], tools=tools, add_generation_prompt=True)
            if len(ids) > a.context:
                break
            cut = ids
        if cut is not None and len(cut) > 0.85 * a.context:
            prompts.append(list(cut))
            names.append(task)
        if len(prompts) == a.prompts:
            break
    print("prompts:", [(n.split("/")[-1], len(p)) for n, p in zip(names, prompts)], flush=True)
    e = Qwen36Engine(a.model, context=a.context + a.tokens + 1024, context_explicit=True)
    windows = [int(x) for x in a.windows.split(",")]
    graphs = {w: ({}, {}) for w in windows}     # each window's captured draft graphs (the window is baked in)
    sampling = Sampling(seed=1234, temperature=1.0, top_k=20, top_p=0.95)

    def run(w, p):
        mtp.DRAFT_WINDOW = w
        e.graphs.mtp, e.graphs.chains = graphs[w]
        out = []
        s = e.generate(prompts[p], a.tokens, sampling, lambda t: out.extend(t) and False, stop_eos=False)
        return s, out

    for p in range(len(prompts)):               # warm: every graph captured (prompt outermost: one prefill each)
        for w in windows:
            run(w, p)
    agg = collections.defaultdict(lambda: [0.0, 0, 0, 0])
    outs = collections.defaultdict(dict)
    for rep in range(a.reps):
        for p in range(len(prompts)):
            for w in (windows if rep % 2 == 0 else windows[::-1]):
                s, out = run(w, p)
                g = agg[(w, p)]
                g[0] += s["decode_s"]; g[1] += s["rounds"]; g[2] += s["drafted"]; g[3] += s["accepted"]
                outs[p].setdefault(w, out)
    print("outputs identical:", all(len({tuple(o) for o in d.values()}) == 1 for d in outs.values()))
    tot = collections.defaultdict(lambda: [0.0, 0, 0, 0])
    for p in range(len(prompts)):
        line = f"prompt {p} ({len(prompts[p])} tokens):"
        for w in windows:
            sec, r, dr, ac = agg[(w, p)]
            for i, x in enumerate((sec, r, dr, ac)):
                tot[w][i] += x
            line += f"\n  window {w:6}: {a.reps * a.tokens / sec:6.1f} tok/s  {1000 * sec / r:5.2f} ms/round  " \
                    f"accepted {ac / dr:5.1%} of drafts  {a.reps * a.tokens / r:4.2f} tokens/round"
        print(line, flush=True)
    n = a.reps * a.tokens * len(prompts)
    base = n / tot[windows[0]][0]
    print("all: " + ", ".join(f"window {w} {n / tot[w][0]:.1f} tok/s ({100 * (n / tot[w][0] / base - 1):+.1f}%, "
                              f"accepted {tot[w][3] / tot[w][2]:.1%})" for w in windows))


if __name__ == "__main__":
    main()
