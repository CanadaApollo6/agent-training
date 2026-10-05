"""Today's three sampled-decode changes together against the port as it stood this morning, in-process and
interleaved: the keyed draw chained on the device (``decode.SAMPLED_CHAIN``), the attention merge that loads four chunks
at once, and the draft head's attention window (``mtp.DRAFT_WINDOW``). "morning" turns the chain off, windows nothing,
and runs the attention module from port commit 3d9e583 (its kernels compiled alongside; same plans and offsets).

Short contexts are sampled_ab's prompts (seg_ab's four plus two held-out agent tasks); long ones are real prime_agent
transcripts cut after a tool result near ``--context`` (as in draft_window_ab). Sampled as agents run (T 1, top_k 20,
top_p 0.95). None of the changes alters a token, so both variants' outputs must be identical.

Usage: python combined_ab.py MODEL_DIR TRACES.jsonl [--context 64000] [--prompts 2] [--tokens 1024] [--reps 2]
"""

import argparse
import collections
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "draftvocab"))

from tensorfold.cuda import capacity  # noqa: E402

PORT = HERE.parents[1] / "tools" / "TensorFold-060"
MORNING = "3d9e583"


def morning_attention():
    """The port's attention module at MORNING, as ``mtp.tree_attention`` would see it (window ignored: it had none)."""

    src = subprocess.run(["git", "-C", str(PORT), "show", f"{MORNING}:src/tensorfold/cuda/kernels/attention.py"],
                         capture_output=True, text=True, check=True).stdout
    path = Path("/tmp/attention_morning.py")
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("attention_morning", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["attention_morning"] = old
    spec.loader.exec_module(old)
    view = types.SimpleNamespace(**{k: getattr(old, k) for k in dir(old) if not k.startswith("__")})
    view.attention = lambda *args, window=0, **kw: old.attention(*args, **kw)
    return view


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("traces")
    ap.add_argument("--context", type=int, default=64000)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    from tokenizers import Tokenizer

    from ab import agent_prompts
    from draft_window_ab import clean, transcripts
    from seg_ab import PROMPTS
    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe.cuda import decode, mtp
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    raw = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    tok = TemplateTokens(raw, ChatTemplate(Path(a.model)))
    prompts = [("short", list(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                                      enable_thinking=False)) if k == "chat"
                else raw.encode(t, add_special_tokens=False).ids) for k, t in PROMPTS + agent_prompts()]
    names = []
    for task, tools, messages in transcripts(a.traces):
        if task in names:
            continue
        messages, cut = [clean(m) for m in messages], None
        for n in range(2, len(messages) + 1):
            if messages[n - 1].get("role") != "tool":
                continue
            ids = tok.apply_chat_template(messages[:n], tools=tools, add_generation_prompt=True)
            if len(ids) > a.context:
                break
            cut = ids
        if cut is not None and len(cut) > 0.85 * a.context:
            prompts.append(("long", list(cut)))
            names.append(task)
        if len(names) == a.prompts:
            break
    print("prompts:", [(k, len(p)) for k, p in prompts], "long tasks:", names, flush=True)
    e = Qwen36Engine(a.model, context=a.context + a.tokens + 1024, context_explicit=True)
    new_attention = mtp.tree_attention
    variants = {"morning": (False, 0, morning_attention()), "now": (True, mtp.DRAFT_WINDOW, new_attention)}
    graphs = {v: ({}, {}, {}, {}) for v in variants}    # each variant's captured graphs (kernels baked in)
    sampling = Sampling(seed=1234, temperature=1.0, top_k=20, top_p=0.95)

    def run(v, p):
        decode.SAMPLED_CHAIN, mtp.DRAFT_WINDOW, mtp.tree_attention = variants[v]
        e.graphs.target, e.graphs.mtp, e.graphs.chains, e.graphs.commits = graphs[v]
        out, rows = [], e.graphs.rows
        s = e.generate(prompts[p][1], a.tokens, sampling, lambda t: out.extend(t) and False, stop_eos=False)
        if e.graphs.rows != rows:                       # the buffers moved: the other variant's graphs are stale too
            for d in (d for g in graphs.values() for d in g):
                d.clear()
        return s, out

    for p in range(len(prompts)):                       # warm: every graph captured (prompt outermost: one prefill each)
        for v in variants:
            run(v, p)
    agg = collections.defaultdict(lambda: [0.0, 0, 0, 0])
    outs = collections.defaultdict(dict)
    for rep in range(a.reps):
        for p in range(len(prompts)):
            for v in (list(variants) if rep % 2 == 0 else list(variants)[::-1]):
                s, out = run(v, p)
                g = agg[(v, p)]
                g[0] += s["decode_s"]; g[1] += s["rounds"]; g[2] += s["drafted"]; g[3] += s["accepted"]
                outs[p].setdefault(v, out)
    print("outputs identical:", all(len({tuple(o) for o in d.values()}) == 1 for d in outs.values()))
    tot = collections.defaultdict(lambda: [0.0, 0, 0, 0])
    for p, (kind, ids) in enumerate(prompts):
        line = f"prompt {p} ({kind}, {len(ids)} tokens):"
        for v in variants:
            sec, r, dr, ac = agg[(v, p)]
            for i, x in enumerate((sec, r, dr, ac)):
                tot[(kind, v)][i] += x
            line += f"  {v} {a.reps * a.tokens / sec:6.1f} tok/s ({1000 * sec / r:5.2f} ms/round, accepted {ac / dr:.1%})"
        print(line, flush=True)
    for kind in ("short", "long"):
        n = a.reps * a.tokens * sum(k == kind for k, _ in prompts)
        if not n:
            continue
        m, w = n / tot[(kind, "morning")][0], n / tot[(kind, "now")][0]
        print(f"{kind}: morning {m:.1f} tok/s, now {w:.1f} tok/s ({100 * (w / m - 1):+.1f}%), accepted "
              f"{tot[(kind, 'morning')][3] / tot[(kind, 'morning')][2]:.1%} -> {tot[(kind, 'now')][3] / tot[(kind, 'now')][2]:.1%}",
              flush=True)


if __name__ == "__main__":
    main()
