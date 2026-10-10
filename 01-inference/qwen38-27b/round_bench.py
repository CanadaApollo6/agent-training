"""In-process decode bench for Qwen3.8-27B on TensorFold (CUDA): where a draft round's time goes, per setting.

Loads the engine once (the daily 4-bit-cache build unless overridden by env), then for each max_rows in ROWS and each
prompt runs prefill + draft_decode with a per-round trace. Reports tok/s, tokens a round, and ms per stage (draft,
verify, sample, commit). Greedy replies must match across settings (drafting is exact); a mismatch is printed.

Usage (from envs/tensorfold):
  TENSORFOLD_KV_BITS=4 TENSORFOLD_ONE_BUFFER=1 uv run python ../qwen38-27b/round_bench.py \
      [--rows 12,16] [--tokens 512] [--temps 0,1] [--context 32768] [--out results/round-TAG.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

PROMPTS = [
    ("code", "Write a Python class for a bounded LRU cache with get and put methods, then add unit tests with pytest."),
    ("chat", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."),
    ("agent", "You are working in a git repository. The test `tests/test_parse.py::test_dates` fails with "
              "`ValueError: unconverted data remains: Z`. Explain the likely cause in `parse.py`, propose a fix, and "
              "show the patch as a unified diff."),
]


def snapshot(repo: str) -> Path:
    base = Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    return sorted(base.iterdir())[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Vontra/Qwen3.8-27B-MLX-4bit"))
    ap.add_argument("--drafter", default="z-lab/Qwen3.8-27B-DFlash2")
    ap.add_argument("--rows", default="12")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--temps", default="0,1")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--prompts", default=",".join(p[0] for p in PROMPTS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--prefix-tokens", type=int, default=0,
                    help="put this many tokens of the repo's READMEs before each task (a long conversation's context)")
    ap.add_argument("--branch", default="", help="draft_tree.POLICY branch values to sweep (1 = chains)")
    ap.add_argument("--serial", action="store_true", help="also decode greedy serially and check drafted == serial")
    ap.add_argument("--sync", default="1", help="draft_decode sync_stages values to sweep (0: no timing syncs)")
    ap.add_argument("--graphs", action="store_true", help="replay verify windows from CUDA graphs (graphs27)")
    a = ap.parse_args()

    # torch leaves its build lock files behind, and the next start waits on them forever (run one TensorFold at a time)
    for lock in (Path.home() / ".cache/torch_extensions").glob("*/tensorfold_*/lock"):
        lock.unlink()
    from tensorfold.cuda import capacity
    capacity.available_bytes = lambda torch_mod: 200 * capacity.GIB      # planner overcounts a 24 GB card (README)

    from tokenizers import Tokenizer
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
    from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill, serial_decode

    model_dir, draft_dir = snapshot(a.model), snapshot(a.drafter)
    tok, template = Tokenizer.from_file(str(model_dir / "tokenizer.json")), ChatTemplate(model_dir)
    rows_list = [int(r) for r in a.rows.split(",")]
    t0 = time.perf_counter()
    eng = Qwen27Engine(model_dir, draft_dir, max_rows=max(rows_list), context=a.context, context_explicit=True)
    print(f"loaded in {time.perf_counter() - t0:.0f} s; window {eng.context_window}; "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)

    wanted = a.prompts.split(",")
    prompts = []
    prefix = ""
    if a.prefix_tokens:                                                  # frozen text: the READMEs as of this commit
        root = Path(__file__).resolve().parents[2]
        docs = "\n\n".join(p.read_text() for p in sorted(root.glob("*/**/README.md"))[:200])
        prefix = tok.decode(tok.encode(docs, add_special_tokens=False).ids[:a.prefix_tokens])
        prefix = "Project notes for context:\n\n" + prefix + "\n\nNow the task.\n\n"
    for name, text in PROMPTS:
        if name in wanted:
            rendered = template.render([{"role": "user", "content": prefix + text}], tools=None, enable_thinking=True,
                                       extra={"reasoning_effort": "medium"})
            prompts.append((name, tok.encode(rendered, add_special_tokens=False).ids))

    graphs = None
    if a.graphs:
        from tensorfold.families.qwen3_5.cuda.graphs27 import TreeGraphs
        graphs = TreeGraphs(eng.w)
        graphs.check = os.environ.get("GRAPH_CHECK") == "1"
        if os.environ.get("DRAFT_GRAPHS", "1") == "1":
            from tensorfold.families.qwen3_5.cuda.graphs27 import DraftGraphs
            dgraphs = DraftGraphs(eng.draft)
            dgraphs.check = graphs.check
            if os.environ.get("TAP_GRAPHS", "1") == "1":
                dgraphs.install()
            else:
                eng.draft.launch_block = dgraphs.launch_block
    temps = [float(t) for t in a.temps.split(",")]
    results, greedy_ref = [], {}

    def run(rows: int, name: str, ids: list[int], temp: float, prof=None, sync: bool = True):
        smp = None if temp <= 0 else Sampling(seed=1234, temperature=temp)
        eng.draft.restore(([None] * eng.draft.layers, [None] * eng.draft.layers, 0, 0))
        st, pending = prefill(eng.w, ids, smp, eng.draft, limit=eng.context_window)
        torch.cuda.synchronize()
        if prof is not None:
            prof.start()
        trace: list = []
        r = draft_decode(eng.w, st, ids, pending, a.tokens, smp, eng.draft, max_rows=rows, trace=trace, inplace=True,
                         graphs=graphs, sync_stages=sync)
        return r, trace

    timing = collections.defaultdict(float)
    if os.environ.get("TIMING") == "1":                                  # synced wall time of each drafter/commit call
        import tensorfold.families.qwen3_5.cuda.decode as dec
        def timed(obj, name, label):
            fn = getattr(obj, name)
            def inner(*args, **kw):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = fn(*args, **kw)
                torch.cuda.synchronize()
                timing[label] += time.perf_counter() - t
                return out
            setattr(obj, name, inner)
        for name in ("launch_block", "finish_tree", "add_taps"):
            timed(eng.draft, name, "draft." + name)
        timed(dec, "commit", "commit")
        timed(dec, "sample_rows", "sample_rows")
        timed(dec, "tree_forward", "tree_forward")
    for rows in rows_list:                                               # warm-up: kernel builds, autotune per width
        run(rows, *prompts[0], 0.0)
    if a.serial:
        for name, ids in prompts:
            eng.draft.restore(([None] * eng.draft.layers, [None] * eng.draft.layers, 0, 0))
            st, pending = prefill(eng.w, ids, None, eng.draft, limit=eng.context_window)
            r = serial_decode(eng.w, st, pending, a.tokens, None)
            greedy_ref[(name,)] = r.tokens
            print(json.dumps({"serial": name, "tokens": len(r.tokens) - 1, "tok_s": round(r.tokens_per_second, 1)}),
                  flush=True)
            del st, pending, r                                           # a long window's cache, twice, won't fit
            torch.cuda.empty_cache()
    if os.environ.get("DRAFT_PROFILE") == "1":                           # the drafter's launch alone, kernels by name
        from torch.profiler import ProfilerActivity, profile
        run(rows_list[0], *prompts[0], 0.0)                              # leaves a full drafter context
        calls = 50
        for _ in range(5):
            eng.draft.launch_block(1000, rows_list[0] - 1)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(calls):
            eng.draft.launch_block(1000, rows_list[0] - 1)
        torch.cuda.synchronize()
        print(f"launch_block: {(time.perf_counter() - t) * 1000 / calls:.3f} ms a call (synced loop)", flush=True)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                eng.draft.launch_block(1000, rows_list[0] - 1)
            torch.cuda.synchronize()
        avg = prof.key_averages()
        gpu_us = sum(getattr(e, "self_device_time_total", 0) for e in avg)
        print(f"GPU kernel time {gpu_us / 1000 / calls:.3f} ms a call", flush=True)
        for e in sorted(avg, key=lambda e: -getattr(e, "self_device_time_total", 0))[:40]:
            t = getattr(e, "self_device_time_total", 0)
            if t:
                print(f"{t / calls:8.1f} us/call {e.count / calls:6.1f} calls  {e.key[:100]}")
        return
    if os.environ.get("PROFILE") == "1":                                 # kernel time by name over one short decode
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        r, _ = run(rows_list[0], *prompts[0], 0.0, prof)
        torch.cuda.synchronize()
        prof.stop()
        ev = [e for e in prof.key_averages() if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0)]
        gpu_us = sum(getattr(e, "self_device_time_total", 0) for e in prof.key_averages())
        print(f"profiled: {len(r.tokens) - 1} tokens, {r.rounds} rounds, wall {r.seconds * 1000:.0f} ms, "
              f"GPU kernel time {gpu_us / 1000:.0f} ms ({gpu_us / 1000 / (r.seconds * 1000):.0%} of wall)", flush=True)
        rows_ = sorted(prof.key_averages(), key=lambda e: -getattr(e, "self_device_time_total", 0))[:50]
        for e in rows_:
            t = getattr(e, "self_device_time_total", 0)
            if t:
                print(f"{t / 1000 / r.rounds:8.3f} ms/round {e.count / r.rounds:7.1f} calls/round  {e.key[:90]}")
        return
    from tensorfold.families.qwen3_5.cuda import draft_tree
    branches = [int(b) for b in a.branch.split(",")] if a.branch else [draft_tree.POLICY["branch"]]
    syncs = [s == "1" for s in a.sync.split(",")]
    for rows, branch, sync in [(r, b, s) for r in rows_list for b in branches for s in syncs]:
        draft_tree.POLICY["branch"] = branch
        for name, ids in prompts:
            for temp in temps:
                for rep in range(a.reps):
                    timing.clear()
                    r, trace = run(rows, name, ids, temp, sync=sync)
                    n = len(r.tokens) - 1
                    rec = {
                        "rows": rows, "branch": branch, "sync": int(sync), "prompt": name, "prompt_tokens": len(ids), "temp": temp, "rep": rep, "tokens": n,
                        "tok_s": round(n / r.seconds, 1), "rounds": r.rounds,
                        "tok_per_round": round(n / r.rounds, 2),
                        "copy_rounds": sum(t["source"] == "copy" for t in trace),
                        **{f"{k}_ms": round(1000 * getattr(r, f"{k}_seconds") / r.rounds, 2)
                           for k in ("draft", "verify", "sample", "commit")},
                        "width_mean": round(statistics.mean(r.widths), 1),
                        **{f"t.{k}": round(1000 * v / r.rounds, 2) for k, v in timing.items()},
                    }
                    if graphs is not None:
                        rec.update(graphs=len(graphs.graphs), eager=graphs.eager, mismatches=graphs.mismatches,
                                   commit_checks=graphs.commit_checks, commit_mismatches=graphs.commit_mismatches)
                        if os.environ.get("DRAFT_GRAPHS", "1") == "1":
                            rec.update(draft_mismatches=dgraphs.mismatches, draft_replays=dgraphs.replays,
                                       tap_replays=dgraphs.tap_replays, tap_mismatches=dgraphs.tap_mismatches)
                    if temp <= 0:
                        key = (name,)
                        if key in greedy_ref and greedy_ref[key] != r.tokens:
                            same = next((i for i, (x, y) in enumerate(zip(greedy_ref[key], r.tokens)) if x != y), None)
                            rec["MISMATCH_at"] = same
                        greedy_ref.setdefault(key, r.tokens)
                    results.append(rec)
                    print(json.dumps(rec), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=1))


if __name__ == "__main__":
    main()
