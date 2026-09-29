"""Per-task table for vf-eval Terminal-Bench 2 runs, plus a paired comparison of the solved sets.

    uv run summarize.py outputs/primeintellect/<run dir> [<run dir> ...]

Columns: reward, stop condition, model calls, seconds in the model vs in tools/harness, output tokens, the largest
single-call prompt (uncached tokens as llama-server reports them), median output per call, and decode rate per call
(output tokens over call wall time, so it includes prefill and queueing on a shared server).
"""
import json
import statistics as st
import sys
from itertools import combinations
from math import comb


def load(run_dir):
    rows = {}
    for line in open(f"{run_dir}/traces.jsonl"):
        r = json.loads(line)
        name = r["task"]["data"]["name"].split("/")[-1]
        if not r["traces"]:
            rows[name] = dict(reward=0, stop="no_trace", calls=0, model_s=0, tool_s=0, out=0, max_prompt=0, med_out=0, rate=0)
            continue
        t = r["traces"][0]
        agent = t["timing"].get("agent", {})
        calls = t["calls"]
        out = [c.get("usage", {}).get("completion_tokens", 0) for c in calls]
        prompt = [c.get("usage", {}).get("prompt_tokens", 0) for c in calls]
        rate = [o / (c["time"]["end"] - c["time"]["start"]) for o, c in zip(out, calls)
                if "time" in c and c["time"]["end"] > c["time"]["start"]]
        rows[name] = dict(
            reward=sum(v["score"] * v.get("weight", 1) for v in t["rewards"].values()),
            stop=t["stop_condition"], calls=len(calls),
            model_s=agent.get("model", {}).get("duration", 0), tool_s=agent.get("harness", {}).get("duration", 0),
            out=sum(out), max_prompt=max(prompt, default=0), med_out=st.median(out) if out else 0,
            rate=st.median(rate) if rate else 0)
    return rows


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b vs c discordant pairs."""
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


runs = {d.rstrip("/").split("/")[-1]: load(d) for d in sys.argv[1:]}
for label, rows in runs.items():
    print(f"== {label}")
    print(f"{'task':30} {'rew':>3} {'stop':15} {'calls':>5} {'model s':>7} {'tool s':>6} {'out tok':>7} "
          f"{'max prm':>7} {'med out':>7} {'tok/s':>5}")
    for name, x in sorted(rows.items()):
        print(f"{name:30} {x['reward']:3.0f} {x['stop']:15} {x['calls']:5} {x['model_s']:7.0f} {x['tool_s']:6.0f} "
              f"{x['out']:7} {x['max_prompt']:7} {x['med_out']:7.0f} {x['rate']:5.1f}")
    stops = {}
    for x in rows.values():
        stops[x["stop"]] = stops.get(x["stop"], 0) + 1
    print(f"solved {sum(x['reward'] > 0 for x in rows.values())}/{len(rows)}; stops {stops}; "
          f"model {sum(x['model_s'] for x in rows.values()):.0f} s, tools {sum(x['tool_s'] for x in rows.values()):.0f} s, "
          f"output {sum(x['out'] for x in rows.values())} tokens\n")

for a, b in combinations(runs, 2):
    common = runs[a].keys() & runs[b].keys()
    only_a = sorted(t for t in common if runs[a][t]["reward"] > 0 and not runs[b][t]["reward"] > 0)
    only_b = sorted(t for t in common if runs[b][t]["reward"] > 0 and not runs[a][t]["reward"] > 0)
    print(f"{a} vs {b}: only first {only_a}, only second {only_b}, exact McNemar p = {mcnemar(len(only_a), len(only_b)):.2f}")
