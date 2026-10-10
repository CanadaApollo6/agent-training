"""Pilot vs continuation on one scale: per-task solve rate within 30 min (attempts over 30 min count as fails, as under
the pilot's cap), by policy stage. Each run drops attempts dispatched in its last cap+1 minutes (see pool_progress.py).

    python compare_runs.py PILOT_TRACES CONT_TRACES     # .../monitors/file/traces dirs
"""
import json, collections, statistics as st, sys
def load(D, cut_min):
    idx = {}
    for l in open(f"{D}/stream.index.jsonl"):
        if l.strip(): x = json.loads(l); idx[x["id"]] = x
    recs = []
    for line in open(f"{D}/stream/00000.jsonl"):
        r = json.loads(line)
        if r["run"]["work"]["type"] != "train" or r["id"] not in idx: continue
        x = idx[r["id"]]
        s = (r["traces"][0].get("rewards") or {}).get("solved"); s = s.get("score") if isinstance(s, dict) else s
        recs.append(dict(task=r["task"]["data"]["name"].split("/")[-1], d=x["dispatch"], dur=x.get("duration") or 0,
                         to=bool(r.get("is_timeout") or x.get("timeout")), pol=r["run"]["work"]["policy"]["start"], s=bool(s)))
    tend = max(x["arrival"] for x in idx.values())
    return [r for r in recs if r["d"] <= tend - cut_min * 60]
P = load(sys.argv[1], 31); C = load(sys.argv[2], 61)
def ok30(r): return r["s"] and not r["to"] and r["dur"] <= 1800
stages = {"pilot early (base, v0-2)": [r for r in P if r["pol"] <= 2], "pilot late (v6-11)": [r for r in P if r["pol"] >= 6],
          "cont early (pilot wts, v0-2)": [r for r in C if r["pol"] <= 2], "cont late (v6-11)": [r for r in C if r["pol"] >= 6]}
rate = {k: {} for k in stages}
for k, rs in stages.items():
    by = collections.defaultdict(list)
    for r in rs: by[r["task"]].append(ok30(r))
    rate[k] = {t: sum(v) / len(v) for t, v in by.items()}
common = set.intersection(*[set(v) for v in rate.values()])
print(f"tasks in all four stages: {len(common)}")
for k in stages:
    n = len(stages[k]); print(f"{k:30s} {n:4d} attempts  pooled {100*sum(map(ok30,stages[k]))/n:5.1f}%  per-task mean (common) {100*st.mean(rate[k][t] for t in common):5.1f}%  (all its tasks: {100*st.mean(rate[k].values()):5.1f}%, {len(rate[k])} tasks)")
base = "pilot early (base, v0-2)"
for k in list(stages)[1:]:
    d = [rate[k][t] - rate[base][t] for t in common]
    print(f"  {k} vs base: {100*st.mean(d):+5.1f} pts per task, {sum(x>0 for x in d)} up, {sum(x<0 for x in d)} down")
