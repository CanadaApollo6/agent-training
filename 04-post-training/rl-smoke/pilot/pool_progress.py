"""Pilot go/no-go, part 1: did the 16 pool tasks get solved more often as training went on?

Reads a prime-rl run's trace stream (stream/00000.jsonl + stream.index.jsonl) and compares attempts by the policy
version that made them: early (v0-2), mid (v3-5), late (v6+). Attempts dispatched in the last 31 minutes are left out:
the run stopped before slow ones among them could finish or time out, so keeping them would make late versions look
faster and better than they are. Rates are given two ways: of all attempts (a 30-minute timeout counts as a fail), and
of attempts that finished.

    python pool_progress.py OUTPUTS/<run>/monitors/file/traces
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1])
idx = {}
for l in open(D / "stream.index.jsonl"):
    if l.strip():
        x = json.loads(l)
        idx[x["id"]] = x
recs = []
for line in open(D / "stream/00000.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r["run"]["work"]["type"] != "train" or r["id"] not in idx:
        continue
    x = idx[r["id"]]
    s = (r["traces"][0].get("rewards") or {}).get("solved")
    s = s.get("score") if isinstance(s, dict) else s
    recs.append(dict(task=r["task"]["data"]["name"].split("/")[-1], d=x["dispatch"], pol=r["run"]["work"]["policy"]["start"],
                     to=x["timeout"], ok=bool(s and s > 0.5) and not x["timeout"]))
tend = max(x["arrival"] for x in idx.values())
kept = [r for r in recs if r["d"] <= tend - 31 * 60]
print(f"{len(kept)} of {len(recs)} training attempts kept (left out: dispatched in the run's last 31 min)")
B = lambda p: "early v0-2" if p <= 2 else ("mid v3-5" if p <= 5 else "late v6+")
agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))  # solved, finished, all
for r in kept:
    c = agg[r["task"]][B(r["pol"])]
    c[0] += r["ok"]; c[1] += not r["to"]; c[2] += 1
print(f"\n{'task':34s} {'early: solved/all (finished)':>30s} {'late: solved/all (finished)':>30s}")
for t in sorted(agg):
    e, l = agg[t]["early v0-2"], agg[t]["late v6+"]
    print(f"{t:34s} {f'{e[0]}/{e[2]} ({e[1]})':>30s} {f'{l[0]}/{l[2]} ({l[1]})':>30s}")
print()
for k in ("early v0-2", "mid v3-5", "late v6+"):
    s = sum(agg[t][k][0] for t in agg); f = sum(agg[t][k][1] for t in agg); a = sum(agg[t][k][2] for t in agg)
    print(f"{k:11s} {a:3d} attempts: solved {s / a:.0%} of all, {s / f:.0%} of finished; timeouts {(a - f) / a:.0%}")
dA, dF, up, dn = [], [], 0, 0
for t in agg:
    e, l = agg[t]["early v0-2"], agg[t]["late v6+"]
    if e[2] >= 4 and l[2] >= 4 and e[1] and l[1]:
        dA.append(l[0] / l[2] - e[0] / e[2]); dF.append(l[0] / l[1] - e[0] / e[1])
        up += l[0] / l[2] > e[0] / e[2]; dn += l[0] / l[2] < e[0] / e[2]
print(f"per-task mean change early -> late ({len(dA)} tasks): {st.mean(dA) * 100:+.1f} pts of all attempts, "
      f"{st.mean(dF) * 100:+.1f} pts of finished; {up} tasks up, {dn} down")
