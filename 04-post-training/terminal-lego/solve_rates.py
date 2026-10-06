"""Per-task solve rates of R1s-SD under prime_agent on Terminal-Lego tasksets, from vf-eval traces (task names come
from the traces: the log's task index doesn't follow the taskset file order).

    python solve_rates.py pilot hard

For each taskset NAME, reads ../../06-agents/harness-evals/outputs/terminal-lego--*-tl<NAME>*/traces.jsonl and
results/taskset_<NAME>.txt (difficulty, category). An attempt that errored (sandbox or server failure, no reward) is
an infra fail, kept apart from the model's own misses. Writes results/solve_rates_<NAME>.jsonl (one line per task)
and prints the bands: always solved, sometimes (the RL candidates: some signal to learn from), never, infra only.
"""
import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "../../06-agents/harness-evals/outputs"


def reward(t: dict) -> float:
    return sum((v or {}).get("score", 0) * (v or {}).get("weight", 1) for v in (t.get("rewards") or {}).values())


def rates(name: str) -> list[dict]:
    meta = {l.split()[0]: l.split()[1:] for l in open(HERE / "results" / f"taskset_{name}.txt") if l.strip()}
    tasks = collections.defaultdict(lambda: {"solved": 0, "failed": 0, "infra": 0, "turns": []})
    for f in glob.glob(str(OUT / f"terminal-lego--*-tl{name}*--*/traces.jsonl")):
        for line in open(f):
            r = json.loads(line)
            t = r["traces"][0] if r.get("traces") else None
            rec = tasks[r["task"]["data"]["name"].split("/")[-1]]
            if t is None or r.get("errors") or t.get("errors"):
                rec["infra"] += 1
            elif reward(t) > 0:
                rec["solved"] += 1
                rec["turns"].append(len(t["calls"]))
            else:
                rec["failed"] += 1
    rows = []
    for task, rec in sorted(tasks.items()):
        diff, cat = (meta.get(task) or ["?", "?"])[:2]
        n = rec["solved"] + rec["failed"]
        band = ("infra" if n == 0 else "always" if rec["failed"] == 0 else "never" if rec["solved"] == 0
                else "sometimes")
        rows.append({"task": task, "difficulty": diff, "category": cat, "band": band, **rec})
    return rows


def main():
    for name in sys.argv[1:]:
        rows = rates(name)
        with open(HERE / "results" / f"solve_rates_{name}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        att = sum(r["solved"] + r["failed"] + r["infra"] for r in rows)
        print(f"## {name}: {len(rows)} tasks, {att} attempts, {sum(r['solved'] for r in rows)} solved, "
              f"{sum(r['infra'] for r in rows)} infra fails")
        for diff in sorted({r["difficulty"] for r in rows}):
            c = collections.Counter(r["band"] for r in rows if r["difficulty"] == diff)
            print(f"  {diff:7} " + "  ".join(f"{b} {c[b]}" for b in ("always", "sometimes", "never", "infra")))
        print("  sometimes:", " ".join(r["task"] for r in rows if r["band"] == "sometimes"))


if __name__ == "__main__":
    main()
