"""Compare reasoning-probe runs against a baseline, problem by problem.

    python3 compare.py <baseline label> <label> [<label> ...]   (labels of results/<label>.json)

With several samples per problem, each problem's score is its number of correct attempts (0..samples). The test is
an exact sign-flip permutation test on the per-problem score differences: under "no effect", each problem's
difference is as likely to be negative as positive. "Loopy" counts traces where more than 20% of the substantial
lines (over 15 characters) repeat an earlier line, the same measure used on the 27B and 9B.
"""
import collections
import gzip
import itertools
import json
import statistics as st
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def loopiness(text):
    lines = [x.strip() for x in text.split("\n") if len(x.strip()) > 15]
    counts = collections.Counter(lines)
    return sum(n - 1 for n in counts.values() if n > 1) / max(len(lines), 1)


def load(label):
    summary = json.loads((RESULTS / f"{label}.json").read_text())
    per = collections.defaultdict(list)
    for p in summary["per_problem"]:
        per[p["id"]].append(p)
    loopy = sum(loopiness(json.loads(line)["output"]) > 0.2 for line in gzip.open(RESULTS / f"{label}.traces.jsonl.gz", "rt"))
    return summary, per, loopy


def sign_flip_p(diffs):
    diffs = [d for d in diffs if d]
    observed = abs(sum(diffs))
    hits = sum(abs(sum(s * d for s, d in zip(signs, diffs))) >= observed
               for signs in itertools.product((1, -1), repeat=len(diffs)))
    return hits / 2 ** len(diffs) if diffs else 1.0


base_label, *labels = sys.argv[1:]
base, base_per, base_loopy = load(base_label)
print(f"{'build':28} {'correct':>9} {'wrong':>5} {'cap':>4} {'loopy':>5} {'med tok':>7} {'lost':>4} {'gained':>6} "
      f"{'p':>5}  median tokens, correct answers on problems both solve")
for label in [base_label, *labels]:
    s, per, loopy = load(label)
    score = {pid: sum(a["outcome"] == "correct" for a in attempts) for pid, attempts in per.items()}
    base_score = {pid: sum(a["outcome"] == "correct" for a in attempts) for pid, attempts in base_per.items()}
    diffs = [score[pid] - base_score[pid] for pid in base_score]
    both = [pid for pid in base_score if score[pid] and base_score[pid]]

    def med(p):
        return st.median(a["tokens"] for pid in both for a in p[pid] if a["outcome"] == "correct")

    print(f"{label:28} {s['correct']:4}/{s['n']:<4} {s['wrong']:5} {s['runaway']:4} {loopy:5} {s['median_tokens']:7} "
          f"{-sum(d for d in diffs if d < 0):4} {sum(d for d in diffs if d > 0):6} {sign_flip_p(diffs):5.3f}  "
          f"{med(base_per):,.0f} -> {med(per):,.0f} ({len(both)} problems)")
