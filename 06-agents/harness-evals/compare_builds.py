"""Terminal-Bench 2 pilot, builds side by side: solves per attempt, per task, paired against a reference build.

    uv run compare_builds.py [--family ornith35b|lfm25] [--ref q8] [--drop qemu-startup]

Tables: solves per attempt and per task, a paired test against the reference, length on shared solves (geometric mean
of per-task median solved-run tokens, bootstrap CI), and how the failed runs ended.

A build is a label prefix: its attempts are the output runs ``terminal-bench-2--<family>-<build>-a<N>--pi--*``; an
attempt split over several runs (one server a lane, each lane a share of the tasks) is their union. Runs set aside (renamed ``broken-*``) are skipped. Tasks that fail before the agent starts on every build (HarnessError) are
dropped with ``--drop``. The paired test permutes build labels within each task: it asks whether the per-task solve
rates differ by more than swapping labels would give.
"""
import argparse
import glob
import json
import math
import random
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).parent / "outputs" / "primeintellect"
FAMILIES = {
    "ornith35b": {"q8": "Q8_0 (pod, llama.cpp)", "q4km": "Q4_K_M (pod, llama.cpp)",
                  "r2-c4s": "R2: gate/up 2-bit, down 3-bit (3090, TensorFold)",
                  "r1-c4s": "R1: all experts 3-bit (3090, TensorFold)",
                  "r1s-c4s": "R1s: R1 + imatrix-searched 4-bit always-on path (3090, TensorFold)"},
    "lfm25": {"q8": "Liquid's Q8_0 (pod, llama.cpp)",
              "m3tis": "M3TiS #3: Liquid's MLX 4-bit + 4-bit table (sm_86 pod, TensorFold Zig)"},
    # across families: a key "<family>:<build>" reads that family's runs
    "flint": {"ornith35b:q8": "Ornith 35B Q8_0 (pod, llama.cpp)",
              "ornith35b:r1s-c4s": "Ornith 35B R1s (3090, TensorFold)",
              "flint:bf16": "Flint BF16 (H200, Recursive's SGLang kit)"},
    "nanbeige": {"ornith35b:q8": "Ornith 35B Q8_0 (pod, llama.cpp)",
                 "ornith35b:r1s-c4s": "Ornith 35B R1s (3090, TensorFold)",
                 "flint:bf16": "Flint BF16 (H200, Recursive's SGLang kit)",
                 "nanbeige:bf16": "Nanbeige4.3-33B BF16 (H200 per attempt, vLLM)"},
}
BUILDS = FAMILIES["ornith35b"]


def attempts(build, family="ornith35b"):
    if ":" in build:
        family, build = build.split(":")
    runs = {}
    for d in sorted(glob.glob(str(OUT / f"terminal-bench-2--{family}-{build}-a*--pi--*"))):
        m = re.search(rf"{family}-{re.escape(build)}-a(\d+)(?:l\d+)?--", d)
        rows = {}
        for line in open(Path(d) / "traces.jsonl"):
            r = json.loads(line)
            t = r["traces"][0] if r.get("traces") else {}
            name = (t.get("task", {}).get("data", {}).get("name") or json.dumps(r.get("task")))
            name = name.split("/")[-1].strip('"')
            rewards = t.get("rewards", {})  # older vf-eval names the reward "solved", newer "reward"
            score = (rewards.get("solved") or rewards.get("reward") or {}).get("score", 0.0)
            rows[name] = {"solved": score >= 1.0, "turns": len(t.get("calls", [])), "out": r.get("num_output_tokens"),
                          "stop": t.get("stop_condition")}
        runs.setdefault(int(m[1]), {}).update(rows)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="ornith35b", choices=FAMILIES)
    ap.add_argument("--ref", default="q8")
    ap.add_argument("--drop", nargs="*", default=["qemu-startup"])
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--fail", nargs="*", default=[], metavar="BUILD:ATTEMPT:TASK",
                    help="score these runs as failed, e.g. runs that looked the benchmark up (lookup_audit.py)")
    args = ap.parse_args()
    global BUILDS
    BUILDS = FAMILIES[args.family]
    data = {b: attempts(b, args.family) for b in BUILDS}
    data = {b: a for b, a in data.items() if a}
    for spec in args.fail:
        build, attempt, task = spec.rsplit(":", 2)
        data[build][int(attempt)][task]["solved"] = False
    tasks = sorted({t for a in data.values() for run in a.values() for t in run} - set(args.drop))
    rate = {b: {t: st.mean(run.get(t, {"solved": False})["solved"] for run in a.values()) for t in tasks}
            for b, a in data.items()}
    print(f"{len(tasks)} tasks (dropped: {', '.join(args.drop)})\n")
    print(f"| Build | Attempts (solved of {len(tasks)}) | Mean | Median output tokens a task |")
    print("|---|---|---|---|")
    for b, a in data.items():
        per = [sum(run.get(t, {"solved": False})["solved"] for t in tasks) for _, run in sorted(a.items())]
        toks = [run[t]["out"] for run in a.values() for t in tasks if t in run and run[t]["out"]]
        print(f"| {BUILDS[b]} | {', '.join(map(str, per))} | {st.mean(per):.1f} ({100 * st.mean(per) / len(tasks):.0f}%) "
              f"| {int(st.median(toks)) if toks else '-'} |")
    print("\n| Task | " + " | ".join(data) + " |")
    print("|---|" + "---|" * len(data))
    for t in tasks:
        print(f"| {t} | " + " | ".join(f"{sum(run.get(t, {'solved': False})['solved'] for run in a.values())}/{len(a)}"
                                       for a in data.values()) + " |")
    rng = random.Random(0)
    print(f"\nPaired against {args.ref} (per-task solve rates, permutation test):")
    for b in data:
        if b == args.ref or args.ref not in data:
            continue
        diffs = [rate[b][t] - rate[args.ref][t] for t in tasks]
        obs = sum(diffs)
        hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(obs) - 1e-9
                   for _ in range(args.perms))
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        print(f"- {b}: {obs / len(tasks) * 100:+.1f} points a task on average; better on {wins} tasks, worse on "
              f"{losses}; p = {hits / args.perms:.3f}")

    # Length on shared solves: per task, the median output tokens of each build's solved runs, as a ratio to the
    # reference; geometric mean over tasks both builds solved at least once, 95% bootstrap CI over those tasks.
    def solved_median(a, t):
        toks = [run[t]["out"] for run in a.values() if t in run and run[t]["solved"] and run[t]["out"]]
        return st.median(toks) if toks else None
    print(f"\n| Build | Solved-run tokens vs {args.ref} | 95% CI | Longer on | Output tokens per call (median run) |")
    print("|---|---|---|---|---|")
    for b, a in data.items():
        runs = [run[t] for run in a.values() for t in tasks if t in run]
        per_call = st.median(r["out"] / r["turns"] for r in runs if r["turns"] and r["out"])  # a run's tokens per call
        if b == args.ref:
            print(f"| {b} | - | - | - | {per_call:.0f} |")
            continue
        logs = [math.log(solved_median(a, t) / solved_median(data[args.ref], t)) for t in tasks
                if solved_median(a, t) and solved_median(data[args.ref], t)]
        boots = sorted(math.exp(st.mean(rng.choice(logs) for _ in logs)) for _ in range(5000))
        print(f"| {b} | {math.exp(st.mean(logs)):.2f}x | {boots[125]:.2f}-{boots[4874]:.2f} "
              f"| {sum(x > 0 for x in logs)} of {len(logs)} tasks | {per_call:.0f} |")

    # How the failed runs ended: a wrong answer (the agent finished), the turn cap, the time cap, or anything else.
    ends = {"agent_completed": "Gave a wrong answer", "max_turns": "Hit the turn cap", "agent_timeout": "Hit the time cap"}
    print("\n| Build | " + " | ".join(ends.values()) + " | Other |")
    print("|---|" + "---|" * (len(ends) + 1))
    for b, a in data.items():
        stops = [run[t]["stop"] for run in a.values() for t in tasks if t in run and not run[t]["solved"]]
        print(f"| {b} | " + " | ".join(str(stops.count(k)) for k in ends)
              + f" | {sum(x not in ends for x in stops)} |")


if __name__ == "__main__":
    main()
