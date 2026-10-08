"""How R1s-SD uses prime_agent's features, from every saved prime_agent rollout under outputs/.

prime_agent offers one tool, `ipython`; everything else (edit, compact, sub-agents, bash handles, the continual
harness layer, skills) is a Python call inside it. So features are found by pattern in each ipython cell's code.

Per rollout: which features it used, how it ended, whether it was solved. Then, per feature: how many rollouts
used it, and the solve rate with and without it, overall and within tasks that have runs both ways (less confounded
by task difficulty). Runs lost to infrastructure (errors, harness_error) are left out of solve rates.

usage: python feature_audit.py [--out results/feature_audit]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "r1s-sd"

# name -> (pattern over a cell's code, what it means)
FEATURES = {
    "edit": (r"\bedit\s*\(\s*path\s*=|await\s+edit\s*\(", "the edit skill for file changes"),
    "compact": (r"\bcompact\s*[.(]", "context compaction"),
    "spawn": (r"rlm\.spawn\s*\(|rlm\.create_session\s*\(", "start a sub-agent"),
    "agent_msg": (r"\bagent_message\.|\bagent_observe\.|rlm\.collect\s*\(", "talk to or watch sub-agents"),
    "harness_layer": (r"rlm\.harness\.|rlm\.get_harness_state|\brefine\.\w+\s*\(", "memory/skills/refine layer"),
    "goal": (r"\bgoal\.\w+\s*\(", "the goal skill"),
    "websearch": (r"\bwebsearch\.\w+\s*\(|^\s*websearch\s", "web search skill"),
    "attach_image": (r"\battach_image", "look at an image"),
    "skill_docs": (r"SKILL\.md|help\(\s*(edit|compact|goal|refine|websearch|agent_message|agent_observe)", "read a skill's docs"),
    "bash_background": (r"(?<!await )(?<!await  )\b\w+\s*=\s*bash\s*\(", "start a shell job and keep its handle"),
    "bash_await": (r"await\s+bash\s*\(", "run a shell command and wait"),
    "py_file_io": (r"\bopen\s*\(|Path\([^)]*\)\.(read_text|write_text)|\.read_text\(|\.write_text\(", "read/write files in Python"),
}
# things the system prompt tells the model not to do
ANTI = {
    "subprocess": (r"\bsubprocess\.|os\.system\s*\(|os\.popen\s*\(", "subprocess/os.system instead of bash()"),
    "heredoc": (r"bash\s*\([^)]*<<", "heredoc inside bash()"),
    "shell_loop": (r"bash\s*\([^)]*\bfor\s+\w+\s+in\b[^)]*;\s*do", "shell loop inside bash()"),
    "sleep_poll": (r"time\.sleep\s*\(|asyncio\.sleep\s*\(|bash\s*\(\s*['\"]sleep\s", "polling with sleep"),
    "sed_inplace": (r"sed\s+-i", "sed -i edits"),
}
FEAT_RE = {k: re.compile(p, re.M) for k, (p, _) in FEATURES.items()}
ANTI_RE = {k: re.compile(p, re.M | re.S) for k, (p, _) in ANTI.items()}


def experiment(path: Path) -> str:
    """A short experiment name from the run directory."""

    s = str(path.relative_to(HERE / "outputs"))
    for key, name in [("tlhard", "Terminal-Lego hard"), ("tlpilot", "Terminal-Lego pilot"), ("tlsmoke", "Terminal-Lego smoke"),
                      ("lessons", "lessons loop"), ("plans/", "plan-first"), ("fixes-off", "fix test, off"),
                      ("fixes-on", "fix test, on"), ("fixsmoke", "fix smoke"), ("-c16-", "16K reply cap"),
                      ("-c32-", "32K reply cap"), ("-dg-", "round-2 data"), ("-big", "big budget"), ("-hftp-", "HF pod runs")]:
        if key in s:
            return name
    return "other"


def rollouts():
    for f in sorted((HERE / "outputs").rglob("traces.jsonl")):
        s = str(f).lower()
        if MODEL not in s or "prime_agent" not in s and "/lessons/" not in s and "/plans/" not in s:
            continue
        if "--pi" in s or "claude" in s:
            continue
        exp = experiment(f.parent)
        with open(f) as fh:
            for line in fh:
                try:
                    yield exp, f, json.loads(line)
                except json.JSONDecodeError:
                    continue


def cells(trace):
    """(tool name, code) for every tool call in a trace."""

    for node in trace.get("nodes", []):
        for tc in node.get("message", {}).get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or ""
            try:
                code = json.loads(args).get("code", "") if isinstance(args, str) else args.get("code", "")
            except (json.JSONDecodeError, AttributeError):
                code = args if isinstance(args, str) else ""
            yield name, code or ""


def analyse(rec):
    traces = rec.get("traces") or []
    if not traces:
        return None
    tr = traces[0]
    task = (rec.get("task") or {}).get("id") or (rec.get("task") or {}).get("name") or json.dumps(rec.get("task"))[:80]
    stop = tr.get("stop_condition")
    lost = bool(rec.get("errors")) or stop in ("harness_error", "provider_error", "sandbox_error")
    score = ((tr.get("rewards") or {}).get("solved") or {}).get("score")
    calls = tr.get("calls") or []
    used, anti, bad_tools = Counter(), Counter(), Counter()
    n_cells = 0
    for name, code in cells(tr):
        if name != "ipython":
            bad_tools[name] += 1
            continue
        n_cells += 1
        for k, rx in FEAT_RE.items():
            if rx.search(code):
                used[k] += 1
        for k, rx in ANTI_RE.items():
            if rx.search(code):
                anti[k] += 1
    prompts = [((c.get("usage") or {}).get("prompt_tokens") or 0) for c in calls]
    return {
        "task": str(task), "stop": stop, "lost": lost, "solved": bool(score and score > 0),
        "turns": len(calls), "cells": n_cells, "length_cut": sum(c.get("finish_reason") == "length" for c in calls),
        "ended_on_cut": bool(calls) and calls[-1].get("finish_reason") == "length",
        "peak_prompt": max(prompts) if prompts else 0, "n_traces": len(traces),
        "used": dict(used), "anti": dict(anti), "bad_tools": dict(bad_tools),
    }


def rate(rows):
    return (sum(r["solved"] for r in rows) / len(rows)) if rows else float("nan")


def within_task(rows, key, kind="used"):
    """Solve rate with vs without the feature, over tasks that have usable runs both ways (task-weighted)."""

    by = defaultdict(lambda: ([], []))
    for r in rows:
        by[r["task"]][0 if r[kind].get(key) else 1].append(r)
    pairs = [(rate(a), rate(b)) for a, b in by.values() if a and b]
    if not pairs:
        return None, 0
    return (sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs)), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "feature_audit"))
    args = ap.parse_args()
    rows = []
    for exp, f, rec in rollouts():
        a = analyse(rec)
        if a:
            a["experiment"] = exp
            rows.append(a)
    usable = [r for r in rows if not r["lost"]]
    lines = [f"# prime_agent feature use by R1s-SD\n",
             f"{len(rows)} rollouts read, {len(usable)} usable (not lost to infrastructure), "
             f"{len({r['task'] for r in usable})} tasks. Solve rate overall: {rate(usable):.1%}.\n",
             "## By experiment\n", "| Experiment | Usable runs | Solved |", "|---|---|---|"]
    for exp in sorted({r["experiment"] for r in usable}):
        rs = [r for r in usable if r["experiment"] == exp]
        lines.append(f"| {exp} | {len(rs)} | {rate(rs):.0%} |")

    def table(title, spec, kind):
        out = [f"\n## {title}\n", "| What | Meaning | Runs using it | Share | Solved with | Solved without | Same tasks: with / without (tasks) |",
               "|---|---|---|---|---|---|---|"]
        for k, (_, meaning) in spec.items():
            w = [r for r in usable if r[kind].get(k)]
            wo = [r for r in usable if not r[kind].get(k)]
            wt, n = within_task(usable, k, kind)
            wt_s = f"{wt[0]:.0%} / {wt[1]:.0%} ({n})" if wt else "–"
            out.append(f"| `{k}` | {meaning} | {len(w)} | {len(w) / len(usable):.1%} | "
                       f"{rate(w):.0%} | {rate(wo):.0%} | {wt_s} |")
        return out

    lines += table("Features (pattern found in at least one ipython cell)", FEATURES, "used")
    lines += table("Habits the system prompt warns against", ANTI, "anti")

    bad = Counter()
    for r in usable:
        bad.update(r["bad_tools"])
    runs_bad = [r for r in usable if r["bad_tools"]]
    wasted = sum(sum(r["bad_tools"].values()) for r in usable)
    lines += ["\n## Calls to tools that don't exist\n",
              f"{len(runs_bad)} runs ({len(runs_bad) / len(usable):.1%}) called a tool other than `ipython`: "
              + ", ".join(f"`{k}` x{v}" for k, v in bad.most_common(8)) + ".",
              f"Each got back \"Tool <name> not found\": {wasted:,} wasted turns, "
              f"{wasted / sum(r['turns'] for r in usable):.1%} of all turns.",
              f"Solved with: {rate(runs_bad):.0%}; without: {rate([r for r in usable if not r['bad_tools']]):.0%}."]

    stops = Counter(r["stop"] for r in usable)
    cut = [r for r in usable if r["ended_on_cut"]]
    peak = sorted(r["peak_prompt"] for r in usable)
    lines += ["\n## How runs end\n", "| Ending | Runs | Solved |", "|---|---|---|"]
    for s, n in stops.most_common():
        lines.append(f"| {s} | {n} | {rate([r for r in usable if r['stop'] == s]):.0%} |")
    lines += [f"\nRuns whose last model reply was cut off by the length cap: {len(cut)} ({len(cut) / len(usable):.1%}), "
              f"solved {rate(cut):.0%}.",
              f"Turns per run: median {sorted(r['turns'] for r in usable)[len(usable) // 2]}. "
              f"Peak prompt size: median {peak[len(peak) // 2]:,} tokens, 90th percentile {peak[int(len(peak) * 0.9)]:,}, "
              f"max {peak[-1]:,}; runs past 100K: {sum(p > 100_000 for p in peak)}.",
              f"Runs with more than one trace (sub-agents ran): {sum(r['n_traces'] > 1 for r in usable)}."]
    lines += ["\n## By peak prompt size (no run ever compacts its context)\n",
              "| Peak prompt | Runs | Solved | Ended: done / turn cap / time limit |", "|---|---|---|---|"]
    for lo, hi in [(0, 32_000), (32_000, 64_000), (64_000, 100_000), (100_000, 10**9)]:
        rs = [r for r in usable if lo <= r["peak_prompt"] < hi]
        c = Counter(r["stop"] for r in rs)
        span = f"{lo // 1000}K+" if hi == 10**9 else f"{lo // 1000}-{hi // 1000}K"
        lines.append(f"| {span} | {len(rs)} | {rate(rs):.0%} | {c['agent_completed']} / {c['max_turns']} / {c['agent_timeout']} |")
    Path(args.out + ".md").write_text("\n".join(lines) + "\n")
    Path(args.out + ".json").write_text(json.dumps(rows))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
