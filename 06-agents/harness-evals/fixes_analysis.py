"""Fix test analysis (fixes_full.sh + fixes_rerun.sh): 87 TB2 tasks, 2 tries, prime_agent, R1s-SD, fixes off vs on.

Runs from the original servers (on-p1 kept to task 25; p1b ran the rest) plus the reruns of lost runs. A run is
"usable" unless it ended on an infrastructure error (sandbox terminated, server or harness error, the hung off-p2 run).
Image-request endings (fixes off only: the server rejects image input) are the model's own failure, so they count.
Each fix firing is tied to its run by time: the proxy logs when it answered, and every trace records each model
call's start and end on the same server.

    python3 fixes_analysis.py            # prints the tables, writes results/fixes_test.json
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "primeintellect"
ERRORS = {"ProviderError", "HarnessError", "SandboxError"}
DONE = re.compile(r"rollout done: id=(\S+) task=(\d+) reward=(\S+) turns=(\d+) stop=(\S+)")
HEAVY = {"compile-compcert", "gpt2-codegolf", "path-tracing", "mteb-retrieve", "portfolio-optimization",
         "torch-pipeline-parallelism", "train-fasttext"}  # Prime terminates their sandbox most runs, both sides
# (server name, tasks file, keep task index <= this)
SOURCES = {
    "off": [("fixes-off-p1", "tb2_noqemu_p1.txt", None), ("fixes-off-p2", "tb2_noqemu_p2.txt", None),
            ("fixes-off-r1", "tb2_lost_off_s1.txt", None), ("fixes-off-r2", "tb2_lost_off_s2.txt", None)],
    "on": [("fixes-on-p1", "tb2_noqemu_p1.txt", 25), ("fixes-on-p1b", "tb2_noqemu_p1b.txt", None),
           ("fixes-on-p2", "tb2_noqemu_p2.txt", None),
           ("fixes-on-r1", "tb2_lost_on_s1.txt", None), ("fixes-on-r2", "tb2_lost_on_s2.txt", None)],
}
HUNG = {"fixes-off-p2": [21]}  # never logged "rollout done"; stopped by hand


def load_traces(name):
    """id -> trace, for one server's output dir."""
    by_id = {}
    for path in OUT.glob(f"terminal-bench-2--ornith35b-r1s-sd-{name}--*/traces.jsonl"):
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            t = (obj.get("traces") or [None])[0]
            if isinstance(t, dict):
                t["_errors"] = json.dumps(obj.get("errors") or t.get("errors") or "")
                by_id[t["id"]] = t
    return by_id


def load_runs():
    runs = []
    for arm, sources in SOURCES.items():
        for name, tasks_file, last in sources:
            log = HERE / "logs" / f"tb2full-ornith35b-r1s-sd-{name}-prime_agent.log"
            if not log.exists():
                continue
            tasks = (HERE / tasks_file).read_text().split()
            traces = load_traces(name)
            for m in DONE.finditer(log.read_text()):
                rid, i, reward, turns, stop = m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4)), m.group(5)
                if last is not None and i > last:
                    continue
                t = traces.get(rid, {})
                if "image input" in t.get("_errors", ""):
                    stop = "image_request"
                calls = t.get("calls") or []
                runs.append({"arm": arm, "server": name, "id": rid, "task": tasks[i], "solved": reward >= 0.999,
                             "turns": turns, "stop": stop, "usable": stop not in ERRORS,
                             "rerun": name.endswith(("-r1", "-r2")),
                             "last_finish": calls[-1].get("finish_reason") if calls else None,
                             "calls": [(c["time"]["start"], c["time"]["end"]) for c in calls if c.get("time")],
                             "fixes": Counter()})
            for i in HUNG.get(name, []):
                runs.append({"arm": arm, "server": name, "id": None, "task": tasks[i], "solved": False, "turns": 0,
                             "stop": "hung", "usable": False, "rerun": False, "last_finish": None, "calls": [],
                             "fixes": Counter()})
    return runs


def attach_fixes(runs):
    """Tie each logged fix to the run whose model call was in flight when the proxy answered."""
    unmatched = Counter()
    by_server = defaultdict(list)
    for r in runs:
        by_server[r["server"]].append(r)
    for server, rs in by_server.items():
        path = HERE / "logs" / f"fixes-{server}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            f = json.loads(line)
            # the proxy logs as it answers, so the call that ended nearest that moment is the one it fixed
            gaps = sorted((min(abs(e - f["time"]) for s, e in r["calls"]), k) for k, r in enumerate(rs) if r["calls"])
            if gaps and gaps[0][0] <= 3 and (len(gaps) == 1 or gaps[1][0] > gaps[0][0] + 0.5):
                rs[gaps[0][1]]["fixes"][f["fix"]] += 1
            else:
                unmatched[(f["fix"], "none" if not gaps or gaps[0][0] > 3 else "several")] += 1
    return unmatched


def per_task(runs, arm, skip=()):
    out = defaultdict(list)
    for r in runs:
        if r["arm"] == arm and r["usable"] and r["task"] not in skip:
            out[r["task"]].append(r["solved"])
    return out


def paired(runs, skip=(), seed=0):
    off, on = per_task(runs, "off", skip), per_task(runs, "on", skip)
    both = sorted(set(off) & set(on))
    diffs = [sum(on[t]) / len(on[t]) - sum(off[t]) / len(off[t]) for t in both]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    # sign-flip permutation test and bootstrap CI over tasks
    flips = sum(abs(sum(d if rng.random() < .5 else -d for d in diffs) / len(diffs)) >= abs(mean) - 1e-12
                for _ in range(20000))
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(20000))
    return {"tasks": len(both), "off_rate": sum(sum(off[t]) / len(off[t]) for t in both) / len(both),
            "on_rate": sum(sum(on[t]) / len(on[t]) for t in both) / len(both), "mean_diff": mean,
            "ci95": (boots[500], boots[19499]), "p_two_sided": flips / 20000,
            "on_better": sum(d > 0 for d in diffs), "off_better": sum(d < 0 for d in diffs),
            "same": sum(d == 0 for d in diffs)}


def main():
    runs = load_runs()
    unmatched = attach_fixes(runs)
    res = {}
    print("== Runs per side")
    for arm in ("off", "on"):
        rs = [r for r in runs if r["arm"] == arm]
        u = [r for r in rs if r["usable"]]
        res[arm] = {"runs": len(rs), "usable": len(u), "solved": sum(r["solved"] for r in u),
                    "reruns": sum(r["rerun"] for r in rs), "rerun_usable": sum(r["rerun"] and r["usable"] for r in rs),
                    "endings_usable": dict(Counter(r["stop"] for r in u)),
                    "lost_by_task": dict(Counter(r["task"] for r in rs if not r["usable"]))}
        print(f"  {arm}: {len(rs)} runs ({res[arm]['reruns']} reruns), {len(u)} usable, "
              f"{res[arm]['solved']} solved = {res[arm]['solved'] / len(u):.1%} of usable")
        print(f"    usable endings: {res[arm]['endings_usable']}")
    for label, skip in (("all tasks", ()), ("without the 7 heavy tasks", HEAVY)):
        p = paired(runs, skip)
        res["paired_" + ("all" if not skip else "no_heavy")] = p
        print(f"== Paired by task, {label}: {p['tasks']} tasks with usable runs on both sides")
        print(f"  mean per-task solve rate: off {p['off_rate']:.1%}, on {p['on_rate']:.1%}; "
              f"difference {p['mean_diff']:+.1%} (95% CI {p['ci95'][0]:+.1%} to {p['ci95'][1]:+.1%}), "
              f"p = {p['p_two_sided']:.2f}")
        print(f"  tasks: on better {p['on_better']}, off better {p['off_better']}, same {p['same']}")
    # what each fix did
    print("== Fixes (fixes-on runs; firing tied to its run by call time)")
    on_u = [r for r in runs if r["arm"] == "on" and r["usable"]]
    for fix in ("cutoff", "check", "image"):
        hit = [r for r in on_u if r["fixes"][fix]]
        res[f"fix_{fix}"] = {"runs": len(hit), "solved": sum(r["solved"] for r in hit),
                             "firings": sum(r["fixes"][fix] for r in hit)}
        if hit:
            print(f"  {fix}: fired in {len(hit)} usable runs ({res[f'fix_{fix}']['firings']} times), "
                  f"{res[f'fix_{fix}']['solved']} of them solved")
    off_u = [r for r in runs if r["arm"] == "off" and r["usable"]]
    cut_off = [r for r in off_u if r["last_finish"] == "length"]
    img_off = [r for r in off_u if r["stop"] == "image_request"]
    done_off = [r for r in off_u if r["stop"] == "agent_completed" and r["last_finish"] != "length"]
    res["off_said_done"] = {"runs": len(done_off), "solved": sum(r["solved"] for r in done_off)}
    print(f"  fixes off, runs where the model said it was done: {len(done_off)}, "
          f"{res['off_said_done']['solved']} solved (check fires at that moment with fixes on)")
    print(f"  fixes off, for comparison: {len(cut_off)} runs ended on a cut-off reply (what cutoff targets), "
          f"{len(img_off)} on an image request (what image targets)")
    # tasks where the model asked to look at a picture, on either side
    pics = {r["task"] for r in runs if r["stop"] == "image_request" or r["fixes"]["image"]}
    res["picture_tasks"] = {"tasks": len(pics)}
    for arm in ("off", "on"):
        u = [r for r in runs if r["arm"] == arm and r["usable"] and r["task"] in pics]
        res["picture_tasks"][arm] = {"usable": len(u), "solved": sum(r["solved"] for r in u)}
    other = paired(runs, HEAVY | pics)
    res["paired_other"] = other
    pt = res["picture_tasks"]
    print(f"  picture tasks ({len(pics)}): off {pt['off']['solved']}/{pt['off']['usable']}, "
          f"on {pt['on']['solved']}/{pt['on']['usable']}; other {other['tasks']} tasks (no pictures, not heavy): "
          f"off {other['off_rate']:.1%}, on {other['on_rate']:.1%} per task")
    if unmatched:
        print(f"  fix log lines not tied to one run: {dict(unmatched)}")
    res["off_cutoff_endings"], res["off_image_endings"] = len(cut_off), len(img_off)
    res["unmatched_fix_lines"] = {f"{a}/{b}": n for (a, b), n in unmatched.items()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "fixes_test.json").write_text(json.dumps(res, indent=1, default=list))


if __name__ == "__main__":
    main()
