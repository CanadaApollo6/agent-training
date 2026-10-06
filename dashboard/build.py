#!/usr/bin/env python3
"""Read the research notes and live logs, and write data.json for the dashboard.

Standard library only. Reads files outside this directory and writes only data.json here.
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

DASH = Path(__file__).resolve().parent
REPO = DASH.parent
EVAL = REPO / "06-agents" / "harness-evals"
LEGO = REPO / "04-post-training" / "terminal-lego" / "results"
MEM = Path(
    "/home/riels/.claude/projects/-home-riels-Projects-Personal-Research-agent-training/memory"
)
SPEND = Path("/tmp/prime-spend.log")
OUT = DASH / "data.json"

# Fixes-on server 1 keeps its tasks 0..P1_SPLIT; a helper server (on-p1b) runs the rest of its list.
P1_SPLIT = 25
# Rollouts that never logged "rollout done" because the driver was stopped by hand: {server: {id: (task, clock)}}.
# Counted as lost to errors. off-p2's task 21 hung after its 13:27 agent timeout; stopped at 16:50.
HUNG = {"off-p2": {"f580321e19134711af2b6e275cc97a27": (21, "13:27:08")}}
ERROR_STOPS = {"providererror", "harnesserror", "sandboxerror"}
STOP_PLAIN = {
    "agentcompleted": "finished on its own",
    "agenttimeout": "hit the time limit",
    "maxturns": "hit the turn limit",
    "providererror": "stopped on a server error",
    "harnesserror": "stopped on a harness error",
    "sandboxerror": "stopped on a sandbox error",
    "imageerror": "tried to view an image, which this model can't (the image fix prevents this)",
}
FIX_PLAIN = {
    "cutoff": "If a reply runs out of room while the model is still thinking, try that reply again with thinking turned off.",
    "check": "Before it finishes, the model has to check its own outputs.",
    "image": "A picture in the task is replaced with a short text note, because this model only reads text.",
}
FIX_ORDER = ("cutoff", "check", "image")


def stop_key(stop):
    return re.sub(r"[^a-z]", "", (stop or "").lower())


def stop_plain(stop):
    return STOP_PLAIN.get(stop_key(stop), "stopped for another reason")


def stop_is_error(stop):
    return stop_key(stop) in ERROR_STOPS


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_jsonl(path):
    """Complete JSON lines only. A trailing partial line is skipped."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if not data:
        return []
    if not data.endswith(b"\n"):
        data = data.rsplit(b"\n", 1)[0]
    rows = []
    if not data:
        return rows
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt_clock(dt):
    if dt is None:
        return None
    hour = dt.strftime("%I").lstrip("0") or "12"
    ampm = "a.m." if dt.hour < 12 else "p.m."
    return f"{hour}:{dt.strftime('%M')}\u00a0{ampm}"


def fmt_day(dt):
    if dt is None:
        return None
    return f"{dt.strftime('%d').lstrip('0')} {dt.strftime('%b %Y')}"


def local_now():
    return datetime.now().astimezone()


def search(pattern, text, flags=0):
    if not text:
        return None
    return re.search(pattern, text, flags)


def num(match, group):
    if not match:
        return None
    return int(match.group(group))


def load_bands(path):
    groups = defaultdict(lambda: Counter())
    solved = Counter()
    tries = Counter()
    for row in read_jsonl(path):
        band = row.get("band")
        difficulty = row.get("difficulty") or "unknown"
        if band not in ("always", "sometimes", "never"):
            continue
        groups[difficulty][band] += 1
        solved[difficulty] += int(row.get("solved") or 0)
        tries[difficulty] += int(row.get("solved") or 0) + int(row.get("failed") or 0) + int(
            row.get("infra") or 0
        )
    return groups, solved, tries


def lego_section(md):
    groups_p, solved_p, tries_p = load_bands(LEGO / "solve_rates_pilot.jsonl")
    groups_h, solved_h, tries_h = load_bands(LEGO / "solve_rates_hard.jsonl")
    sets = []
    for key, label, groups, solved, tries in (
        ("medium", "Pilot, medium tasks", groups_p, solved_p, tries_p),
        ("hard", "Pilot, hard tasks", groups_p, solved_p, tries_p),
    ):
        c = groups.get(key)
        if not c:
            continue
        sets.append(
            {
                "label": label,
                "always": c["always"],
                "sometimes": c["sometimes"],
                "never": c["never"],
                "tasks": sum(c.values()),
                "solved": solved[key],
                "tries": tries[key],
            }
        )
    hard = groups_h.get("hard")
    if hard:
        sets.append(
            {
                "label": "Hard round",
                "always": hard["always"],
                "sometimes": hard["sometimes"],
                "never": hard["never"],
                "tasks": sum(hard.values()),
                "solved": solved_h["hard"],
                "tries": tries_h["hard"],
            }
        )
    candidates = sum(s["sometimes"] for s in sets)
    misses = {}
    hour = search(r"(\d+) hit the 1-hour limit", md)
    turns = search(r"(\d+) ran out of 60 turns", md)
    wrong = search(r"(\d+) finished with a wrong answer", md)
    avg = search(r"average (\d+) turns", md)
    pool = search(r"(\d+) of the (\d+) eligible hard tasks", md)
    if hour and turns and wrong:
        misses = {
            "time_limit": num(hour, 1),
            "turn_limit": num(turns, 1),
            "wrong": num(wrong, 1),
        }
    return {
        "sets": sets,
        "candidates": candidates,
        "pilot_sometimes": sum(s["sometimes"] for s in sets if s["label"].startswith("Pilot")),
        "hard_sometimes": next((s["sometimes"] for s in sets if s["label"] == "Hard round"), None),
        "misses": misses or None,
        "average_solve_turns": num(avg, 1),
        "pool_tried": num(pool, 1),
        "pool_eligible": num(pool, 2),
    }


def experiments(harness, root, distill, thesis, series):
    rows = []

    full = search(r"Full TB2 \(89 tasks[^)]*\):\s*pi\s*(\d+),\s*prime_agent\s*(\d+)", harness)
    big = search(
        r"pi rescued \d+, for (\d+)/(\d+)\. prime_agent rescued \d+, for (\d+)/(\d+)",
        harness,
    )
    claude = search(r"R1s solved (\d+)/89", harness)
    if full and big and claude:
        rows.append(
            {
                "when": "1 Oct",
                "title": "Full terminal test, three harnesses",
                "outcome": (
                    f"On all 89 tasks, the model solved {full.group(1)} in the pi harness "
                    f"and {full.group(2)} in prime_agent, the harness in use now. "
                    f"Giving the failures more time raised those to {big.group(1)} and {big.group(3)}. "
                    f"In Claude Code, the harness it was originally trained for, it solved {claude.group(1)} of 89."
                ),
            }
        )

    qwen = search(r"\|\s*Qwen3\.8-Max\s*\|\s*(\d+)\s*\|", harness)
    deep = search(r"\|\s*DeepSeek V4\.1 Flash\s*\|\s*(\d+)\s*\|", harness)
    glm = search(r"\|\s*GLM-5\.3\s*\|\s*(\d+)\s*\|", harness)
    if qwen and deep and glm:
        together = search(r"Together DeepSeek and Qwen solved (\d+) tasks", harness)
        covered = f" Together the first two covered {together.group(1)} tasks." if together else ""
        rows.append(
            {
                "when": "2 Oct",
                "title": "Asked stronger models to solve what this one never has",
                "outcome": (
                    f"On 17 text tasks, Qwen solved {qwen.group(1)}, DeepSeek solved {deep.group(1)}, "
                    f"and GLM solved {glm.group(1)}.{covered} "
                    "GLM, the predicted winner, came last."
                ),
            }
        )

    means = search(
        r"\|\s*solved per attempt\s*\|\s*[^|]*mean\s*([0-9.]+)\)\s*\|\s*[^|]*mean\s*([0-9.]+)\)\s*\|\s*[^|]*mean\s*([0-9.]+)\)",
        distill,
    )
    if means:
        rows.append(
            {
                "when": "5 Oct",
                "title": "Trained on those stronger models' solutions",
                "outcome": (
                    f"No gain. On a 20-task check, the untouched model averaged {means.group(1)} solves per try. "
                    f"Training on the raw solutions averaged {means.group(2)}. "
                    f"Training on solutions rewritten in this model's own words averaged {means.group(3)}."
                ),
            }
        )

    lessons = search(r"(\d+)/(\d+) solved, and only (\d+) with a lesson in context", root)
    if lessons:
        rows.append(
            {
                "when": "5 Oct",
                "title": "Lessons written into the prompt",
                "outcome": (
                    f"On {lessons.group(2)} tasks the model had never solved, six rounds produced "
                    f"{lessons.group(1)} solves. Only {lessons.group(3)} of those had a lesson in the prompt. "
                    "The others were ordinary retries. No real gain."
                ),
            }
        )

    cap = search(
        r"\|\s*Solved \(3 attempts\)\s*\|\s*(\d+)/(\d+)[^|]*\|\s*(\d+)/(\d+)",
        harness,
    )
    cuts = search(
        r"\|\s*Runs ending on a length-cut reply\s*\|\s*(\d+)[^|]*\|\s*(\d+)",
        harness,
    )
    if cap:
        extra = ""
        if cuts:
            extra = (
                f" Replies that ran out of room did not get rarer: {cuts.group(1)} runs at the shorter cap "
                f"and {cuts.group(2)} at the longer one."
            )
        rows.append(
            {
                "when": "5 Oct",
                "title": "Longer replies: 16K vs 32K tokens",
                "outcome": (
                    f"On 20 hard tasks, three tries each: {cap.group(1)} of {cap.group(2)} solved with the shorter cap, "
                    f"{cap.group(3)} of {cap.group(4)} with the longer one. "
                    "A small gain. The longer cap is what later runs use."
                    + extra
                ),
            }
        )

    base = search(r"\|\s*32K baseline\s*\|[^|\n]*\|\s*(\d+)/(\d+)\s*\|\s*(\d+)", harness)
    blind = search(r"\|\s*blind\s*\|[^|\n]*\|\s*(\d+)/(\d+)\s*\|\s*(\d+)", harness)
    look = search(r"\|\s*look\s*\|[^|\n]*\|\s*(\d+)/(\d+)\s*\|\s*(\d+)", harness)
    if base and blind and look:
        rows.append(
            {
                "when": "5 Oct",
                "title": "Plan first, then do the task",
                "outcome": (
                    f"No gain. A plan from the task text scored {blind.group(1)} of {blind.group(2)}. "
                    f"A plan after a look around scored {look.group(1)} of {look.group(2)}. "
                    f"The same runs with no plan scored {base.group(1)} of {base.group(2)}. "
                    f"The plan runs also lost more tries to sandbox and server failures "
                    f"({blind.group(3)} and {look.group(3)}, against {base.group(3)})."
                ),
            }
        )

    smokes = []
    for line in series.splitlines():
        if "bigoff" in line or "bigon" in line:
            continue
        m = re.search(
            r"(fixsmoke2?) prime_agent done \(exit \d+\): (\d+) solved of (\d+)",
            line,
        )
        if m:
            smokes.append((m.group(1), int(m.group(2)), int(m.group(3))))
    if smokes:
        bits = []
        for name, solved, total in smokes:
            label = "the rerun" if name.endswith("2") else "the first check"
            bits.append(f"{label} scored {solved} of {total}")
        bits[0] = bits[0][0].upper() + bits[0][1:]
        first_solve = ""
        if "sqlite-db-truncate first ever solve" in thesis:
            first_solve = " The rerun includes the first solve of sqlite-db-truncate."
        rows.append(
            {
                "when": "5 Oct",
                "title": "Smoke test of the three fixes",
                "outcome": (
                    "Four held-out tasks, one try, fixes on. "
                    + ", and ".join(bits)
                    + ". The first check was firing on the wrong turns, so it was rerun."
                    + first_solve
                ),
            }
        )

    return rows


def parse_series_expected(series):
    expected = {}
    for line in series.splitlines():
        if any(tag in line for tag in ("bigoffa", "bigoffb", "bigona", "bigonb")):
            continue
        m = re.search(
            r"ornith35b-r1s-sd-fixes-(off|on)-p([12]) prime_agent start \([^)]*?(\d+) tasks x (\d+)",
            line,
        )
        if not m:
            continue
        key = f"{m.group(1)}-p{m.group(2)}"
        expected[key] = int(m.group(3)) * int(m.group(4))
    return expected


def fallback_expected():
    out = {}
    for half, name in (("1", "p1"), ("2", "p2")):
        path = EVAL / "logs" / f"../tb2_noqemu_{name}.txt"
        path = EVAL / f"tb2_noqemu_{name}.txt"
        try:
            n = sum(1 for line in path.read_text().splitlines() if line.strip())
        except OSError:
            continue
        out[f"off-p{half}"] = n * 2
        out[f"on-p{half}"] = n * 2
    return out


def clock_on(day, hms, tz):
    hour, minute, second = (int(x) for x in hms.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=tz)


def parse_rollouts(path, day, tz, start):
    runs = []
    text = read_text(path)
    for line in text.splitlines():
        m = re.search(
            r"(\d{2}:\d{2}:\d{2})\s+INFO rollout done: id=(\S+) task=(\d+) reward=([0-9.]+) turns=(\d+) stop=(\S+)",
            line,
        )
        if not m:
            continue
        when = clock_on(day, m.group(1), tz)
        if start and when < start - timedelta(hours=1):
            when += timedelta(days=1)
        reward = float(m.group(4))
        runs.append(
            {
                "id": m.group(2),
                "task": int(m.group(3)),
                "when": when,
                "reward": reward,
                "solved": reward >= 0.999,
                "turns": int(m.group(5)),
                "stop": m.group(6),
            }
        )
    # A rerun of the same id should count once, last write wins.
    by_id = {}
    for run in runs:
        by_id[run["id"]] = run
    return list(by_id.values())


def parse_fixes(paths):
    counts = {name: {"fired": 0, "acted": 0, "acted_known": 0} for name in FIX_ORDER}
    for path in paths:
        for row in read_jsonl(path):
            fix = row.get("fix")
            if fix not in counts:
                counts[fix] = {"fired": 0, "acted": 0, "acted_known": 0}
            counts[fix]["fired"] += 1
            if "acted" in row:
                counts[fix]["acted_known"] += 1
                if row.get("acted") is True:
                    counts[fix]["acted"] += 1
    ordered = []
    for name in FIX_ORDER:
        if name not in counts:
            continue
        item = counts[name]
        ordered.append(
            {
                "id": name,
                "label": {"cutoff": "Cutoff", "check": "Check", "image": "Image"}[name],
                "detail": FIX_PLAIN[name],
                "fired": item["fired"],
                "acted": item["acted"] if item["acted_known"] else None,
                "acted_known": item["acted_known"] > 0,
            }
        )
    return ordered


def task_short(name):
    if not isinstance(name, str) or not name:
        return None
    return name.split("/")[-1]


def reward_of(obj):
    traces = obj.get("traces") or []
    if not traces or not isinstance(traces[0], dict):
        return None, None
    trace = traces[0]
    rewards = trace.get("rewards")
    total = 0.0
    found = False
    if isinstance(rewards, dict):
        values = rewards.values()
    elif isinstance(rewards, list):
        values = rewards
    else:
        values = []
    for value in values:
        if isinstance(value, dict) and "score" in value:
            total += float(value["score"])
            found = True
        elif isinstance(value, (int, float)):
            total += float(value)
            found = True
    stop = trace.get("stop_condition")
    if "image input" in str(obj.get("errors") or trace.get("errors") or ""):
        stop = "image_error"
    if not found:
        return None, stop
    return total, stop


def image_error_ids():
    """Ids of fix-test runs that ended on the server's image-input 400 (the failure the image fix prevents)."""
    ids = set()
    for path in (EVAL / "outputs" / "primeintellect").glob("terminal-bench-2--ornith35b-r1s-sd-fixes-*/**/traces.jsonl"):
        for obj in read_jsonl(path):
            traces = obj.get("traces") or []
            if traces and isinstance(traces[0], dict) and reward_of(obj)[1] == "image_error":
                ids.add(traces[0].get("id"))
    return ids


def parse_traces(reward_by_id, dropped_ids=()):
    root = EVAL / "outputs" / "primeintellect"
    grouped = defaultdict(lambda: {"off": [], "on": []})
    counts = Counter()
    if not root.is_dir():
        return [], counts
    for path in sorted(root.glob("terminal-bench-2--ornith35b-r1s-sd-fixes-*/traces.jsonl")):
        m = re.search(r"fixes-(off|on)-p[12]", path.parent.name)
        if not m:
            continue
        arm = m.group(1)
        seen = set()
        for obj in read_jsonl(path):
            rid = obj.get("id")
            if rid and rid in seen:
                continue
            if rid:
                seen.add(rid)
            try:
                name = task_short(obj["task"]["data"]["name"])
            except (KeyError, TypeError):
                continue
            if not name:
                continue
            score, stop = reward_of(obj)
            traces = obj.get("traces") or []
            trace_id = traces[0].get("id") if traces and isinstance(traces[0], dict) else None
            if trace_id in dropped_ids:
                continue
            if score is None and trace_id in reward_by_id:
                score = reward_by_id[trace_id]["reward"]
                stop = stop or reward_by_id[trace_id]["stop"]
            if score is None:
                continue
            stop = stop or "unknown"
            grouped[name][arm].append(
                {
                    "solved": score >= 0.999,
                    "stop": stop,
                    "plain": stop_plain(stop),
                    "error": stop_is_error(stop),
                }
            )
            counts[arm] += 1
    tasks = []
    for name in sorted(grouped):
        tasks.append({"name": name, "off": grouped[name]["off"], "on": grouped[name]["on"]})
    return tasks, counts


def parse_spend(now):
    text = read_text(SPEND)
    starts = {}
    notes = {}
    dup = set()
    missing_start = 0
    unmatched_times = []
    line_re = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+"
        r"(?P<kind>START|TERMINATE)\s+hfjob\s+(?P<job>\S+)\s*(?P<rest>.*)$"
    )
    multi_re = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+START\s+\d+x\s+hfjob\b.*\((?P<ids>[0-9a-f,]+)\)\s*(?P<rest>.*)$"
    )
    events = []
    for line in text.splitlines():
        multi = multi_re.match(line.strip())
        if multi:
            when = datetime.fromisoformat(multi.group("ts"))
            rest = multi.group("rest") or ""
            for prefix in multi.group("ids").split(","):
                prefix = prefix.strip()
                if not prefix:
                    continue
                events.append(("start", when, prefix, rest, True))
            continue
        m = line_re.match(line.strip())
        if not m:
            continue
        when = datetime.fromisoformat(m.group("ts"))
        kind = m.group("kind")
        job = m.group("job").rstrip(":,")
        rest = m.group("rest") or ""
        if kind == "START":
            events.append(("start", when, job, rest, False))
        else:
            events.append(("stop", when, job, rest, False))

    def remember_start(job, when, rest, prefix):
        key = ("prefix", job) if prefix else ("id", job)
        if key not in starts or when < starts[key]:
            starts[key] = when
            notes[key] = rest

    stops = []
    for kind, when, job, rest, prefix in events:
        if kind == "start":
            remember_start(job, when, rest, prefix)
        else:
            stops.append((when, job, rest))

    intervals = []
    used_starts = set()

    def find_start(job):
        exact = ("id", job)
        if exact in starts:
            return exact
        best = None
        for key, when in starts.items():
            mode, token = key
            if mode == "prefix" and job.startswith(token):
                if best is None or len(token) > len(best[0][1]):
                    best = (key, when)
            if mode == "id" and job == token:
                return key
        return best[0] if best else None

    for when, job, rest in stops:
        key = find_start(job)
        if key is None:
            missing_start += 1
            unmatched_times.append(when)
            continue
        used_starts.add(key)
        blob = f"{notes.get(key, '')} {rest}".lower()
        intervals.append(
            {
                "start": starts[key],
                "end": when,
                "duplicate": ("big test" in blob or "duplicate" in blob or "bigoff" in blob or "bigon" in blob),
                "open": False,
            }
        )
        if intervals[-1]["duplicate"]:
            dup.add(key)
    running = 0
    for key, when in starts.items():
        if key in used_starts:
            continue
        blob = notes.get(key, "").lower()
        intervals.append(
            {
                "start": when,
                "end": now,
                "duplicate": ("big test" in blob or "duplicate" in blob or "bigoff" in blob or "bigon" in blob),
                "open": True,
            }
        )
        running += 1

    def hours(window_start, window_end, duplicates):
        total = 0.0
        for item in intervals:
            if duplicates is True and not item["duplicate"]:
                continue
            if duplicates is False and item["duplicate"]:
                continue
            start = max(item["start"], window_start)
            end = min(item["end"], window_end)
            if end > start:
                total += (end - start).total_seconds() / 3600
        return total

    return intervals, missing_start, running, hours, unmatched_times


def hourly_rate(distill):
    m = search(r"~(\d+)h(\d+)m each,\s*~\$(\d+)\s+for all three", distill)
    if m:
        each = int(m.group(1)) + int(m.group(2)) / 60
        hours = 3 * each
        dollars = int(m.group(3))
        if hours > 0:
            return dollars / hours, (
                f"About ${dollars / hours:.2f} an hour, from earlier notes: "
                f"three of these servers cost about ${dollars} over roughly {hours:.0f} hours."
            )
    m = search(r"~([0-9.]+) GPU-hours, ~\$([0-9.]+)", distill)
    if m:
        hours = float(m.group(1))
        dollars = float(m.group(2))
        if hours > 0:
            return dollars / hours, (
                f"About ${dollars / hours:.2f} an hour, from earlier notes: "
                f"about ${dollars:.0f} for {hours:g} server-hours."
            )
    return None, None


def stop_breakdown(runs):
    labels = [
        ("finished", "Finished on its own", lambda s: stop_key(s) == "agentcompleted"),
        ("time", "Hit the time limit", lambda s: stop_key(s) == "agenttimeout"),
        ("turns", "Hit the turn limit", lambda s: stop_key(s) == "maxturns"),
        ("image", "Tried to view an image (what the image fix prevents)", lambda s: stop_key(s) == "imageerror"),
        ("error", "Stopped on an error", lambda s: stop_is_error(s)),
    ]
    used = set()
    out = []
    for key, label, pred in labels:
        n = sum(1 for run in runs if pred(run["stop"]))
        used.update(run["stop"] for run in runs if pred(run["stop"]))
        out.append({"id": key, "label": label, "count": n})
    other = sum(1 for run in runs if run["stop"] not in used)
    if other:
        out.append({"id": "other", "label": "Other ending", "count": other})
    return out


def arm_view(name, label, servers, now, pace_from):
    runs = [run for server in servers for run in server["runs"]]
    done = len(runs)
    expected = sum(server["expected"] for server in servers)
    solved = sum(1 for run in runs if run["solved"])
    errors = sum(1 for run in runs if stop_is_error(run["stop"]))
    image_stops = sum(1 for run in runs if stop_key(run["stop"]) == "imageerror")
    server_etas = []
    slow_half = None
    unfinished_known = True
    server_rows = []
    for server in servers:
        recent = [run for run in server["runs"] if run["when"] >= pace_from]
        window_hours = max((now - pace_from).total_seconds() / 3600, 1 / 60)
        rate = (len(recent) / window_hours) if recent else 0
        left = max(server["expected"] - len(server["runs"]), 0)
        eta = None
        if left == 0 and server["runs"]:
            eta = max(run["when"] for run in server["runs"])
        elif left > 0 and rate > 0 and len(recent) >= 3:
            eta = now + timedelta(hours=left / rate)
            server_etas.append((eta, server["label"]))
        elif left > 0:
            unfinished_known = False
        server_rows.append(
            {
                "id": server["id"],
                "label": server["label"],
                "done": len(server["runs"]),
                "expected": server["expected"],
                "solved": sum(1 for run in server["runs"] if run["solved"]),
            }
        )
    remaining = max(expected - done, 0)
    if remaining == 0 and runs:
        eta = max(run["when"] for run in runs)
        eta_basis = "finished"
    elif server_etas and unfinished_known:
        eta, slow_half = max(server_etas, key=lambda item: item[0])
        eta_basis = "pace since 9:03 a.m."
    else:
        eta = None
        eta_basis = None
    return {
        "id": name,
        "label": label,
        "done": done,
        "expected": expected,
        "solved": solved,
        "errors": errors,
        "image_stops": image_stops,
        "interim": remaining > 0,
        "servers": server_rows,
        "slow_half": slow_half,
        "endings": stop_breakdown(runs),
        "eta": fmt_clock(eta),
        "eta_basis": eta_basis,
        "eta_dt": eta.isoformat() if eta else None,
    }


def build():
    now = local_now()
    tz = now.tzinfo
    harness = read_text(EVAL / "README.md")
    root = read_text(REPO / "README.md")
    distill = read_text(REPO / "04-post-training" / "self-distill" / "README.md")
    thesis = read_text(MEM / "harness-training-thesis.md")
    budget = read_text(MEM / "prime-compute-budget.md")
    series = read_text(EVAL / "logs" / "hf-series.log")
    lego_md = read_text(LEGO / "solve_rates.md")

    lego = lego_section(lego_md)
    history = experiments(harness, root, distill, thesis, series)

    expected = parse_series_expected(series)
    if len(expected) < 4:
        for key, value in fallback_expected().items():
            expected.setdefault(key, value)

    spend_start = {}
    for line in read_text(SPEND).splitlines():
        m = re.search(
            r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+START hfjob\s+\S+\s+a10g-large fixes (off|on) p([12])\b",
            line.strip(),
        )
        if m:
            spend_start[f"{m.group(2)}-p{m.group(3)}"] = datetime.fromisoformat(m.group(1))
    test_start = min(spend_start.values()) if spend_start else None
    day = (test_start or now).date()

    servers = []
    dropped_ids = set()
    for arm, half, label in (
        ("off", "1", "Server 1"),
        ("off", "2", "Server 2"),
        ("on", "1", "Server 1"),
        ("on", "2", "Server 2"),
    ):
        key = f"{arm}-p{half}"
        path = EVAL / "logs" / f"tb2full-ornith35b-r1s-sd-fixes-{arm}-p{half}-prime_agent.log"
        start = spend_start.get(key)
        runs = parse_rollouts(path, day, tz, start)
        for run_id, (task, clock) in HUNG.get(key, {}).items():
            runs.append({"id": run_id, "task": task, "when": clock_on(day, clock, tz), "reward": 0.0,
                         "solved": False, "turns": 0, "stop": "HarnessError"})
        if key == "on-p1":
            dropped_ids.update(run["id"] for run in runs if run["task"] > P1_SPLIT)
            runs = [run for run in runs if run["task"] <= P1_SPLIT]
            runs += parse_rollouts(EVAL / "logs" / "tb2full-ornith35b-r1s-sd-fixes-on-p1b-prime_agent.log", day, tz, start)
        servers.append(
            {
                "id": key,
                "arm": arm,
                "label": label,
                "expected": expected.get(key, 0),
                "runs": runs,
            }
        )

    image_ids = image_error_ids()
    for server in servers:
        for run in server["runs"]:
            if run.get("id") in image_ids:
                run["stop"] = "image_error"
    pace_from = datetime(day.year, day.month, day.day, 9, 3, tzinfo=tz)
    off = arm_view("off", "Fixes off", [s for s in servers if s["arm"] == "off"], now, pace_from)
    on = arm_view("on", "Fixes on", [s for s in servers if s["arm"] == "on"], now, pace_from)
    reward_by_id = {run["id"]: run for server in servers for run in server["runs"]}
    tasks, trace_counts = parse_traces(reward_by_id, dropped_ids)
    fixes = parse_fixes(
        [
            EVAL / "logs" / "fixes-fixes-on-p1.jsonl",
            EVAL / "logs" / "fixes-fixes-on-p2.jsonl",
            EVAL / "logs" / "fixes-fixes-on-p1b.jsonl",
        ]
    )

    eta_candidates = []
    for arm in (off, on):
        if arm["eta_dt"] and arm["interim"]:
            eta_candidates.append((datetime.fromisoformat(arm["eta_dt"]), arm["label"], arm.get("slow_half")))
    overall_eta = max(eta_candidates)[0] if eta_candidates else None
    limiter = None
    if eta_candidates:
        _, arm_label, half = max(eta_candidates)
        limiter = f"{arm_label}, {half}".lower() if half else arm_label.lower()
    total_done = off["done"] + on["done"]
    total_expected = off["expected"] + on["expected"]
    all_in = total_expected > 0 and total_done >= total_expected and not off["interim"] and not on["interim"]

    if all_in:
        stand = (
            "The fix test has finished. Cheap changes to the prompt, tried earlier, did not make the model "
            "reliably better. The hard-task round found tasks the model solves only sometimes, which are the "
            "ones worth training on."
        )
    else:
        stand = (
            "Cheap changes to the prompt are finished, and none of them made the model reliably better. "
            "Overnight, a hard-task round found tasks the model solves only sometimes. Those are the ones "
            "worth training on. A four-server test is running now: the same tasks with three small fixes off, "
            "and with them on."
        )
    if off["done"] and on["done"] and (off["interim"] or on["interim"]):
        stand += " It is too early to say whether the fixes help."

    rate, rate_note = hourly_rate(distill)
    intervals_unused, missing_start, running, hours, unmatched_times = parse_spend(now)
    today_start = datetime(now.year, now.month, now.day, 7, 0, tzinfo=tz)
    if now < today_start:
        today_start -= timedelta(days=1)
    last_night_end = today_start
    last_night_start = last_night_end - timedelta(hours=13)  # 6:00 p.m. the day before
    # 7:00 a.m. minus 13 hours is 6:00 p.m.
    night_hours = hours(last_night_start, last_night_end, None)
    today_hours = hours(today_start, now, None)
    today_dup_hours = hours(today_start, now, True)
    omitted_last_night = sum(1 for when in unmatched_times if last_night_start <= when < last_night_end)
    nightly = search(r"\$100", budget)
    rl = search(r"\$300-600", thesis)

    def money(server_hours):
        if rate is None:
            return None
        return int(round(server_hours * rate))

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "generated_label": f"{fmt_clock(now)} on {fmt_day(now)}",
        "model": {
            "name": "Ornith R1s-SD",
            "blurb": (
                "A 35-billion-parameter model, quantized and sped up. It is tested as a coding agent "
                "on Terminal-Bench 2: terminal tasks graded by hidden tests."
            ),
        },
        "headline": {
            "paragraph": stand,
            "numbers": [
                {
                    "value": f"{total_done} / {total_expected}" if total_expected else str(total_done),
                    "label": "Live test runs finished",
                    "note": "Interim" if not all_in else "Complete",
                },
                {
                    "value": f"{off['solved']} / {off['done']}" if off["done"] else "—",
                    "label": "Solved, fixes off",
                    "note": "Interim" if off["interim"] else "Complete",
                },
                {
                    "value": f"{on['solved']} / {on['done']}" if on["done"] else "—",
                    "label": "Solved, fixes on",
                    "note": "Interim" if on["interim"] else "Complete",
                },
                {
                    "value": str(lego["candidates"]) if lego["candidates"] else "—",
                    "label": "Tasks worth training on",
                    "note": "Solved only sometimes",
                },
            ],
        },
        "live": {
            "started": fmt_clock(test_start),
            "started_full": test_start.isoformat(timespec="seconds") if test_start else None,
            "tasks": 87 if total_expected == 348 else None,
            "tries": 2,
            "total_done": total_done,
            "total_expected": total_expected,
            "interim": not all_in,
            "eta": fmt_clock(overall_eta) if not all_in else None,
            "eta_note": (
                None
                if all_in or overall_eta is None
                else (
                    "Rough. From the pace since 9:03 a.m., after four extra servers were stopped. "
                    + (f"The slow part is {limiter}, and it sets the time." if limiter else "The slower half sets the time.")
                )
            ),
            "noise": (
                "These solve rates will move. A gap of a few tasks is not a result until most runs are in."
                if not all_in
                else "Every scheduled run has a result in the logs."
            ),
            "arms": [off, on],
            "fixes": fixes,
            "trace_runs": {"off": trace_counts["off"], "on": trace_counts["on"]},
            "tasks": tasks,
        },
        "history": history,
        "lego": lego,
        "next": {
            "thirty_five_b": rl.group(0) if rl else None,
            "nightly": "$100" if nightly else None,
            "nine_b_price": None,
        },
        "spend": {
            "rate": round(rate, 2) if rate else None,
            "rate_note": rate_note,
            "running_servers": running,
            "missing_start": missing_start,
            "omitted_last_night": omitted_last_night,
            "last_night": {
                "label": f"{fmt_clock(last_night_start)} yesterday to {fmt_clock(last_night_end)} today",
                "hours": round(night_hours, 1),
                "dollars": money(night_hours),
            },
            "today": {
                "label": f"since {fmt_clock(today_start)} today",
                "hours": round(today_hours, 1),
                "dollars": money(today_hours),
                "duplicate_hours": round(today_dup_hours, 1),
                "duplicate_dollars": money(today_dup_hours),
            },
        },
    }
    # eta_dt is useful internally but noisy on the page; drop it from the public arms.
    for arm in payload["live"]["arms"]:
        arm.pop("eta_dt", None)
        arm.pop("slow_half", None)

    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)
    return payload


if __name__ == "__main__":
    data = build()
    live = data["live"]
    print(
        f"wrote {OUT.name}: {live['total_done']}/{live['total_expected']} runs, "
        f"off {live['arms'][0]['solved']}/{live['arms'][0]['done']}, "
        f"on {live['arms'][1]['solved']}/{live['arms'][1]['done']}, "
        f"eta {live['eta']}, tasks {len(live['tasks'])}, "
        f"candidates {data['lego']['candidates']}"
    )
