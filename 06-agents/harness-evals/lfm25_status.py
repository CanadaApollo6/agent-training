"""Live status page for the LFM2.5 Terminal-Bench 2 run (M3TiS #3 vs Liquid's Q8_0): tasks done and solved per build
and attempt, a task-by-attempt grid, lane states and pod spend so far. Reads logs/; writes logs/lfm25-status.html.
qemu-startup is shown but not scored (it fails for every build). Usage: python3 lfm25_status.py"""

import html
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
TASKS = (HERE / "tasks.txt").read_text().split()
DROP = {"qemu-startup"}
BUILDS = {"m3tis": "M3TiS #3 (A10s, TensorFold)", "q8": "Liquid Q8_0 (A100, llama.cpp)"}
DONE = re.compile(r"^(\d\d:\d\d:\d\d).*rollout done: id=\w+ task=(\d+) reward=([0-9.]+) turns=(\d+) stop=(\w+)")


def lanes():
    """{(build, attempt): {task name: (solved, turns, stop)}}, plus per-lane state."""
    grid, state = {}, []
    for build in BUILDS:
        for a in (1, 2, 3):
            for half, first in ((1, 0), (2, 10)):
                label = f"lfm25-{build}-a{a}l{half}"
                log = LOGS / f"{label}-pi.log"
                rows = {}
                if log.exists():
                    for line in log.read_text(errors="replace").splitlines():
                        m = DONE.match(line)
                        if m:
                            rows[TASKS[first + int(m[2])]] = (float(m[3]) >= 1, int(m[4]), m[5])
                grid.setdefault((build, a), {}).update(rows)
                q = (LOGS / f"lfm25-{build}-queue.log")
                qtext = q.read_text() if q.exists() else ""
                starts, ends = qtext.count(f"start {label} "), qtext.count(f"end {label} ")  # a rerun starts again
                st = "queued" if not starts else "running" if starts > ends else "done"
                state.append((label, st, len(rows)))
    return grid, state


def spend():
    """Pod-hours and dollars from logs/lfm25-pods.log create lines (pods still up are counted to now)."""
    lines = (LOGS / "lfm25-pods.log").read_text().splitlines()
    stamp = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    ends = {m[2]: stamp(m[1]) for l in lines if (m := re.match(r"(\S+ \S+) terminate (\S+)", l))}
    total = 0.0
    for line in lines:
        m = re.match(r"(\S+ \S+) create (\S+) \w+ .*\$([0-9.]+)/h", line)
        if m:
            total += (ends.get(m[2], datetime.now()) - stamp(m[1])).total_seconds() / 3600 * float(m[3])
    return total


def page():
    grid, state = lanes()
    scored = [t for t in TASKS if t not in DROP]
    head = []
    for build, name in BUILDS.items():
        per = []
        for a in (1, 2, 3):
            rows = grid.get((build, a), {})
            done = sum(t in rows for t in scored)
            per.append(f"{sum(rows[t][0] for t in scored if t in rows)}/{done}")
        head.append(f"<tr><td>{html.escape(name)}</td>" + "".join(f"<td>{p}</td>" for p in per) + "</tr>")
    cells = []
    for t in TASKS:
        row = [f"<td class=t>{html.escape(t)}{' (not scored)' if t in DROP else ''}</td>"]
        for build in BUILDS:
            for a in (1, 2, 3):
                r = grid.get((build, a), {}).get(t)
                if r is None:
                    row.append("<td class=w>·</td>")
                else:
                    row.append(f"<td class={'y' if r[0] else 'n'} title='{r[1]} turns, {r[2]}'>"
                               f"{'✓' if r[0] else '✗'}</td>")
        cells.append("<tr>" + "".join(row) + "</tr>")
    lane_rows = "".join(f"<tr><td>{l}</td><td>{s}</td><td>{n}/10</td></tr>" for l, s, n in state)
    cols = "".join(f"<th>{b} a{a}</th>" for b in BUILDS for a in (1, 2, 3))
    return f"""<!doctype html><meta charset=utf-8><title>LFM2.5 TB2</title>
<style>body{{font:13px var(--font-sans,system-ui);margin:0}}table{{border-collapse:collapse;margin:6px 0 16px}}
td,th{{border:1px solid var(--border,#ddd);padding:2px 7px;text-align:center}}td.t{{text-align:left}}
.y{{background:var(--success,#3a3);color:var(--success-foreground,#fff)}}.n{{background:var(--destructive-surface,#f8dada)}}
.w{{color:var(--muted-foreground,#aaa)}}h2{{font-size:15px;margin:12px 0 4px}}</style>
<h1 style="font-size:17px;margin:0 0 6px">LFM2.5-24B-A2B on Terminal-Bench 2 (pi harness) — live</h1>
<p>Updated {datetime.now():%H:%M}. Both builds at Liquid's settings (T 0.1, top_k 50, repetition penalty 1.05).
19 scored tasks × 3 attempts each. Pod spend so far: <b>${spend():.2f}</b>.</p>
<h2>Solved / finished, per attempt</h2><table><tr><th>Build</th><th>a1</th><th>a2</th><th>a3</th></tr>{''.join(head)}</table>
<h2>Task by attempt</h2><table><tr><th>Task</th>{cols}</tr>{''.join(cells)}</table>
<h2>Lanes (10 tasks each)</h2><table><tr><th>Lane</th><th>State</th><th>Finished</th></tr>{lane_rows}</table>"""


if __name__ == "__main__":
    out = LOGS / "lfm25-status.html"
    out.write_text(page())
    print(out)
