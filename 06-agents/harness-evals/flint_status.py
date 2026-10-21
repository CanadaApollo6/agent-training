"""Live status page for the Flint Terminal-Bench 2 pilot (Flint BF16 on SGLang vs R1s's earlier runs on the same 19
tasks): solved per attempt, a task-by-attempt grid with R1s's per-task record beside it, the server's last decode line
and pod spend so far. Reads logs/ and results/flint-pods.log; writes logs/flint-status.html.

    python3 flint_status.py [pod ip]
"""

import html
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from compare_builds import attempts

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
TASKS = [t for t in (HERE / "tasks.txt").read_text().split() if t != "qemu-startup"]
DONE = re.compile(r"^(\d\d:\d\d:\d\d).*rollout done: id=\w+ task=(\d+) reward=([0-9.]+) turns=(\d+) stop=(\w+)")
STARTED = re.compile(r"rollout start: id=\w+ task=(\d+)")


def grid():
    """{attempt: {task: (solved, turns, stop)}}, {attempt: set of running tasks}"""
    done, running = {}, {}
    for a in (1, 2, 3):
        log = LOGS / f"flint-bf16-a{a}-pi.log"
        rows, started = {}, set()
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                if m := DONE.match(line):
                    rows[TASKS[int(m[2])]] = (float(m[3]) >= 1, int(m[4]), m[5])
                elif m := STARTED.search(line):
                    started.add(TASKS[int(m[1])])
        done[a], running[a] = rows, started - set(rows)
    return done, running


def spend():
    lines = (HERE / "results" / "flint-pods.log").read_text().splitlines()
    stamp = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    ends = {m[2]: stamp(m[1]) for l in lines if (m := re.match(r"(\S+ \S+) terminate (\S+)", l))}
    total = 0.0
    for line in lines:
        if m := re.match(r"(\S+ \S+) create (\S+) .*\$([0-9.]+)/h", line):
            total += (ends.get(m[2], datetime.now()) - stamp(m[1])).total_seconds() / 3600 * float(m[3])
    return total


def server_line(ip):
    if not ip:
        return ""
    try:
        out = subprocess.run(["ssh", "-i", str(Path.home() / ".ssh/prime_ed25519"), "-o", "ConnectTimeout=10",
                              f"ubuntu@{ip}", "sudo docker logs --tail 50 flint 2>&1 | grep 'Decode batch' | tail -1"],
                             capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return ""
    m = re.search(r"#running-req: (\d+).*full token usage: ([0-9.]+).*gen throughput \(token/s\): ([0-9.]+)"
                  r".*#queue-req: (\d+)", out)
    return (f"Server: {m[1]} requests decoding, {m[3]} tok/s total, KV {float(m[2]) * 100:.0f}% full, "
            f"{m[4]} queued." if m else "")


def page(ip=None):
    done, running = grid()
    r1s = attempts("r1s-c4s", "ornith35b")
    per = [f"{sum(r[0] for r in done[a].values())}/{len(done[a])}" for a in (1, 2, 3)]
    r1s_per = [str(sum(run.get(t, {"solved": False})["solved"] for t in TASKS)) for _, run in sorted(r1s.items())]
    cells = []
    for t in TASKS:
        row = [f"<td class=t>{html.escape(t)}</td>"]
        for a in (1, 2, 3):
            r = done[a].get(t)
            if r is None:
                row.append(f"<td class=w>{'…' if t in running[a] else '·'}</td>")
            else:
                row.append(f"<td class={'y' if r[0] else 'n'} title='{r[1]} turns, {r[2]}'>{'✓' if r[0] else '✗'}</td>")
        k = sum(run.get(t, {"solved": False})["solved"] for run in r1s.values())
        row.append(f"<td>{k}/{len(r1s)}</td>")
        cells.append("<tr>" + "".join(row) + "</tr>")
    return f"""<!doctype html><meta charset=utf-8><title>Flint TB2</title>
<style>body{{font:13px var(--font-sans,system-ui);margin:0}}table{{border-collapse:collapse;margin:6px 0 16px}}
td,th{{border:1px solid var(--border,#ddd);padding:2px 7px;text-align:center}}td.t{{text-align:left}}
.y{{background:var(--success,#3a3);color:var(--success-foreground,#fff)}}.n{{background:var(--destructive-surface,#f8dada)}}
.w{{color:var(--muted-foreground,#aaa)}}h2{{font-size:15px;margin:12px 0 4px}}</style>
<h1 style="font-size:17px;margin:0 0 6px">Flint BF16 on Terminal-Bench 2 (pi harness, 19 tasks × 3)</h1>
<p>Updated {datetime.now():%H:%M}. One H200, Recursive's SGLang kit, their sampling (T 1.0, top_p 0.95, top_k 20),
60 turns / 60 min. Pod spend so far: <b>${spend():.2f}</b>. {server_line(ip)}</p>
<h2>Solved / finished, per attempt</h2><table><tr><th>Model</th><th>a1</th><th>a2</th><th>a3</th></tr>
<tr><td>Flint BF16</td>{''.join(f'<td>{p}</td>' for p in per)}</tr>
<tr><td>R1s (Ornith 35B, earlier run)</td>{''.join(f'<td>{p}/19</td>' for p in r1s_per)}</tr></table>
<h2>Task by attempt (… running)</h2><table><tr><th>Task</th><th>a1</th><th>a2</th><th>a3</th><th>R1s</th></tr>
{''.join(cells)}</table>"""


if __name__ == "__main__":
    out = LOGS / "flint-status.html"
    out.write_text(page(sys.argv[1] if len(sys.argv) > 1 else None))
    print(out)
