"""Live status page for a Terminal-Bench 2 pilot run on pods (19 tasks x 3 attempts under pi): solved per attempt, a
task-by-attempt grid beside R1s's and Flint's records on the same tasks, each pod's last vLLM stats line and pod spend
so far. Reads logs/<prefix>-a<N>-pi.log and results/<pods log>; writes logs/<prefix>-status.html. Generalises
flint_status.py (SGLang, one pod) to vLLM with one pod per attempt; the pods log's "ip <name> <ip>" lines map pods.

    python3 pilot_status.py <label prefix> <pods log> "<title>" "<setup note>"
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
STAMP = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def grid(prefix):
    """{attempt: {task: (solved, turns, stop)}}, {attempt: set of running tasks}"""
    done, running = {}, {}
    for a in (1, 2, 3):
        log = LOGS / f"{prefix}-a{a}-pi.log"
        rows, started = {}, set()
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                if m := DONE.match(line):
                    rows[TASKS[int(m[2])]] = (float(m[3]) >= 1, int(m[4]), m[5])
                elif m := STARTED.search(line):
                    started.add(TASKS[int(m[1])])
        done[a], running[a] = rows, started - set(rows)
    return done, running


def pods(lines):
    """[(name, ip)] of pods not yet terminated, and the spend across all pods in the log"""
    ends = {m[2]: STAMP(m[1]) for l in lines if (m := re.match(r"(\S+ \S+) terminate (\S+)", l))}
    total, ips = 0.0, {}
    for line in lines:
        if m := re.match(r"(\S+ \S+) create (\S+) .*\$([0-9.]+)/h", line):
            total += (ends.get(m[2], datetime.now()) - STAMP(m[1])).total_seconds() / 3600 * float(m[3])
        elif (m := re.match(r"\S+ \S+ ip (\S+) (\S+)", line)) and m[1] not in ends:
            ips[m[1]] = m[2]
    return sorted(ips.items()), total


def server_line(name, ip):
    try:
        out = subprocess.run(["ssh", "-i", str(Path.home() / ".ssh/prime_ed25519"), "-o", "ConnectTimeout=10",
                              f"ubuntu@{ip}", "grep 'generation throughput' /tmp/vllm.log | tail -1"],
                             capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return ""
    m = re.search(r"generation throughput: ([0-9.]+).*Running: (\d+) reqs, Waiting: (\d+) reqs.*KV cache usage: "
                  r"([0-9.]+)%", out)
    return (f"<li>{name}: {m[2]} decoding, {m[3]} waiting, {float(m[1]):.0f} tok/s, KV {float(m[4]):.0f}% full</li>"
            if m else f"<li>{name}: no stats yet</li>")


def page(prefix, pods_log, title, note):
    done, running = grid(prefix)
    live, spend = pods((HERE / "results" / pods_log).read_text().splitlines())
    refs = {"R1s": attempts("r1s-c4s", "ornith35b"), "Flint": attempts("flint:bf16")}
    per = [f"{sum(r[0] for r in done[a].values())}/{len(done[a])}" for a in (1, 2, 3)]
    ref_rows = "".join(
        f"<tr><td>{n}</td>" + "".join(f"<td>{sum(run.get(t, {'solved': False})['solved'] for t in TASKS)}/19</td>"
                                      for _, run in sorted(r.items())) + "</tr>" for n, r in refs.items())
    cells = []
    for t in TASKS:
        row = [f"<td class=t>{html.escape(t)}</td>"]
        for a in (1, 2, 3):
            r = done[a].get(t)
            if r is None:
                row.append(f"<td class=w>{'…' if t in running[a] else '·'}</td>")
            else:
                row.append(f"<td class={'y' if r[0] else 'n'} title='{r[1]} turns, {r[2]}'>{'✓' if r[0] else '✗'}</td>")
        for r in refs.values():
            row.append(f"<td>{sum(run.get(t, {'solved': False})['solved'] for run in r.values())}/{len(r)}</td>")
        cells.append("<tr>" + "".join(row) + "</tr>")
    servers = "".join(server_line(n, ip) for n, ip in live)
    return f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font:13px var(--font-sans,system-ui);margin:0}}table{{border-collapse:collapse;margin:6px 0 16px}}
td,th{{border:1px solid var(--border,#ddd);padding:2px 7px;text-align:center}}td.t{{text-align:left}}
.y{{background:var(--success,#3a3);color:var(--success-foreground,#fff)}}.n{{background:var(--destructive-surface,#f8dada)}}
.w{{color:var(--muted-foreground,#aaa)}}h2{{font-size:15px;margin:12px 0 4px}}ul{{margin:4px 0}}</style>
<h1 style="font-size:17px;margin:0 0 6px">{html.escape(title)}</h1>
<p>Updated {datetime.now():%H:%M}. {html.escape(note)} Pod spend so far: <b>${spend:.2f}</b>.</p>
{f'<ul>{servers}</ul>' if servers else ''}
<h2>Solved / finished, per attempt</h2><table><tr><th>Model</th><th>a1</th><th>a2</th><th>a3</th></tr>
<tr><td>{html.escape(prefix)}</td>{''.join(f'<td>{p}</td>' for p in per)}</tr>{ref_rows}</table>
<h2>Task by attempt (… running)</h2><table><tr><th>Task</th><th>a1</th><th>a2</th><th>a3</th>
{''.join(f'<th>{n}</th>' for n in refs)}</tr>
{''.join(cells)}</table>"""


if __name__ == "__main__":
    prefix, pods_log, title, note = sys.argv[1:5]
    out = LOGS / f"{prefix}-status.html"
    out.write_text(page(prefix, pods_log, title, note))
    print(out)
