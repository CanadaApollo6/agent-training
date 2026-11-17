"""Live status page for the harness-hub pi adapter test (run_q27pi.sh): three arms x 3 attempts on the 20-task pilot
under pi. Solved per attempt and per arm (also without qemu-startup, to match the 19-task pilot records), a
task-by-arm grid, each server's last vLLM stats line, and pod spend. Reads logs/tb2-<prefix>-<arm>-a<N>-pi.log and
results/q27pi-pods.log; writes logs/<prefix>-status.html.

    python3 q27pi_status.py <label prefix> [pod ip]
"""
import html
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = (HERE / "tasks.txt").read_text().split()
ARMS = [("med", "Base, thinking medium", 8000), ("off", "Base, thinking off", 8001), ("ad", "pi adapter, thinking off", 8002)]
DONE = re.compile(r"rollout done: id=\w+ task=(\d+) reward=([0-9.]+) turns=(\d+) stop=(\w+)")
START = re.compile(r"rollout start: id=\w+ task=(\d+)")


def attempt(prefix, arm, a):
    log = HERE / "logs" / f"tb2-{prefix}-{arm}-a{a}-pi.log"
    rows, started = {}, set()
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if m := DONE.search(line):
                rows[TASKS[int(m[1])]] = (float(m[2]) >= 1, int(m[3]), m[4])
            elif m := START.search(line):
                started.add(TASKS[int(m[1])])
    return rows, started - set(rows)


def spend():
    lines = (HERE / "results" / "q27pi-pods.log").read_text().splitlines()
    stamp = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    end = next((stamp(l[:19]) for l in lines if " terminate " in l), datetime.now())
    return sum((end - stamp(l[:19])).total_seconds() / 3600 * float(m[1])
               for l in lines if " create " in l and (m := re.search(r"\$([0-9.]+)/h", l)))


def servers(ip):
    if not ip:
        return ""
    cmd = "for p in 8000 8001 8002; do echo $p $(grep 'generation throughput' /tmp/vllm-$p.log | tail -1); done"
    try:
        out = subprocess.run(["ssh", "-i", str(Path.home() / ".ssh/prime_ed25519"), "-o", "ConnectTimeout=10",
                              f"root@{ip}", cmd], capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return ""
    items = []
    for line in out.splitlines():
        port = int(line.split()[0])
        name = next(n for _, n, p in ARMS if p == port)
        m = re.search(r"generation throughput: ([0-9.]+).*Running: (\d+) reqs, Waiting: (\d+) reqs.*KV cache usage: "
                      r"([0-9.]+)%", line)
        items.append(f"<li>{name}: {m[2]} decoding, {m[3]} waiting, {float(m[1]):.0f} tok/s, KV {float(m[4]):.0f}%</li>"
                     if m else f"<li>{name}: no stats yet</li>")
    return "<ul>" + "".join(items) + "</ul>"


def page(prefix, ip):
    data = {arm: [attempt(prefix, arm, a) for a in (1, 2, 3)] for arm, _, _ in ARMS}
    summary = []
    for arm, name, _ in ARMS:
        per = [f"{sum(r[0] for r in rows.values())}/{len(rows)}" for rows, _ in data[arm]]
        n19 = [sum(r[0] for t, r in rows.items() if t != "qemu-startup") for rows, _ in data[arm]]
        summary.append(f"<tr><td>{name}</td>" + "".join(f"<td>{p}</td>" for p in per)
                       + f"<td>{sum(n19) / 3:.1f}</td></tr>")
    head = "".join(f"<th colspan=3>{n}</th>" for _, n, _ in ARMS)
    cells = []
    for t in TASKS:
        row = [f"<td class=t>{html.escape(t)}</td>"]
        for arm, _, _ in ARMS:
            for rows, running in data[arm]:
                r = rows.get(t)
                if r:
                    row.append(f"<td class={'y' if r[0] else 'n'} title='{r[1]} turns, {r[2]}'>"
                               f"{'✓' if r[0] else '✗'}</td>")
                else:
                    row.append(f"<td class=r>{'…' if t in running else ''}</td>")
        cells.append("<tr>" + "".join(row) + "</tr>")
    return f"""<!doctype html><meta charset=utf-8><title>Adapter Test Status</title>
<style>
body{{font:14px/1.45 var(--font-sans,system-ui,sans-serif);margin:0}}
table{{border-collapse:collapse;margin:10px 0}}
td,th{{border-bottom:1px solid var(--border,#8884);padding:3px 7px;text-align:center}}
td.t{{text-align:left;font-family:var(--font-mono,ui-monospace,monospace);font-size:12px}}
.y{{color:var(--success,#1f8a4c)}}.n{{color:var(--destructive,#c0392b)}}
.r,p,li{{color:var(--muted-foreground,#888)}}h2{{margin:0 0 6px}}</style>
<h2>Qwen3.8-27B pi adapter test</h2>
<p>20-task TB2 pilot under pi, 3 attempts per arm, 60 turns, 60 min, 32K reply cap. vLLM BF16 on one 8x A100 pod
(TP 2 per arm). Updated {datetime.now():%H:%M}. Pod spend so far ${spend():.2f}.</p>
<table><tr><th>Arm</th><th>a1</th><th>a2</th><th>a3</th><th>Mean of 19</th></tr>{''.join(summary)}</table>
<p>Mean of 19 leaves out qemu-startup, as the R1s (10.0) and Nanbeige (11.0) pilot records do.</p>
{servers(ip)}
<table><tr><th></th>{head}</tr>{''.join(cells)}</table>"""


if __name__ == "__main__":
    prefix, ip = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    out = HERE / "logs" / f"{prefix}-status.html"
    out.write_text(page(prefix, ip))
    print(out)
