"""Live status page for the Nanbeige draft head build on one pod: data generation per shard (responses, tokens,
each server's vLLM stats), the trainer's last log line once training runs, and pod spend from pods.log.
Writes status.html next to this script.

    python3 drafter_status.py <[user@]pod ip> [total prompts]
"""
import html
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAMP = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
REMOTE = r"""python3 -I - <<'EOF'
import json, glob, os
out = {"shards": {}, "servers": {}, "train": ""}
for f in sorted(glob.glob(os.path.expanduser("~/data/gen-s*.jsonl"))):
    n = toks = cut = 0
    for l in open(f):
        try: r = json.loads(l)
        except ValueError: continue
        n += 1; toks += r["usage"]["completion_tokens"]; cut += r["finish"] == "length"
    out["shards"][f[-7]] = [n, toks, cut]
for f in sorted(glob.glob("/tmp/vllm-*.log")):
    lines = [l for l in open(f, errors="replace") if "generation throughput" in l]
    out["servers"][f.split("-")[-1][:-4]] = lines[-1] if lines else ""
for f in sorted(glob.glob("/tmp/train*.log")):
    lines = [l.strip() for l in open(f, errors="replace") if l.strip()]
    out["train"] = lines[-1][-300:] if lines else ""
print(json.dumps(out))
EOF"""


def spend():
    lines = (HERE / "pods.log").read_text().splitlines()
    ends = {m[2]: STAMP(m[1]) for l in lines if (m := re.match(r"(\S+ \S+) terminate (\S+)", l))}
    return sum((ends.get(m[2], datetime.now()) - STAMP(m[1])).total_seconds() / 3600 * float(m[3])
               for l in lines if (m := re.match(r"(\S+ \S+) create (\S+) .*\$([0-9.]+)/h", l)))


def page(ip, total):
    out = subprocess.run(["ssh", "-i", str(Path.home() / ".ssh/prime_ed25519"), "-o", "ConnectTimeout=10",
                          ip if "@" in ip else f"root@{ip}", REMOTE], capture_output=True, text=True, timeout=60).stdout
    s = json.loads(out)
    n = sum(v[0] for v in s["shards"].values())
    toks = sum(v[1] for v in s["shards"].values())
    rate = 0.0
    rows = []
    for g, line in s["servers"].items():
        m = re.search(r"generation throughput: ([0-9.]+).*Running: (\d+) reqs, Waiting: (\d+) reqs.*KV cache usage: "
                      r"([0-9.]+)%", line)
        if m:
            rate += float(m[1])
            rows.append(f"<tr><td>GPU {g}</td><td>{m[2]}</td><td>{m[3]}</td><td>{float(m[1]):,.0f}</td>"
                        f"<td>{float(m[4]):.0f}%</td></tr>")
    shard_rows = "".join(f"<tr><td>shard {k}</td><td>{v[0]:,}</td><td>{v[1] / 1e6:.1f}M</td><td>{v[2]:,}</td></tr>"
                         for k, v in s["shards"].items())
    mean = toks / n if n else 2200
    eta = (total - n) * mean / rate / 3600 if rate else float("nan")
    pct = 100 * n / total
    return f"""<!doctype html><meta charset=utf-8><title>Nanbeige drafter build</title>
<style>body{{font:13px var(--font-sans,system-ui);margin:0}}table{{border-collapse:collapse;margin:6px 0 14px}}
td,th{{border:1px solid var(--border,#ddd);padding:2px 8px;text-align:right}}td:first-child{{text-align:left}}
.bar{{height:14px;background:var(--muted,#eee);border-radius:3px;width:420px}}
.fill{{height:14px;background:var(--primary,#36c);border-radius:3px}}h2{{font-size:15px;margin:10px 0 4px}}</style>
<h1 style="font-size:17px;margin:0 0 6px">Nanbeige4.3-33B draft head: data generation</h1>
<p>Updated {datetime.now():%H:%M}. One on-demand H200, FP8 weights and KV.
Pod spend so far: <b>${spend():.2f}</b>.</p>
<div class=bar><div class=fill style="width:{pct:.1f}%"></div></div>
<p><b>{n:,} / {total:,}</b> responses ({pct:.0f}%), {toks / 1e6:.1f}M tokens, mean {mean:,.0f}.
Now {rate:,.0f} tok/s across servers; about {eta:.1f} h to go.</p>
<h2>Servers</h2><table><tr><th></th><th>decoding</th><th>waiting</th><th>tok/s</th><th>KV full</th></tr>
{''.join(rows)}</table>
<h2>Shards</h2><table><tr><th></th><th>responses</th><th>tokens</th><th>cut at cap</th></tr>{shard_rows}</table>
{f'<h2>Trainer</h2><pre>{html.escape(s["train"])}</pre>' if s["train"] else ''}"""


if __name__ == "__main__":
    out = HERE / "status.html"
    out.write_text(page(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 36000))
    print(out)
