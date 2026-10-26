"""Status page for a draft-head training run, from the log train_watch.sh syncs home: progress and ETA, loss,
next-token accuracy and accepted length over steps, checkpoints copied home, and pod spend. Writes train-status.html.

    python3 train_status.py [run name, default full] [total steps] [$/h] [start, e.g. "2026-10-26 14:09"]
"""
import ast
import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
run = sys.argv[1] if len(sys.argv) > 1 else "full"
total = int(sys.argv[2]) if len(sys.argv) > 2 else 2278
rate = float(sys.argv[3]) if len(sys.argv) > 3 else 5.40
start = datetime.strptime(sys.argv[4] if len(sys.argv) > 4 else "2026-10-26 09:31", "%Y-%m-%d %H:%M")  # pod create, local
d = HERE / "data/live" / f"train-{run}"
log = (d / f"train-{run}-1gpu.log").read_text(errors="replace")

steps = []
for m in re.finditer(r"^step (\d+): (\{.*\})$", log, re.M):
    v = ast.literal_eval(m.group(2))
    steps.append((int(m.group(1)), v["train/ce_loss"], v["train/acc"],
                  v.get("train/dflash/hard_label/walk_accepted_length", 1.0), v["perf/global_samples_per_second"]))
ckpts = sorted(p.name for p in (d / "ckpt").glob("*.pt"))
done = "TRAIN_EXIT" in log
last = steps[-1] if steps else (0, 0, 0, 1, 0)
sps = sum(s[4] for s in steps[-10:]) / max(1, len(steps[-10:]))
eta_h = (total - last[0]) * 32 / sps / 3600 if sps else 0
spent = (datetime.now() - start).total_seconds() / 3600 * rate


def chart(idx, label, color, fmt):
    if len(steps) < 2:
        return ""
    W, H, P = 640, 180, 36
    xs = [s[0] for s in steps]
    ys = [s[idx] for s in steps]
    lo, hi = min(ys), max(ys)
    hi = hi if hi > lo else lo + 1
    pt = lambda x, y: (P + (W - 2 * P) * x / total, H - P / 2 - (H - P) * (y - lo) / (hi - lo))
    path = " ".join(f"{'M' if i == 0 else 'L'}{pt(x, y)[0]:.1f},{pt(x, y)[1]:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    return f"""<figure><figcaption>{label} — now <b>{fmt.format(ys[-1])}</b></figcaption>
<svg viewBox="0 0 {W} {H}" width="100%"><line x1="{P}" y1="{H - P / 2}" x2="{W - P}" y2="{H - P / 2}" class="ax"/>
<text x="{P - 4}" y="{H - P / 2}" class="t" text-anchor="end">{fmt.format(lo)}</text>
<text x="{P - 4}" y="{P / 2 + 4}" class="t" text-anchor="end">{fmt.format(hi)}</text>
<text x="{W - P}" y="{H - 4}" class="t" text-anchor="end">step {total}</text>
<path d="{path}" style="fill:none;stroke:{color};stroke-width:2"/></svg></figure>"""


state = "finished" if done else "training"
page = f"""<!doctype html><meta charset="utf-8"><title>Nanbeige draft head training</title>
<style>body{{font:14px var(--font-sans,system-ui);margin:0;color:var(--foreground,#1d1d1f)}}
h1{{font-size:15px;font-weight:600;margin:0 0 2px}}.sub{{color:var(--muted-foreground,#666);font-size:13px}}
.k{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}.k div{{border:1px solid var(--border,#ddd);border-radius:var(--radius,8px);padding:8px 12px;font-size:12px;color:var(--muted-foreground,#666)}}
.k b{{display:block;font-size:18px;color:var(--foreground,#1d1d1f)}}.bar{{height:6px;background:var(--muted,#eee);border-radius:3px}}
.bar i{{display:block;height:6px;background:var(--chart-1,#3a7bd5);border-radius:3px;width:{100 * last[0] / total:.1f}%}}
figure{{margin:12px 0 0}}figcaption{{font-size:13px;color:var(--muted-foreground,#555)}}figcaption b{{color:var(--foreground,#111)}}
.ax{{stroke:var(--border,#ccc)}}.t{{font-size:11px;fill:var(--muted-foreground,#888)}}svg{{height:180px}}</style>
<h1>Nanbeige4.3-33B DFlash2 draft head — {state}</h1>
<div class="sub">One H200 (capture server + trainer share it). Updated {datetime.now():%H:%M} local.</div>
<div class="k"><div>Step<b>{last[0]} / {total}</b></div><div>Speed<b>{sps:.2f} samples/s</b></div>
<div>Left<b>{eta_h:.1f} h</b></div><div>Pod spend<b>${spent:.0f}</b></div><div>Checkpoints home<b>{len(ckpts)}</b></div></div>
<div class="bar"><i></i></div>
{chart(1, "Loss (cross-entropy)", "var(--chart-2,#d5573a)", "{:.2f}")}
{chart(3, "Accepted length on training blocks (tokens per verify; 1.0 = nothing accepted)", "var(--chart-3,#2a9d5c)", "{:.2f}")}
{chart(2, "Draft-token accuracy", "var(--chart-1,#3a7bd5)", "{:.1%}")}
<p class="sub">Accepted length here is measured on training batches, so it is optimistic. The real test comes
after training: drafting on held-out prompts against the full model.</p>"""
(HERE / "train-status.html").write_text(page)
print(f"step {last[0]}/{total} loss {last[1]:.2f} acc {last[2]:.3f} accept {last[3]:.2f} {sps:.2f} samples/s eta {eta_h:.1f}h ${spent:.0f}")
