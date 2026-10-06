"""data.json (from build.py) -> inline.html: one self-contained page for T3's inline html_render, drawn with T3's
theme variables (--foreground, --success, ...) so it follows the thread's light/dark theme.

    python3 build.py && python3 render.py
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).parent


def e(s) -> str:
    return html.escape(str(s))


def pct(a, b) -> str:
    return f"{100 * a / b:.0f}%" if b else "–"


def bar(parts, height=10) -> str:
    """A stacked bar: parts = [(count, css colour, title)]."""
    total = sum(n for n, _, _ in parts) or 1
    segs = "".join(f'<span title="{e(t)}: {n}" style="width:{100 * n / total:.2f}%;background:{c}"></span>'
                   for n, c, t in parts if n)
    return f'<div class="bar" style="height:{height}px">{segs}</div>'


def main():
    d = json.loads((HERE / "data.json").read_text())
    live, lego, spend = d["live"], d["lego"], d["spend"]
    arms = {a["id"]: a for a in live["arms"]}
    ending_colour = {"finished": "var(--muted-foreground)", "time": "var(--warning)", "turns": "#a78bfa",
                     "image": "var(--info)", "error": "var(--destructive)"}

    # Fix test, per arm
    arm_html = []
    for a in (arms["off"], arms["on"]):
        clean = a["done"] - a["errors"]
        servers = "".join(
            f'<div class="srow"><span>{e(s["label"])}</span>{bar([(s["done"], "var(--primary)", "done"), (s["expected"] - s["done"], "var(--border)", "left")], 6)}'
            f'<span class="num">{s["done"]}/{s["expected"]}</span></div>' for s in a["servers"])
        endings = bar([(x["count"], ending_colour.get(x["id"], "var(--border)"), x["label"]) for x in a["endings"]], 12)
        arm_html.append(f"""
        <div class="arm">
          <div class="armhead"><b>{e(a["label"])}</b><span class="muted">{a["done"]} of {a["expected"]} runs</span></div>
          {servers}
          <div class="big">{a["solved"]} solved <span class="muted">({pct(a["solved"], a["done"])} of finished runs)</span></div>
          <div class="muted small">{pct(a["solved"], clean)} leaving out the {a["errors"]} runs lost to sandbox or server failures</div>
          <div class="lbl">How runs ended</div>{endings}
        </div>""")
    legend = "".join(f'<span><i style="background:{ending_colour[x["id"]]}"></i>{e(x["label"])}</span>'
                     for x in arms["off"]["endings"] if x["id"] in ending_colour)

    # Tasks where the two sides differ (only tasks both sides have finished, failures from infra left out)
    diffs = []
    for t in live["tasks"]:
        off = [r for r in t["off"] if not r["error"]]
        on = [r for r in t["on"] if not r["error"]]
        if len(t["off"]) < live["tries"] or len(t["on"]) < live["tries"] or not off or not on:
            continue
        so, sn = sum(r["solved"] for r in off), sum(r["solved"] for r in on)
        if so / len(off) != sn / len(on):
            diffs.append((sn / len(on) - so / len(off), t["name"], f"{so}/{len(off)}", f"{sn}/{len(on)}"))
    diffs.sort(reverse=True)
    better = sum(1 for x in diffs if x[0] > 0)
    worse = len(diffs) - better
    diff_rows = "".join(
        f'<tr><td>{e(n)}</td><td class="num">{o}</td><td class="num">{w}</td>'
        f'<td class="{"good" if delta > 0 else "bad"}">{"fixes on better" if delta > 0 else "fixes off better"}</td></tr>'
        for delta, n, o, w in diffs)

    fixes = "".join(
        f'<tr><td><b>{e(f["label"])}</b><div class="muted small">{e(f["detail"])}</div></td>'
        f'<td class="num">{f["fired"]}</td>'
        f'<td class="num">{(str(f["acted"]) + " acted") if f["acted_known"] else "–"}</td></tr>' for f in live["fixes"])

    history = list(d["history"]) + [{
        "when": "6 Oct", "title": "Hard Terminal-Lego round (overnight)",
        "outcome": f"140 hard tasks, two tries each: 117 of 280 solved. 31 tasks solved exactly once, the kind worth "
                   f"training on; {lego['candidates']} with the pilot's."}]
    hist = "".join(f'<div class="ev"><div class="when">{e(h["when"])}</div><div><b>{e(h["title"])}</b>'
                   f'<div class="muted">{e(h["outcome"])}</div></div></div>' for h in history)

    lego_rows = "".join(
        f'<div class="lrow"><span>{e(s["label"])}<span class="muted small"> · {s["tasks"]} tasks</span></span>'
        + bar([(s["always"], "var(--muted-foreground)", "always solved"), (s["sometimes"], "var(--success)", "solved sometimes"),
               (s["never"], "var(--border)", "never solved")], 14)
        + f'<span class="num">{s["always"]} · <b class="goodtxt">{s["sometimes"]}</b> · {s["never"]}</span></div>'
        for s in lego["sets"])

    stats = "".join(f'<div class="stat"><div class="sv">{e(n["value"])}</div><div class="muted small">{e(n["label"])}</div></div>'
                    for n in d["headline"]["numbers"])

    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:var(--background);color:var(--foreground);font:14px/1.5 system-ui,sans-serif}}
h2{{font-size:15px;margin:26px 0 8px}} .muted{{color:var(--muted-foreground)}} .small{{font-size:12px}}
.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:10px}}
.stat{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 12px}}
.sv{{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}}
.arms{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.arm{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}}
.armhead{{display:flex;justify-content:space-between;margin-bottom:6px}}
.srow{{display:grid;grid-template-columns:62px 1fr 56px;gap:8px;align-items:center;font-size:12px}}
.big{{font-size:17px;font-weight:600;margin-top:10px}} .lbl{{font-size:12px;margin:10px 0 4px}}
.bar{{display:flex;border-radius:4px;overflow:hidden;background:var(--border)}} .bar span{{display:block}}
.legend{{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--muted-foreground);margin-top:8px}}
.legend i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}}
.note{{background:var(--warning-surface,var(--secondary));border-radius:8px;padding:8px 12px;margin-top:10px;font-size:13px}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:6px 4px;border-bottom:1px solid var(--border);vertical-align:top;text-align:left}}
th{{font-size:12px;font-weight:500;color:var(--muted-foreground)}}
.good{{color:var(--success)}} .bad{{color:var(--destructive)}} .goodtxt{{color:var(--success)}}
.ev{{display:grid;grid-template-columns:52px 1fr;gap:10px;padding:7px 0;border-bottom:1px solid var(--border)}}
.when{{color:var(--muted-foreground);font-size:12px;padding-top:2px}}
.lrow{{display:grid;grid-template-columns:170px 1fr 90px;gap:10px;align-items:center;margin:6px 0;font-size:13px}}
@media (max-width:520px){{.lrow{{grid-template-columns:1fr;gap:3px}}}}
</style></head><body>
<div class="muted small">{e(d["model"]["name"])} · updated {e(d["generated_label"])}</div>
<div style="margin-top:6px">{e(d["headline"]["paragraph"])}</div>
<div class="stats">{stats}</div>

<h2>The fix test, running now</h2>
<div class="muted">87 Terminal-Bench 2 tasks, two tries each, with three small fixes off and on. Started {e(live["started"])};
expected to finish around <b>{e(live["eta"])}</b>.</div>
<div class="arms" style="margin-top:10px">{"".join(arm_html)}</div>
<div class="legend">{legend}</div>
<div class="note">{e(live["noise"])} Both sides lose about the same number of runs to sandboxes shut down mid-run on
Prime's side; the second figure leaves those out.</div>

<h2>How often each fix fired</h2>
<table><tr><th>Fix</th><th>Fired</th><th>Then</th></tr>{fixes}</table>
<div class="muted small" style="margin-top:4px">The image fix fires on every request that carries a picture, so its count is requests, not runs.</div>

<h2>Tasks where the two sides differ so far</h2>
<div class="muted">Tasks both sides have finished, runs lost to failures left out: fixes on did better on <b class="good">{better}</b>,
fixes off on <b class="bad">{worse}</b>.</div>
<table style="margin-top:6px"><tr><th>Task</th><th>Off</th><th>On</th><th></th></tr>{diff_rows}</table>

<h2>What has been tried</h2>{hist}

<h2>Tasks worth training on (Terminal-Lego)</h2>
<div class="muted">Two tries per task: solved both times, <span class="goodtxt">solved once</span>, or never. The solved-once
tasks are what RL learns from: <b>{lego["candidates"]}</b> so far. {lego["pool_tried"]} of {lego["pool_eligible"]} hard tasks used.</div>
{lego_rows}

<h2>Next, and spend</h2>
<div>RL is next. A first run on the 35B model costs about <b>{e(d["next"]["thirty_five_b"])}</b>, over the usual
{e(d["next"]["nightly"])} a night, so it waits for your call; proving the pipeline on the 9B first is cheaper.</div>
<div class="muted" style="margin-top:6px">GPU servers: about ${spend["last_night"]["dollars"]} last night, about
${spend["today"]["dollars"]} today (${spend["today"]["duplicate_dollars"]} of it the duplicate launch). Sandbox fees not included.</div>
</body></html>"""
    (HERE / "inline.html").write_text(page)
    print(f"wrote inline.html ({len(page)} chars); differing tasks {len(diffs)}: on better {better}, off better {worse}")


if __name__ == "__main__":
    main()
