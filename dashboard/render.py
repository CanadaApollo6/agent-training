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
    now = json.loads((HERE / "now.json").read_text()) if (HERE / "now.json").exists() else {}
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

    rerun_rows = "".join(
        f'<div class="srow" style="grid-template-columns:70px 1fr 130px"><span>{e(r["label"])}</span>'
        + bar([(r["done"], "var(--primary)", "done"), (r["expected"] - r["done"], "var(--border)", "left")], 6)
        + f'<span class="num">{r["done"]}/{r["expected"]} · {r["solved"]} solved</span></div>'
        for r in live.get("reruns", []))

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

    final = live.get("final") if not live["interim"] else None
    if final:
        pa, pn, pt, po = final["paired_all"], final["paired_no_heavy"], final["picture_tasks"], final["paired_other"]
        ci = lambda p: f'{100 * p["ci95"][0]:+.0f} to {100 * p["ci95"][1]:+.0f}'
        fx = {k: final[f"fix_{k}"] for k in ("cutoff", "check", "image")}
        lost = {a: final[a]["runs"] - final[a]["reruns"] - (final[a]["usable"] - final[a]["rerun_usable"]) for a in ("off", "on")}
        reruns = final["off"]["reruns"] + final["on"]["reruns"]
        recovered = final["off"]["rerun_usable"] + final["on"]["rerun_usable"]
        row = lambda label, off, on, note="": (f'<tr><td>{label}<div class="muted small">{note}</div></td>'
                                               f'<td class="num">{off}</td><td class="num">{on}</td></tr>')
        test_html = f"""
<h2>The fix test: result</h2>
<div class="muted">87 Terminal-Bench 2 tasks, two tries each, with three small fixes off and on. Every run lost to a
failure outside the model got one more try.</div>
<table style="margin-top:8px"><tr><th></th><th>Fixes off</th><th>Fixes on</th></tr>
{row("Usable runs (of 174)", final["off"]["usable"], final["on"]["usable"], "Lost to sandbox, server or network failures: left out")}
{row("Solved", f'{final["off"]["solved"]} ({pct(final["off"]["solved"], final["off"]["usable"])})', f'{final["on"]["solved"]} ({pct(final["on"]["solved"], final["on"]["usable"])})')}
{row("Average solve rate per task", f'{100 * pa["off_rate"]:.0f}%', f'{100 * pa["on_rate"]:.0f}%', f'{pa["tasks"]} tasks with usable runs on both sides')}
{row("Same, without the 7 heaviest tasks", f'{100 * pn["off_rate"]:.0f}%', f'{100 * pn["on_rate"]:.0f}%', "Prime shuts their sandboxes down in most runs")}
{row("Tasks that involve a picture", f'{pt["off"]["solved"]} of {pt["off"]["usable"]} ({pct(pt["off"]["solved"], pt["off"]["usable"])})', f'{pt["on"]["solved"]} of {pt["on"]["usable"]} ({pct(pt["on"]["solved"], pt["on"]["usable"])})', f'{pt["tasks"]} tasks')}
{row("All other tasks, per task", f'{100 * po["off_rate"]:.0f}%', f'{100 * po["on_rate"]:.0f}%', f'{po["tasks"]} tasks, no pictures, not heavy')}
</table>
<div class="note"><b>Probably a small gain, not proven.</b> Fixes on is ahead by {100 * pa["mean_diff"]:.0f} points per task,
but the likely range runs from {ci(pa)} points, so it could be nothing (p = {pa["p_two_sided"]:.2f}). Fixes on did better on
{pa["on_better"]} tasks, fixes off on {pa["off_better"]}, and {pa["same"]} came out the same. The whole gain is on tasks
with pictures.</div>

<h2>What each fix did</h2>
<table><tr><th>Fix</th><th>What happened</th></tr>
<tr><td><b>Image</b><div class="muted small">A picture becomes a text note instead of an error</div></td>
<td>The clear win. Without it, {final["off_image_endings"]} runs ended when the model asked to see a picture, and none of
them was solved. Picture tasks went from {pct(pt["off"]["solved"], pt["off"]["usable"])} to {pct(pt["on"]["solved"], pt["on"]["usable"])} solved.</td></tr>
<tr><td><b>Check</b><div class="muted small">Re-check the outputs before saying done</div></td>
<td>Maybe a small help. When the model said it was done, {pct(final["off_said_done"]["solved"], final["off_said_done"]["runs"])}
of those runs were solved without the check and {pct(fx["check"]["solved"], fx["check"]["runs"])} with it.</td></tr>
<tr><td><b>Cut-off</b><div class="muted small">Retry a reply that ran out of room, with thinking off</div></td>
<td>Keeps runs going but rarely saves them: {fx["cutoff"]["solved"]} of {fx["cutoff"]["runs"]} runs where it fired were solved.
A model that thinks itself into the limit is usually on a task it was going to fail.</td></tr>
</table>

<h2>The main problem now: Prime shutting sandboxes down</h2>
<div>{lost["off"]} runs with fixes off and {lost["on"]} with fixes on were lost the first time, mostly to "the sandbox has
been terminated". The retry recovered {recovered} of {reruns}; the rest were lost again. Seven heavy tasks (compiling,
training, rendering) lose their sandbox in most runs on both sides. RL on Prime sandboxes needs this fixed or worked
around first.</div>"""
    else:
        test_html = f"""<h2>The fix test, running now</h2>
<div class="muted">87 Terminal-Bench 2 tasks, two tries each, with three small fixes off and on. Started {e(live["started"])};
expected to finish around <b>{e(live["eta"])}</b>.</div>
<div class="arms" style="margin-top:10px">{"".join(arm_html)}</div>
<div class="legend">{legend}</div>
<div class="note">{e(live["noise"])} Runs lost to failures outside the model (sandboxes shut down mid-run on Prime's side,
server errors, a wifi drop here at 4:45 p.m.): {arms["off"]["errors"]} with fixes off, {arms["on"]["errors"]} with fixes on.
The second figure leaves those out.</div>

<h2>Rerunning the lost runs</h2>
<div class="muted" style="margin-bottom:8px">Every run lost to a failure outside the model gets one more try, on four more
servers, so both sides end with about the same number of usable runs. Started 5:10 p.m.; these are not in the
figures above yet.</div>
{rerun_rows}

<h2>How often each fix fired</h2>
<table><tr><th>Fix</th><th>Fired</th><th>Then</th></tr>{fixes}</table>
<div class="muted small" style="margin-top:4px">The image fix fires on every request that carries a picture, so its count is requests, not runs.</div>

<h2>Tasks where the two sides differ so far</h2>
<div class="muted">Tasks both sides have finished, runs lost to failures left out: fixes on did better on <b class="good">{better}</b>,
fixes off on <b class="bad">{worse}</b>.</div>
<table style="margin-top:6px"><tr><th>Task</th><th>Off</th><th>On</th><th></th></tr>{diff_rows}</table>

"""
    history = list(d["history"]) + [{
        "when": "6 Oct", "title": "Hard Terminal-Lego round (overnight)",
        "outcome": f"140 hard tasks, two tries each: 117 of 280 solved. 31 tasks solved exactly once, the kind worth "
                   f"training on; {lego['candidates']} with the pilot's."}]
    if final:
        history.append({"when": "6 Oct", "title": "The three fixes, 87 tasks, off vs on",
                        "outcome": f"About {100 * final['paired_all']['mean_diff']:.0f} points better per task with the fixes, "
                                   "not proven at two tries. The image fix is the real win; keep it."})
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

    now_html = ""
    if now.get("rl"):
        r = now["rl"]
        rows = "".join(f'<tr><td class="num">{e(x["step"])}</td><td>{e(x["time"])}</td><td class="num">{e(x["reward"])}</td>'
                       f'<td class="num">{e(x["cut"])}</td></tr>' for x in r["steps"])
        now_html += f"""<h2>{e(r["title"])}</h2><div class="muted">{e(r["intro"])}</div>
<table style="margin-top:8px"><tr><th>Step</th><th>Time</th><th>Average reward</th><th>Attempts cut off by length</th></tr>{rows}</table>
<div class="note">{e(r["note"])}</div>"""
    if now.get("qwen"):
        q = now["qwen"]
        rows = "".join(f'<tr><td>{e(a)}</td><td class="num">{e(b)}</td><td class="num"><b>{e(c)}</b></td><td class="num">{e(x)}</td></tr>'
                       for a, b, c, x in q["rows"])
        now_html += f"""<h2>{e(q["title"])}</h2><div class="muted">{e(q["intro"])}</div>
<table style="margin-top:8px"><tr><th>Words per second</th><th>No draft</th><th>With draft model</th><th>Old best (EXL3)</th></tr>{rows}</table>
<div class="note">{e(q["context"])}</div><div class="muted" style="margin-top:6px">{e(q["next"])}</div>"""

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

{now_html}

{test_html}

<h2>What has been tried</h2>{hist}

<h2>Tasks worth training on (Terminal-Lego)</h2>
<div class="muted">Two tries per task: solved both times, <span class="goodtxt">solved once</span>, or never. The solved-once
tasks are what RL learns from: <b>{lego["candidates"]}</b> so far. {lego["pool_tried"]} of {lego["pool_eligible"]} hard tasks used.</div>
{lego_rows}

<h2>Next, and spend</h2>
<div>{e(now["next_text"]) if now.get("next_text") else f'RL is next. A first run on the 35B model costs about <b>{e(d["next"]["thirty_five_b"])}</b>, over the usual {e(d["next"]["nightly"])} a night, so it waits for your call; proving the pipeline on the 9B first is cheaper.'}</div>
<div class="muted" style="margin-top:6px">GPU servers: about ${spend["last_night"]["dollars"]} last night, about
${spend["today"]["dollars"] + now.get("extra_today_dollars", 0)} today{e(now.get("extra_today_note", ""))}. Sandbox fees not included.</div>
</body></html>"""
    (HERE / "inline.html").write_text(page)
    print(f"wrote inline.html ({len(page)} chars); differing tasks {len(diffs)}: on better {better}, off better {worse}")


if __name__ == "__main__":
    main()
