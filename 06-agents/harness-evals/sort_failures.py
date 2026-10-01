"""Why did a run fail? Sort one eval's unsolved runs into setup, budget, harness misuse and task failure.

    python sort_failures.py OUTPUT_DIR [OUTPUT_DIR ...] [--show N]

Per failed run, the misuse events found in its trace (harness-level mistakes, not wrong answers):
- unknown_tool: a call to a tool the harness doesn't offer (prime_agent has only `ipython`)
- shell_as_python: shell written as Python in the REPL (`ls -la`, `cd /app`, `!pip ...`), a SyntaxError
- bad_args: a tool call whose arguments don't parse or lack `code`
- unawaited: a coroutine left unawaited (`bash()` / skills are async), so the call did nothing
- edit_miss: the `edit` skill's old_str wasn't in the file
- skill_misuse: a skill or `rlm` call raised TypeError/AttributeError (wrong signature or name)
- left_running: the run ended (agent_completed) right after starting background work it never read

Primary cause, first match wins:
1. setup: image input on a text-only server, harness crash, sandbox or provider failure. Not the model's doing.
2. budget: max_turns, timeout or context overflow.
3. misuse: the run ended because of the harness. Its last action was an unknown tool or left work running, or at
   least a third of its tool calls were misuse events.
4. task: the model finished in its own time with a wrong result.

Misuse events inside runs with other primary causes are counted too: they cost turns even when they aren't why the
run ended.
"""
import argparse
import collections
import json
import re
from pathlib import Path

SHELL_START = re.compile(r"^\s*(!|\$ |ls\b|cd\b|cat\b|pip\b|uv\b|apt(-get)?\b|git\b|make\b|python3?\s+-|grep\b|find\b|"
                         r"mkdir\b|rm\b|echo\b|chmod\b|curl\b|wget\b|npm\b|cargo\b|gcc\b|bash\b\s)", re.M)


def solved(t):
    return any(isinstance(v, dict) and v.get("score", 0) >= 1 for v in (t.get("rewards") or {}).values())


def misuse_events(t, tools):
    """(events, tool calls) for one trace."""
    ev, calls, pending = collections.Counter(), 0, {}
    msgs = [n["message"] for n in t["nodes"]]
    for m in msgs:
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                calls += 1
                fn = c.get("function") or c
                name, args = fn.get("name"), fn.get("arguments")
                if tools and name not in tools:
                    ev["unknown_tool"] += 1
                    continue
                try:
                    a = json.loads(args) if isinstance(args, str) else (args or {})
                except json.JSONDecodeError:
                    ev["bad_args"] += 1
                    continue
                if tools == {"ipython"}:
                    if not isinstance(a, dict) or "code" not in a:
                        ev["bad_args"] += 1
                        continue
                    pending[c.get("id")] = a["code"]
        elif m.get("role") == "tool":
            out = json.dumps(m.get("content"))
            code = pending.pop(m.get("tool_call_id"), "")
            if "SyntaxError" in out and SHELL_START.search(code or ""):
                ev["shell_as_python"] += 1
            elif "was never awaited" in out or re.search(r"<coroutine object", out):
                ev["unawaited"] += 1
            elif "old_str" in out and re.search(r"not found|did not match|no match|not in", out, re.I):
                ev["edit_miss"] += 1
            elif re.search(r"(TypeError|AttributeError)[^\n]{0,200}(rlm|edit|compact|goal|refine|agent_message|"
                           r"agent_observe|bash)", out):
                ev["skill_misuse"] += 1
    return ev, calls, msgs


def primary(t, ev, calls, msgs):
    stop, err = t.get("stop_condition"), json.dumps(t.get("errors") or "")
    if "image input" in err:
        return "setup", "image input on a text-only server"
    if stop in ("provider_error", "harness_error", "sandbox_error") or (t.get("errors") and stop != "agent_completed"):
        if "exceed" in err and "context" in err:
            return "budget", "context overflow"
        return "setup", f"{stop}: {err[:90]}"
    if stop in ("max_turns", "agent_timeout", "timeout") or t.get("is_timeout"):
        return "budget", stop
    last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), {})
    tail = " ".join(json.dumps(m) for m in msgs[-4:])
    if last.get("tool_calls") and ev["unknown_tool"]:
        return "misuse", "ended on an unknown tool call"
    if re.search(r"bash\([^)]*\)", tail) and re.search(r"(running|background|handle|wait|follow-up)", json.dumps(last.get("content")), re.I):
        ev["left_running"] += 1
        return "misuse", "ended its turn with work still running"
    if calls and sum(ev.values()) >= max(2, calls / 3):
        return "misuse", f"{sum(ev.values())} misuse events in {calls} calls"
    return "task", "finished with a wrong result"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--show", type=int, default=0, help="print the last assistant text of N task-bucket runs")
    a = ap.parse_args()
    for d in a.dirs:
        rows = [json.loads(l)["traces"][0] for l in open(Path(d) / "traces.jsonl")]
        tools = {x.get("name") or (x.get("function") or {}).get("name") for x in rows[0].get("tools") or []}
        fails = [r for r in rows if not solved(r)]
        buckets, why, total_ev, runs_with_ev, out = collections.Counter(), collections.Counter(), collections.Counter(), 0, []
        for r in fails:
            ev, calls, msgs = misuse_events(r, tools)
            cause, detail = primary(r, ev, calls, msgs)
            buckets[cause] += 1
            why[(cause, detail if cause != "setup" else detail[:40])] += 1
            total_ev.update(ev)
            runs_with_ev += bool(ev)
            out.append((cause, r["task"]["data"]["name"].split("/")[-1], calls, dict(ev), msgs))
        print(f"\n{Path(d).name}: {len(rows)} runs, {len(rows) - len(fails)} solved, {len(fails)} failed")
        for c in ("setup", "budget", "misuse", "task"):
            print(f"  {c:7} {buckets[c]:3}  ({buckets[c] / max(1, len(fails)):.0%})")
        print("  detail:", dict(why.most_common()))
        print(f"  failed runs with any misuse event: {runs_with_ev}/{len(fails)}; events: {dict(total_ev)}")
        for cause, name, calls, ev, msgs in out:
            if cause == "misuse" or ev:
                print(f"    {cause:7} {name[:34]:34} calls={calls:3} {ev}")
        shown = 0
        for cause, name, calls, ev, msgs in out:
            if cause == "task" and shown < a.show:
                shown += 1
                last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), {})
                print(f"\n  --- {name} ({calls} calls) last words:\n  {str(last.get('content'))[:500]}")


if __name__ == "__main__":
    main()
