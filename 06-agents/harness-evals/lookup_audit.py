"""Find Terminal-Bench 2 runs where the agent looked the benchmark itself up online: fetched the task's tests or
reference solution from the TB2 / Harbor repos (or anything else naming terminal-bench) during the run.

Flint's card counts "solutions obtained through web retrieval" as failures; this applies the same rule to any build.
A run is flagged when a tool call's arguments name one of the benchmark sources below. Prints one line per flagged run
(build, attempt, task, solved, the matching command) and a per-build count.

    uv run lookup_audit.py [label prefix ...]     # default: flint-bf16 ornith35b-r1s-c4s ornith35b-q8
"""
import glob
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).parent / "outputs" / "primeintellect"
# A URL that names the benchmark or its publishers, or an install/clone of it. Bare mentions don't count: mteb-retrieve's
# query is literally "terminal-bench", and custom-memory-heap-crash ships a "Terminal-Bench Canary String".
SOURCES = re.compile(r"https?://[^\s\"'\\]*(terminal-bench|terminal_bench|tbench|harbor-framework|laude-institute)"
                     r"|(git clone|pip install|uv pip install|uvx|npm install)[^\n\"]{0,80}(terminal-bench|tbench|harbor)",
                     re.I)


def audit(prefix):
    flagged, total = [], 0
    for d in sorted(glob.glob(str(OUT / f"terminal-bench-2--{prefix}-a*--pi--*"))):
        attempt = re.search(rf"{re.escape(prefix)}-a(\d+)", d)[1]
        for line in open(Path(d) / "traces.jsonl"):
            r = json.loads(line)
            t = r["traces"][0] if r.get("traces") else {}
            name = (t.get("task", {}).get("data", {}).get("name") or json.dumps(r.get("task"))).split("/")[-1].strip('"')
            rewards = t.get("rewards", {})
            solved = (rewards.get("solved") or rewards.get("reward") or {}).get("score", 0.0) >= 1.0
            total += 1
            hits = []
            for call in t.get("calls", []):
                for tc in (call.get("response") or call).get("tool_calls", []) if isinstance(call, dict) else []:
                    args = json.dumps(tc)
                    if SOURCES.search(args):
                        hits.append(args)
            if not hits:  # tool-call layout differs between harness versions: fall back to the raw record
                for m in re.finditer(r'"arguments": "((?:[^"\\]|\\.)*)"', json.dumps(t)):
                    if SOURCES.search(m[1]):
                        hits.append(m[1])
            if hits:
                cmd = SOURCES.search(hits[0])
                s = hits[0]
                flagged.append((attempt, name, solved, s[max(0, cmd.start() - 120):cmd.end() + 80]))
    return flagged, total


def main():
    for prefix in sys.argv[1:] or ["flint-bf16", "ornith35b-r1s-c4s", "ornith35b-q8"]:
        flagged, total = audit(prefix)
        print(f"## {prefix}: {len(flagged)} of {total} runs looked the benchmark up "
              f"({sum(f[2] for f in flagged)} of those scored as solved)")
        for a, name, solved, snippet in flagged:
            print(f"- a{a} {name} solved={solved}: …{snippet}…")
        print()


if __name__ == "__main__":
    main()
