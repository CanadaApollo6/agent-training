"""Fix test: list the rollouts each side lost to infrastructure (sandbox, server or harness errors, plus hung runs),
and write task files to rerun them on two servers per side, one rollout per task on each:
  tb2_lost_<arm>_s1.txt, _s2.txt  tasks that lost both rollouts go in both files; tasks that lost one are split
  between the two so each server gets about the same number
The task index in "rollout done" lines follows the tasks-file order. on-p1 was stopped after task 25; p1b ran the rest.

    python3 lost_runs.py
"""
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ERRORS = {"ProviderError", "HarnessError", "SandboxError"}
DONE = re.compile(r"rollout done: id=(\S+) task=(\d+) reward=\S+ turns=\d+ stop=(\S+)")

# (log name, tasks file, keep task index <= this)
SOURCES = {
    "off": [("fixes-off-p1", "tb2_noqemu_p1.txt", None), ("fixes-off-p2", "tb2_noqemu_p2.txt", None)],
    "on": [("fixes-on-p1", "tb2_noqemu_p1.txt", 25), ("fixes-on-p1b", "tb2_noqemu_p1b.txt", None),
           ("fixes-on-p2", "tb2_noqemu_p2.txt", None)],
}
# Rollouts that never logged "rollout done" (driver stopped by hand): (log name, task index)
HUNG = {"off": [("fixes-off-p2", 21)]}


def image_error_ids():
    """Runs that ended because the model asked to view an image (the server rejects image input). That is the
    model's own behaviour, the thing the image fix prevents, so these are results, not lost runs."""
    ids = set()
    for path in (HERE / "outputs" / "primeintellect").glob("terminal-bench-2--ornith35b-r1s-sd-fixes-*/traces.jsonl"):
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            traces = obj.get("traces") or []
            if traces and isinstance(traces[0], dict) and "image input" in str(obj.get("errors") or traces[0].get("errors") or ""):
                ids.add(traces[0].get("id"))
    return ids


def main():
    image_ids = image_error_ids()
    for arm, sources in SOURCES.items():
        lost = Counter()
        for name, tasks_file, last in sources:
            tasks = (HERE / tasks_file).read_text().split()
            log = (HERE / "logs" / f"tb2full-ornith35b-r1s-sd-{name}-prime_agent.log").read_text()
            for m in DONE.finditer(log):
                rid, i, stop = m.group(1), int(m.group(2)), m.group(3)
                if (last is None or i <= last) and stop in ERRORS and rid not in image_ids:
                    lost[tasks[i]] += 1
            for hung_name, i in HUNG.get(arm, []):
                if hung_name == name:
                    lost[tasks[i]] += 1
        twice = sorted(t for t, n in lost.items() if n >= 2)
        once = sorted(t for t, n in lost.items() if n == 1)
        s1, s2 = twice + once[0::2], twice + once[1::2]
        for k, part in (("s1", s1), ("s2", s2)):
            (HERE / f"tb2_lost_{arm}_{k}.txt").write_text("".join(t + "\n" for t in part))
        print(f"{arm}: {sum(lost.values())} lost rollouts on {len(lost)} tasks ({len(twice)} lost both);"
              f" servers get {len(s1)} and {len(s2)}")


if __name__ == "__main__":
    main()
