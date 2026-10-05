"""RLTL;DR's in-context half on Terminal-Bench 2, no training: does Ornith solve tasks it never has when each attempt
carries the one-sentence lessons it wrote after its own failed attempts?

Round k runs every still-unsolved task once under prime_agent (tb2-lessons taskset: the hub's tasks and grading, plus
the lessons so far in the instruction). After each failed attempt the same served model reads a condensed transcript
of its attempt and the tail of the test output, thinks, and writes a summary, the error, a fix, and a one-line TL;DR.
Only the TL;DR is kept, as in the paper. A task leaves the loop once solved.

    uv run python lessons_loop.py --label ornith35b-r1s --attempts 6 --name r1s-unsolved5
    HF_TOKEN=... uv run python lessons_loop.py --base-url https://<job>--8080.hf.jobs/v1 --key-var HF_TOKEN ...

Runs go to outputs/lessons/<name>/attempt-<k> (harvest.py skips outputs/lessons: these prompts carry lessons). State
in results/lessons/<name>/: lessons.json (task -> lessons, also LESSONS_FILE for the runs) and attempts.jsonl (one
record per attempt with reward, turns, test-output tail, lesson). Rerunning resumes after the last finished round.
Never point it at the held-out tasks (tasks.txt): their test output would leak into the lessons.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

LESSON_PROMPT = """You just attempted the task below in a sandbox and the task's tests failed. Read your attempt and \
the test output, and work out what went wrong.

## Task
{task}

## Your attempt (condensed: your actions and what came back, long outputs cut)
{transcript}

## Test output (tail)
{tests}

Write, in order:
1. Summary: what you did, in two or three sentences.
2. Feedback: why the tests failed. Be specific: which test, which requirement, which assumption was wrong.
3. Error step: the first action where the attempt went wrong.
4. Fix: what to do differently next time.
5. A final line starting with "TL;DR:" and one sentence, at most about 25 words, that would most help a fresh attempt \
at this exact task pass. State a concrete fact or rule about the task, not general advice."""


def text_of(content) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n // 2] + f"\n[... {len(s) - n} chars cut ...]\n" + s[-n // 2:]


def transcript(trace: dict, limit=60000) -> str:
    """The main conversation (the longest call path; side calls such as prime_agent's /refine gate start new roots):
    each action (answer text, tool calls) and each tool result."""
    nodes = trace["nodes"]

    def walk(i):
        out = []
        while i is not None:
            out.append(nodes[i]["message"] or {})
            i = nodes[i].get("parent")
        return out

    path = max((walk(c["node"]) for c in trace["calls"] if c.get("node") is not None), key=len, default=[])
    lines = []
    for m in reversed(path):
        role = m.get("role")
        if role == "assistant":
            body = text_of(m.get("content")).strip()
            calls = [c.get("function", c) for c in m.get("tool_calls") or []]
            calls = "\n".join(f"[{c.get('name')}] {cut(str(c.get('arguments', '')), 2500)}" for c in calls)
            lines.append("ACTION:\n" + cut(body, 1500) + ("\n" + calls if calls else ""))
        elif role == "tool":
            lines.append("RESULT:\n" + cut(text_of(m.get("content")), 1500))
    return cut("\n\n".join(lines), limit)


def chat(a, prompt: str) -> str:
    body = json.dumps({"model": a.label, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": a.lesson_tokens}).encode()
    req = urllib.request.Request(a.base_url + "/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {os.environ.get(a.key_var, 'none')}"})
    return json.load(urllib.request.urlopen(req, timeout=3600))["choices"][0]["message"]["content"] or ""


def reward(t: dict) -> float:
    return sum((v or {}).get("score", 0) * (v or {}).get("weight", 1) for v in (t.get("rewards") or {}).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="the served model name")
    ap.add_argument("--name", required=True, help="experiment name (output and state directories)")
    ap.add_argument("--tasks", default=str(HERE / "datagen/unsolved_text.txt"))
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key-var", default="LOCAL_KEY",
                    help="env var holding the API key (HF_TOKEN for a server behind the HF jobs proxy)")
    ap.add_argument("--conc", type=int, default=5)
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="per-call cap for the attempts (prime_agent's default 16K cuts long first thoughts, an empty "
                         "length-capped reply ends the rollout as if done)")
    ap.add_argument("--lesson-tokens", type=int, default=32768)
    a = ap.parse_args()
    held = set((HERE / "tasks.txt").read_text().split())
    tasks = Path(a.tasks).read_text().split()
    assert not held & set(tasks), f"held-out tasks in the list: {held & set(tasks)}"
    state = HERE / "results/lessons" / a.name
    state.mkdir(parents=True, exist_ok=True)
    lessons_f, attempts_f = state / "lessons.json", state / "attempts.jsonl"
    lessons = json.loads(lessons_f.read_text()) if lessons_f.exists() else {}
    done = [json.loads(line) for line in open(attempts_f)] if attempts_f.exists() else []
    solved = {r["task"] for r in done if r["reward"] > 0}
    start = max((r["attempt"] for r in done), default=0) + 1
    for k in range(start, a.attempts + 1):
        left = [t for t in tasks if t not in solved]
        if not left:
            break
        lessons_f.write_text(json.dumps(lessons, indent=1))
        out = HERE / "outputs/lessons" / a.name / f"attempt-{k}"
        print(f"{time.strftime('%T')} attempt {k}: {len(left)} tasks", flush=True)
        log = open(HERE / "logs" / f"lessons-{a.name}-{k}.log", "w")
        subprocess.run(
            ["uv", "run", "--project", str(HERE), "vf-eval", "tb2-lessons", "-m", a.label,
             "--client.base-url", a.base_url, "--client.api-key-var", a.key_var,
             "--env.agent.harness.id", "prime_agent", "--env.agent.runtime.type", "prime",
             "--env.taskset.tasks", json.dumps(left), "--env.agent.max-turns", str(a.turns),
             "--env.agent.timeout.rollout", str(a.timeout), "--sampling.max-tokens", str(a.max_tokens), "-n", str(len(left)), "-r", "1", "-c", str(a.conc),
             "--no-push", "--no-rich", "-o", str(out)],
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env={"LOCAL_KEY": "none", **os.environ,
                                                                  "LESSONS_FILE": str(lessons_f)})
        recs = []
        for f in glob.glob(str(out / "**/traces.jsonl"), recursive=True):
            for line in open(f):
                r = json.loads(line)
                if not r.get("traces"):
                    continue
                t = r["traces"][0]
                task = r["task"]["data"]["name"].split("/")[-1]
                tests = (t.get("info") or {}).get("test_output", "")
                rec = {"task": task, "attempt": k, "reward": reward(t), "turns": len(t["calls"]),
                       "stop": t.get("stop_condition"), "errors": [str(e)[:300] for e in r.get("errors") or []],
                       "lessons_in_prompt": len(lessons.get(task, [])), "tests": tests[-2000:]}
                if rec["reward"] <= 0 and tests:
                    prompt = LESSON_PROMPT.format(task=cut(r["task"]["data"]["prompt"], 8000),
                                                  transcript=transcript(t), tests=cut(tests, 6000))
                    try:
                        reply = chat(a, prompt)
                    except Exception as e:  # noqa: BLE001 - keep the round's records; this task just gets no lesson
                        reply = f"(lesson call failed: {e!r})"
                    m = re.findall(r"TL;DR:\s*(.+)", reply)
                    rec["lesson"] = m[-1].strip() if m else None
                    rec["lesson_reply"] = reply[-3000:]
                    if rec["lesson"]:
                        lessons.setdefault(task, []).append(rec["lesson"])
                recs.append(rec)
                print(f"  {task:28} reward {rec['reward']:.0f}  turns {rec['turns']:3}  lesson: {rec.get('lesson')}",
                      flush=True)
        with attempts_f.open("a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in recs)
        solved |= {r["task"] for r in recs if r["reward"] > 0}
        lessons_f.write_text(json.dumps(lessons, indent=1))
    print(f"solved {len(solved)}/{len(tasks)}: {sorted(solved)}")


if __name__ == "__main__":
    main()
