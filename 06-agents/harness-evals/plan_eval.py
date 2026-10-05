"""Plan first, then execute: does separating "work out what to do" from "do it" help a small model on TB2?

Each attempt, every task gets a fresh plan written by the served model in one call with no tools, then a normal
prime_agent run with that plan appended to the task (tb2-lessons, NOTES_MODE=plan). Two arms:

- blind (Riel's version): the plan is written from the task text alone ("imagine your harness is an IPython REPL ...
  what actions would you take"), and the run is told to carry the plan out.
- look: a short inspection-only run first (read-only, --look-turns), its transcript goes into the planning call, and
  the run follows the plan but rewrites the remaining steps when a step fails or a finding contradicts it.

Same tasks and settings as the reply-cap baseline (tasks.txt, 60 turns, 1-hour rollouts, 32K reply cap, server clamp).
Records go to results/plans/<name>/attempts.jsonl (task, attempt, reward, turns, stop, plan) and plans_<k>.json.

Usage: python plan_eval.py --arm blind|look --label MODEL --name NAME --base-url URL --key-var HF_TOKEN
       [--attempts 3] [--conc 4] [--look-turns 10]
"""

import argparse
import glob
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lessons_loop import chat, cut, reward, transcript

HERE = Path(__file__).parent

TASK_INTRO = """You will solve the task below in a sandbox where your only tool is an IPython REPL: persistent Python \
state, files you can read and edit, and shell commands run from Python when needed. Don't solve it now and don't run \
anything. Imagine the REPL in front of you and work out the best sequence of IPython actions to complete the task.

## Task
{task}
"""
LOOK_PART = """
## What a read-only inspection of the sandbox found (the actions taken and what came back)
{found}

Base the plan on what was actually found, not on guesses about the environment.
"""
PLAN_ASK = """
Write a numbered list of steps. For each step give its goal, what the code should do (briefly, not the full code), \
and how to check it worked before moving on. End with the check that shows the whole task is done. Write only the \
plan."""
EXECUTE_HEADER = {
    "blind": "Your goal is to complete the following set of IPython actions, writing Python code to fulfill them:",
    "look": None,   # tb2-lessons' default: follow the plan, rewrite the remaining steps when a step fails
}


def vf_eval(a, tasks: list[str], out: Path, log: Path, turns: int, timeout: int, env: dict) -> list[dict]:
    """One tb2-lessons run over the tasks; its traces' records."""
    subprocess.run(
        ["uv", "run", "--project", str(HERE), "vf-eval", "tb2-lessons", "-m", a.label,
         "--client.base-url", a.base_url, "--client.api-key-var", a.key_var,
         "--env.agent.harness.id", "prime_agent", "--env.agent.runtime.type", "prime",
         "--env.taskset.tasks", json.dumps(tasks), "--env.agent.max-turns", str(turns),
         "--env.agent.timeout.rollout", str(timeout), "--sampling.max-tokens", str(a.max_tokens),
         "-n", str(len(tasks)), "-r", "1", "-c", str(a.conc), "--no-push", "--no-rich", "-o", str(out)],
        cwd=HERE, stdout=open(log, "w"), stderr=subprocess.STDOUT, env={"LOCAL_KEY": "none", **os.environ, **env})
    recs = []
    for f in glob.glob(str(out / "**/traces.jsonl"), recursive=True):
        for line in open(f):
            r = json.loads(line)
            for t in r.get("traces") or []:
                recs.append({"task": r["task"]["data"]["name"].split("/")[-1], "prompt": r["task"]["data"]["prompt"],
                             "trace": t})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["blind", "look"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--key-var", default="LOCAL_KEY")
    ap.add_argument("--tasks", default=str(HERE / "tasks.txt"))
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--look-turns", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--lesson-tokens", type=int, default=32768, help="reply cap of the planning call")
    a = ap.parse_args()
    tasks = Path(a.tasks).read_text().split()
    state = HERE / "results/plans" / a.name
    state.mkdir(parents=True, exist_ok=True)
    done = [json.loads(line) for line in open(state / "attempts.jsonl")] if (state / "attempts.jsonl").exists() else []
    texts = {}
    for k in range(max((r["attempt"] for r in done), default=0) + 1, a.attempts + 1):
        out = HERE / "outputs/plans" / a.name / f"attempt-{k}"
        found = {}
        if a.arm == "look":
            print(f"{time.strftime('%T')} attempt {k}: inspecting {len(tasks)} tasks", flush=True)
            for r in vf_eval(a, tasks, out / "inspect", HERE / "logs" / f"plans-{a.name}-{k}-inspect.log",
                             a.look_turns, 900, {"NOTES_MODE": "inspect"}):
                found[r["task"]] = transcript(r["trace"], limit=30000)
        if not texts:   # the task texts, from a 0-turn-free source: the taskset itself
            from tb2_lessons import TB2LessonsTaskset
            from terminal_bench_2.taskset import TerminalBench2Config
            os.environ.pop("LESSONS_FILE", None)
            texts = {t.data.name.split("/")[-1]: t.data.prompt
                     for t in TB2LessonsTaskset(TerminalBench2Config(tasks=tasks)).load()}

        def plan(task):
            prompt = TASK_INTRO.format(task=cut(texts[task], 8000))
            if a.arm == "look":
                prompt += LOOK_PART.format(found=found.get(task) or "(the inspection run left no record)")
            try:
                return task, cut(chat(a, prompt + PLAN_ASK).strip(), 6000)
            except Exception as e:  # noqa: BLE001 - no plan: the run goes ahead with the bare task
                print(f"  plan call failed for {task}: {e!r}", flush=True)
                return task, ""

        print(f"{time.strftime('%T')} attempt {k}: planning", flush=True)
        with ThreadPoolExecutor(a.conc) as ex:
            plans = dict(ex.map(plan, tasks))
        plans_f = state / f"plans_{k}.json"
        plans_f.write_text(json.dumps(plans, indent=1))
        env = {"NOTES_MODE": "plan", "LESSONS_FILE": str(plans_f)}
        if EXECUTE_HEADER[a.arm]:
            env["PLAN_HEADER"] = EXECUTE_HEADER[a.arm]
        print(f"{time.strftime('%T')} attempt {k}: running {len(tasks)} tasks "
              f"({sum(bool(p) for p in plans.values())} with plans)", flush=True)
        recs = []
        for r in vf_eval(a, tasks, out / "run", HERE / "logs" / f"plans-{a.name}-{k}.log", a.turns, a.timeout, env):
            t = r["trace"]
            rec = {"task": r["task"], "attempt": k, "reward": reward(t), "turns": len(t["calls"]),
                   "stop": t.get("stop_condition"), "plan_chars": len(plans.get(r["task"], ""))}
            recs.append(rec)
            print(f"  {rec['task']:28} reward {rec['reward']:.0f}  turns {rec['turns']:3}  stop {rec['stop']}", flush=True)
        with (state / "attempts.jsonl").open("a") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in recs)
        print(f"{time.strftime('%T')} attempt {k} done: {sum(x['reward'] > 0 for x in recs)} solved of {len(recs)}",
              flush=True)


if __name__ == "__main__":
    main()
