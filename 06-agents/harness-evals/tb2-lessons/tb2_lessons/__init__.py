"""tb2-lessons: the hub's Terminal-Bench 2 taskset (same tasks, images and grading) for the RLTL;DR-style lessons
experiment (lessons_loop.py). Two changes:

- LESSONS_FILE (JSON, task name -> list of one-sentence lessons) is appended to each listed task's instruction, the
  lessons the model wrote after its own earlier failed attempts.
- NOTES_MODE picks what the file holds (default "lessons"). "plan": task name -> a plan the model wrote before
  starting, with PLAN_HEADER (env, optional) as its heading. "inspect": every task gets INSPECT_NOTE instead, an
  inspection-only run whose transcript a plan is then written from (plan_eval.py).
- The tail of test.sh's output is kept in trace.info["test_output"], so a failed attempt can be turned into a lesson.
  The upstream verifier runs the same script and keeps only the reward.
"""
import json
import os
from pathlib import Path

from terminal_bench_2 import TerminalBench2Taskset
from verifiers.v1.errors import SandboxError
from verifiers.v1.tasksets.harbor.taskset import MAX_REWARD_BYTES, HarborTask, resolve_env

TEST_TAIL = 12000     # characters of test output kept


def lesson_block(lessons: list[str]) -> str:
    return ("\n\nNotes you wrote after your earlier attempts at this task, each of which failed the task's tests:\n"
            + "\n".join(f"- {x}" for x in lessons))


PLAN_HEADER = ("A plan you wrote for this task before starting. Follow it. When a step fails or something you find "
               "contradicts it, rewrite the remaining steps before going on.")
INSPECT_NOTE = ("\n\nTHIS RUN IS FOR INSPECTION ONLY. Do not solve the task yet, and do not change, install, create or "
                "delete anything. Read files, list directories, check tool and library versions, read any tests or "
                "examples you can find, and run only read-only commands, to gather the facts needed to plan a "
                "solution. Use at most a few steps, then stop and summarize what you found.")


def plan_block(plan: str) -> str:
    return f"\n\n{os.environ.get('PLAN_HEADER') or PLAN_HEADER}\n\n{plan}"


class LessonsTask(HarborTask):
    async def run_verifier(self, runtime, trace):
        result = await runtime.run(["bash", "/tests/test.sh"], resolve_env(self.data.verifier_env))
        trace.info["test_output"] = ((result.stdout or "") + "\n" + (result.stderr or ""))[-TEST_TAIL:]
        trace.info["test_exit_code"] = result.exit_code
        # the rest is HarborTask.run_verifier after its test.sh call
        scores = await self.read_reward_json(runtime)
        if scores is not None:
            if isinstance(scores, dict) and "reward" in scores:
                trace.record_metrics({key: value for key, value in scores.items() if key != "reward"})
                return {"reward": scores["reward"]}
            return scores
        try:
            reward = (await runtime.read("/logs/verifier/reward.txt", max_bytes=MAX_REWARD_BYTES)).decode().strip()
            return float(reward or 0)
        except (SandboxError, OSError, ValueError):
            return 0.0


class TB2LessonsTaskset(TerminalBench2Taskset):
    def load(self):
        path, mode = os.environ.get("LESSONS_FILE"), os.environ.get("NOTES_MODE", "lessons")
        lessons = json.loads(Path(path).read_text()) if path else {}
        for task in super().load():
            data = task.data
            notes = lessons.get(data.name.split("/")[-1])
            extra = (INSPECT_NOTE if mode == "inspect" else plan_block(notes) if mode == "plan" and notes
                     else lesson_block(notes) if mode == "lessons" and notes else "")
            if extra:
                data = data.model_copy(update={"prompt": data.prompt + extra})
            yield LessonsTask(data, self.config.task)


__all__ = ["TB2LessonsTaskset"]
