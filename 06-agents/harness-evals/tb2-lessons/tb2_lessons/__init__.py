"""tb2-lessons: the hub's Terminal-Bench 2 taskset (same tasks, images and grading) for the RLTL;DR-style lessons
experiment (lessons_loop.py). Two changes:

- LESSONS_FILE (JSON, task name -> list of one-sentence lessons) is appended to each listed task's instruction, the
  lessons the model wrote after its own earlier failed attempts.
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
        path = os.environ.get("LESSONS_FILE")
        lessons = json.loads(Path(path).read_text()) if path else {}
        for task in super().load():
            data = task.data
            notes = lessons.get(data.name.split("/")[-1])
            if notes:
                data = data.model_copy(update={"prompt": data.prompt + lesson_block(notes)})
            yield LessonsTask(data, self.config.task)


__all__ = ["TB2LessonsTaskset"]
