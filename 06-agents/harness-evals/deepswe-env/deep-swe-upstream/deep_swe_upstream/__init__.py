"""primeintellect/deep-swe, with each task's image pointed at our own copy on Prime.

The hub taskset rewrites every image to Prime's team registry (prime/prime/...), which other accounts can't read
(HTTP 403). The upstream images are public on AWS ECR, but the Prime runtime only auto-builds VM images from Docker Hub
refs, so images.jsonl + `prime images push-bulk --manifest images.jsonl --private` copies them into our registry once.

Grading: upstream submits only *committed* work (`git diff <base> HEAD`), and the instruction says to commit. Our
agents almost never do, so every patch came out empty and every score 0, saying nothing about the code. Here the
verifier grades all of the agent's changes, committed or not (tracked edits plus new files under 1 MB, caches left
out), and each trace records `committed_bytes`, `worktree_bytes` and `committed_all` (1 when the commit held all of
it). The official score is the reward when `committed_all` is 1, else 0.
"""
import os
import re
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborEnv, HarborTask, HarborTaskset

from deep_swe.taskset import DeepSWEConfig, DeepSWETask

NAMESPACE = os.environ.get("DEEPSWE_IMAGE_NS", "prime/riel-stamand")

_CAPTURE = re.compile(r"^git diff --binary (\w+) HEAD > /logs/artifacts/model\.patch.*$", re.M)
_WORKTREE = """git diff --binary --src-prefix=a/ --dst-prefix=b/ {base} HEAD > /logs/artifacts/committed.patch 2>/dev/null || true
git add -u >/dev/null 2>&1 || true
: > /tmp/untracked.lst
git ls-files -z -o --exclude-standard | while IFS= read -r -d '' f; do
  case "/$f" in */node_modules/*|*/.gocache/*|*/__pycache__/*|*/.venv/*|*/target/*|*/.cache/*|*/dist/*) continue ;; esac
  [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -le 1048576 ] && printf '%s\\0' "$f" >> /tmp/untracked.lst
done
[ -s /tmp/untracked.lst ] && git add --pathspec-from-file=/tmp/untracked.lst --pathspec-file-nul >/dev/null 2>&1
git diff --binary --src-prefix=a/ --dst-prefix=b/ --cached {base} > /logs/artifacts/model.patch 2>/dev/null || true"""


def worktree_capture(script: str) -> str:
    """pre_artifacts.sh with its committed-only capture swapped for one of every change (the committed diff kept)."""
    out, n = _CAPTURE.subn(lambda m: _WORKTREE.format(base=m.group(1)), script)
    if n != 1:
        raise RuntimeError("pre_artifacts.sh: expected one `git diff --binary <base> HEAD > model.patch` line")
    return out


class DeepSWEUpstreamTask(DeepSWETask):
    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        script = worktree_capture((Path(self.data.task_dir) / "pre_artifacts.sh").read_text())
        await runtime.write("/tmp/pre_artifacts.sh", script.encode())
        result = await runtime.run(["bash", "/tmp/pre_artifacts.sh"], {})
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"artifact capture failed ({self.data.name}): {detail}")
        sizes = await runtime.run(["bash", "-c", "cd /logs/artifacts && wc -c < committed.patch && wc -c < model.patch "
                                   "&& { cmp -s committed.patch model.patch && echo 1 || echo 0; }"], {})
        committed, worktree, same = (int(x) for x in sizes.stdout.split())
        trace.record_metrics({"committed_bytes": committed, "worktree_bytes": worktree, "committed_all": same})
        await HarborTask.finalize(self, trace, runtime)


class DeepSWEUpstreamTaskset(HarborTaskset, vf.Taskset[DeepSWETask, DeepSWEConfig]):
    def load(self) -> Iterator[DeepSWETask]:
        for task in HarborTaskset.load(self):
            if task.data.verifier is None:
                raise RuntimeError("DeepSWE v1.1 requires Harbor separate-verifier support")
            data = task.data.model_copy(update={"image": f"{NAMESPACE}/{task.data.image.rsplit('/', 1)[-1]}"})
            yield DeepSWEUpstreamTask(data, task.config)


__all__ = ["DeepSWEUpstreamTaskset", "HarborEnv"]
