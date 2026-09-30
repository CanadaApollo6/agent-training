"""primeintellect/deep-swe, with each task's image pointed at our own copy on Prime.

The hub taskset rewrites every image to Prime's team registry (prime/prime/...), which other accounts can't read
(HTTP 403). The upstream images are public on AWS ECR, but the Prime runtime only auto-builds VM images from Docker Hub
refs, so images.jsonl + `prime images push-bulk --manifest images.jsonl --private` copies them into our registry once.
Tasks and verifiers are unchanged.
"""
import os
from collections.abc import Iterator

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborEnv, HarborTaskset

from deep_swe.taskset import DeepSWEConfig, DeepSWETask

NAMESPACE = os.environ.get("DEEPSWE_IMAGE_NS", "prime/riel-stamand")


class DeepSWEUpstreamTaskset(HarborTaskset, vf.Taskset[DeepSWETask, DeepSWEConfig]):
    def load(self) -> Iterator[DeepSWETask]:
        for task in HarborTaskset.load(self):
            if task.data.verifier is None:
                raise RuntimeError("DeepSWE v1.1 requires Harbor separate-verifier support")
            data = task.data.model_copy(update={"image": f"{NAMESPACE}/{task.data.image.rsplit('/', 1)[-1]}"})
            yield DeepSWETask(data, task.config)


__all__ = ["DeepSWEUpstreamTaskset", "HarborEnv"]
