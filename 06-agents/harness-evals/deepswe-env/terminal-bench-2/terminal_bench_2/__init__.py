"""terminal-bench-2: the hub's terminal-bench-2 taskset (harbor, pinned to terminal-bench/terminal-bench-2), copied here
so it runs on this project's newer verifiers, which has the claude_code harness. Same tasks and grading as the
primeintellect/terminal-bench-2 runs in ../.
"""
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalBench2Config(HarborConfig):
    dataset: Literal["terminal-bench/terminal-bench-2"] = "terminal-bench/terminal-bench-2"


class TerminalBench2Taskset(HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config]):
    pass


__all__ = ["TerminalBench2Taskset"]
