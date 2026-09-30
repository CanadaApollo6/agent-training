"""terminal-bench-4: the harbor taskset pinned to Terminal-Bench 4.0.0, same shape as the hub's terminal-bench-2.

Each task declares Docker Hub images for the solver and a separate verifier (harborframework/terminal-bench:...), which
the Prime runtime auto-builds on first use. 11 tasks are multi-container and 3 need GPUs; ../tb4_tasks.txt samples
from the other 52.
"""
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalBench4Config(HarborConfig):
    dataset: Literal["terminal-bench/terminal-bench@4.0.0"] = "terminal-bench/terminal-bench@4.0.0"


class TerminalBench4Taskset(HarborTaskset, vf.Taskset[HarborTask, TerminalBench4Config]):
    pass


__all__ = ["TerminalBench4Taskset"]
