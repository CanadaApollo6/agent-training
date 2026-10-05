"""terminal-lego: Terminal-Lego-15k tasks as a Harbor taskset, read from ~/.cache/harbor/terminal-lego_<name>/ (laid
out with locally built images by 04-post-training/terminal-lego/make_taskset.py). Select one with
--env.taskset.dataset terminal-lego/<name>, put this directory's parent on PYTHONPATH, and run with
--env.agent.runtime.type docker (the images exist only in the local Docker)."""

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalLegoConfig(HarborConfig):
    dataset: str = "terminal-lego/pilot"


class TerminalLegoTaskset(HarborTaskset, vf.Taskset[HarborTask, TerminalLegoConfig]):
    pass


__all__ = ["TerminalLegoTaskset"]
