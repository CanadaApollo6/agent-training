"""Pick Terminal-Lego tasks, build their images with the local Docker, and lay them out as a Harbor taskset that
verifiers' harbor loader reads from its cache (no download): ~/.cache/harbor/terminal-lego_<name>/<task_id>/, each
task.toml given [environment].docker_image = the local tag. The env module ../../06-agents/harness-evals/envs/terminal_lego
points --env.taskset.dataset terminal-lego/<name> at it; run with --env.agent.runtime.type docker.

Eligible: medium or hard, not dropped for TB2 contamination (results/exclude_tasks.txt) or by Prime's quality filter
(oracle failures, no-op passes, build timeouts: data/prime-data-excluded-tasks.jsonl).

Usage: python make_taskset.py NAME [--medium 36] [--hard 12] [--seed 0] [--parallel 8]
"""

import argparse
import collections
import json
import random
import re
import shutil
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
TASKS = HERE / "data" / "tasks"
CACHE = Path.home() / ".cache" / "harbor"


def eligible() -> list[dict]:
    excl = {l.split()[0] for l in open(HERE / "results" / "exclude_tasks.txt") if l.strip() and not l.startswith("#")}
    excl |= {json.loads(l)["task_id"] for l in open(HERE / "data" / "prime-data-excluded-tasks.jsonl")}
    rows = []
    for d in sorted(TASKS.iterdir()):
        tid = d.name.split("__")[1]
        if tid in excl or not (d / "task.toml").exists():
            continue
        md = tomllib.load(open(d / "task.toml", "rb"))["metadata"]
        if md["difficulty"] in ("medium", "hard"):
            rows.append({"task": tid, "difficulty": md["difficulty"], "category": md["category"], "dir": d})
    return rows


def pick(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n tasks spread over categories: round-robin over shuffled per-category lists."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, cats = [], sorted(by)
    while len(out) < n and any(by.values()):
        for c in cats:
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return out


def build(r: dict) -> tuple[str, bool, str]:
    tag = f"tl/{r['task']}:v1"
    p = subprocess.run(["docker", "build", "-q", "-t", tag, str(r["dir"] / "environment")], capture_output=True, text=True)
    return tag, p.returncode == 0, p.stderr[-400:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--medium", type=int, default=36)
    ap.add_argument("--hard", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    rows, rng = eligible(), random.Random(a.seed)
    chosen = pick([r for r in rows if r["difficulty"] == "medium"], a.medium, rng) + \
        pick([r for r in rows if r["difficulty"] == "hard"], a.hard, rng)
    print(f"{len(rows)} eligible; picked {len(chosen)}:", dict(collections.Counter(r["category"] for r in chosen)))
    with ThreadPoolExecutor(a.parallel) as ex:
        built = list(ex.map(build, chosen))
    out = CACHE / f"terminal-lego_{a.name}"
    shutil.rmtree(out, ignore_errors=True)
    ok = []
    for r, (tag, good, err) in zip(chosen, built):
        if not good:
            print("build failed:", r["task"], err.strip().splitlines()[-1:] if err.strip() else "")
            continue
        dst = out / r["task"]
        shutil.copytree(r["dir"], dst)
        toml = (dst / "task.toml").read_text()
        toml = re.sub(r"(\[environment\]\n)", rf'\1docker_image = "{tag}"\n', toml, count=1)
        (dst / "task.toml").write_text(toml)
        ok.append(r)
    (HERE / "results" / f"taskset_{a.name}.txt").write_text("".join(f"{r['task']} {r['difficulty']} {r['category']}\n" for r in ok))
    print(f"{len(ok)} tasks in {out}")


if __name__ == "__main__":
    main()
