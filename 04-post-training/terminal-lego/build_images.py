"""DEAD END (2026-10-05): Prime caps a personal account at 10 images (HTTP 429 "Image limit exceeded"), so per-task
images can't be built there; make_taskset.py builds them in the local Docker instead. Kept for the record.

Build Terminal-Lego task images on Prime from each task's environment/ (Dockerfile + task_file), private to our team.

PrimeIntellect's own prebuilt images (terminal-lego/<id>:latest) aren't pullable by our account, and only 47 of the
hamishi740 Docker Hub images exist, so we build the ones we use. Done builds are recorded in results/images.jsonl
(task id -> image ref), which make_taskset.py writes into each task's [environment].docker_image.

Usage: python build_images.py TASK_ID [TASK_ID ...] | --file ids.txt   [--parallel 16]
"""

import argparse
import asyncio
import io
import json
import tarfile
import time
from pathlib import Path

import httpx
from prime_sandboxes.core import AsyncAPIClient
from prime_sandboxes.images import AsyncImageClient
from prime_sandboxes.models import BuildImageRequest, ImageVisibility

HERE = Path(__file__).parent
TASKS = HERE / "data" / "tasks"
OUT = HERE / "results" / "images.jsonl"
TAG = "v1"


def context_tar(task: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        env = TASKS / f"terminal_lego__{task}" / "environment"
        for item in sorted(env.iterdir()):
            tar.add(item, arcname=item.name)
    return buf.getvalue()


async def build(images: AsyncImageClient, http: httpx.AsyncClient, task: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.time()
        r = await images.initiate_build(BuildImageRequest(image_name=f"tl-{task.replace('_', '-')}", image_tag=TAG,
                                                          visibility=ImageVisibility.PRIVATE))
        put = await http.put(r.upload_url, content=context_tar(task), headers={"Content-Type": "application/octet-stream"})
        put.raise_for_status()
        await images.start_build(r.build_id)
        while True:
            await asyncio.sleep(10)
            s = await images.get_build_status(r.build_id)
            status = str(s.get("status", "")).upper()
            if status in ("COMPLETED", "FAILED", "CANCELLED", "ERROR"):
                break
        rec = {"task": task, "status": status, "image": r.full_image_path, "build_id": r.build_id,
               "seconds": round(time.time() - t0), "error": s.get("error_message") or s.get("errorMessage")}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()
    tasks = list(a.tasks) + (Path(a.file).read_text().split() if a.file else [])
    recs = [json.loads(line) for line in OUT.open()] if OUT.exists() else []
    done = {r["task"] for r in recs if r["status"] == "COMPLETED"}
    tasks = [t for t in tasks if t not in done]
    images = AsyncImageClient(AsyncAPIClient())
    sem = asyncio.Semaphore(a.parallel)
    async with httpx.AsyncClient(timeout=120) as http:
        results = await asyncio.gather(*(build(images, http, t, sem) for t in tasks), return_exceptions=True)
    for t, r in zip(tasks, results):
        if isinstance(r, Exception):
            print(t, "exception:", repr(r))
    print({s: sum(1 for r in results if isinstance(r, dict) and r["status"] == s) for s in ("COMPLETED", "FAILED")},
          "exceptions", sum(isinstance(r, Exception) for r in results))


if __name__ == "__main__":
    asyncio.run(main())
