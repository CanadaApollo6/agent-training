"""Run `tensorfold serve` with the MoE decode path set to one of bench_moe.py's steps, for end-to-end comparisons.

Usage: python serve_kernels.py old|swap|select|new <tensorfold serve arguments>
(old = the one-warp expert kernel and 6 launches; new = TensorFold's defaults as patched)

TF_RANDOM_SEED=1 gives each request without a seed a fresh one (independent eval attempts).
TF_GPU_ONLY_BUDGET=1 sizes the context by GPU memory alone. TensorFold also caps a discrete card by the host's free RAM
less 4 GiB, which on a 32 GB desktop leaves 13 GiB, too little for the 19 GiB MLX 4-bit build. Run it under a memory
cap (systemd-run -p MemoryMax=...) so a loader that does need the RAM gets killed, not the desktop.
"""

import os
import sys

import torch

from tensorfold import cli
from tensorfold.cuda import capacity, experts

if os.environ.get("TF_GPU_ONLY_BUDGET") == "1":
    def gpu_only(torch_mod) -> int:
        free, _ = map(int, torch_mod.cuda.mem_get_info())
        return max(0, free - int(float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "1")) * capacity.GIB))

    capacity.available_bytes = gpu_only
    print(f"[serve_kernels] GPU-only budget: {gpu_only(torch) / capacity.GIB:.1f} GiB", flush=True)

if os.environ.get("TF_RANDOM_SEED") == "1":             # requests without a seed get a fresh one, as llama-server
    import secrets                                       # does; TensorFold's default draws it from the prompt, so
                                                         # repeated attempts of an eval would replay each other
    from tensorfold.engine import exact_sampling

    exact_sampling.seed_for = lambda *a, **k: secrets.randbits(62)
    print("[serve_kernels] a fresh sampling seed per request", flush=True)

STEPS = {"old": (-1, False, 0), "swap": (0, False, 0), "select": (0, True, 0), "new": None}

step = sys.argv.pop(1)
if STEPS[step] is not None:
    experts.SPLIT, experts.SELECT, experts.COMBINE = STEPS[step]
print(f"[serve_kernels] {step}: SPLIT {experts.SPLIT} SELECT {experts.SELECT} COMBINE {experts.COMBINE}", flush=True)
sys.argv[0] = "tensorfold"
sys.exit(cli.main())
