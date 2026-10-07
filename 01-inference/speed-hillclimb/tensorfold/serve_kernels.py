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

if os.environ.get("TF_BUDGET_GIB"):                     # overrides the planner's budget: its bounds overcount a 24 GB
    budget = int(float(os.environ["TF_BUDGET_GIB"]) * capacity.GIB)   # card (Qwen3.8-27B: 3.7 GiB of fixed scratch and
    capacity.available_bytes = lambda torch_mod: budget                # 4x the bf16 KV a token holds), so pass --context
    print(f"[serve_kernels] planner budget forced to {budget / capacity.GIB:.1f} GiB: measure the real peak", flush=True)

if os.environ.get("TF_RANDOM_SEED") == "1":             # requests without a seed get a fresh one, as llama-server
    import secrets                                       # does; TensorFold's default draws it from the prompt, so
                                                         # repeated attempts of an eval would replay each other
    from tensorfold.engine import exact_sampling

    exact_sampling.seed_for = lambda *a, **k: secrets.randbits(62)
    print("[serve_kernels] a fresh sampling seed per request", flush=True)

if os.environ.get("TF_DRAFT_PACKED") == "1":            # the planner counts a DFlash2 drafter at 4 bytes a weight,
    import dataclasses, json, math                       # but dflash2.py packs it to 4-bit (groups of 64, bf16 scale
    from pathlib import Path                             # and bias: 4.5 bits) one tensor at a time from the host;
    whole = capacity.estimate_weights                    # 7.2 GiB counted for Qwen3.8-27B's drafter, about 1 GiB held

    def packed(model_dir, transform, **kw):
        if "dflash_config" not in json.loads((Path(model_dir) / "config.json").read_text()):
            return whole(model_dir, transform, **kw)
        w = whole(model_dir, lambda name, info: (math.prod(info["shape"]) * (4.5 / 8 if len(info["shape"]) == 2
                                                 else 4), 0), **kw)
        # packing one weight on the device: bf16 copy, fp32 groups, int32 codes, words (~14 bytes a weight)
        return dataclasses.replace(w, staging=int(w.staging / 3 / (4.5 / 8) * 14))

    capacity.estimate_weights = packed
    print("[serve_kernels] drafter counted at its packed 4-bit size", flush=True)

STEPS = {"old": (-1, False, 0), "swap": (0, False, 0), "select": (0, True, 0), "new": None}

step = sys.argv.pop(1)
if STEPS[step] is not None:
    experts.SPLIT, experts.SELECT, experts.COMBINE = STEPS[step]
print(f"[serve_kernels] {step}: SPLIT {experts.SPLIT} SELECT {experts.SELECT} COMBINE {experts.COMBINE}", flush=True)
sys.argv[0] = "tensorfold"
sys.exit(cli.main())
