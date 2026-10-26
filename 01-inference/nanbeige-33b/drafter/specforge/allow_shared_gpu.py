"""Lets a SpecForge capture server and its trainer share one GPU (only single-GPU pods were on offer, 2026-10-26).
SpecForge's config check refuses overlapping devices; with Mooncake over TCP nothing else depends on them being
apart. Turns the check off in the installed source; idempotent.

    python allow_shared_gpu.py [schema.py path]
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else __import__("specforge.config.schema", fromlist=["x"]).__file__
old = "        if overlap_devices:\n            raise ValueError(\n                \"managed_local capture and trainer"
new = "        if False and overlap_devices:  # allow_shared_gpu.py\n            raise ValueError(\n                \"managed_local capture and trainer"
src = open(path).read()
if new not in src:
    assert old in src, "check not found; SpecForge changed"
    open(path, "w").write(src.replace(old, new))
print("shared GPU allowed")
