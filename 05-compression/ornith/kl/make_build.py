"""A new checkpoint from an existing one with some MLX tensors replaced (e.g. export_always_on.py's output).

Files without replaced tensors are hard-linked, so the new build costs only the rewritten shards. Where a replaced
matrix's width differs from what the config says, a per-module quantization override goes in config.json.

    python make_build.py --base ~/models/ornith/r1 --replace ~/models/ornith/always-on/always-on-q4.safetensors \
        --out ~/models/ornith/r1s
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def module_bits(config, module):
    q = config["quantization"]
    return q.get(module, q)["bits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--replace", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, out = Path(args.base).expanduser(), Path(args.out).expanduser()
    assert not out.exists(), out

    new = {}
    for r in args.replace:
        with safe_open(r, "pt") as f:
            new |= {k: f.get_tensor(k) for k in f.keys()}
    where = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [k for k in new if k not in where]
    assert not missing, missing[:5]
    touched = {where[k] for k in new}

    config = json.loads((base / "config.json").read_text())
    overrides = set()
    for k in new:
        if k.endswith(".weight"):
            m = k[:-len(".weight")]
            bits = new[k].shape[-1] * 32 // (new[m + ".scales"].shape[-1] * 64)
            if bits != module_bits(config, m):
                config["quantization"][m] = {"group_size": 64, "bits": bits}
                overrides.add(m)
    if "quantization_config" in config:
        config["quantization_config"] = config["quantization"]

    out.mkdir(parents=True)
    for p in base.iterdir():
        if p.name in touched or p.name == "config.json":
            continue
        if p.is_file():
            os.link(p, out / p.name)
        elif p.is_dir():
            shutil.copytree(p, out / p.name)
    for shard in touched:
        with safe_open(base / shard, "pt") as f:
            tensors, meta = {k: f.get_tensor(k) for k in f.keys()}, f.metadata()
        for k, v in new.items():
            if k in tensors:
                if v.dtype != tensors[k].dtype and v.element_size() == tensors[k].element_size() == 4:
                    v = v.view(tensors[k].dtype)             # packed words: int32 vs uint32, same bits
                assert v.dtype == tensors[k].dtype, k
                assert v.shape == tensors[k].shape or k.rsplit(".", 1)[0] in overrides, (k, v.shape, tensors[k].shape)
                tensors[k] = v
        save_file(tensors, out / shard, metadata=meta)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{len(new) // 3} matrices replaced in {sorted(touched)}; {len(overrides)} bit overrides; -> {out}")


if __name__ == "__main__":
    main()
