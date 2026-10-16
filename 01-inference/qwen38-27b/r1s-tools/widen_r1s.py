"""R1s with its 3-bit routed experts rewritten as MLX 4-bit words (same codes, scales, biases): the 4-bit path's reference.

usage: widen_r1s.py <r1s dir> <out dir>
Everything but the expert shards and config.json is symlinked; config drops the switch_mlp overrides (the default 4).
"""
import json
import os
import struct
import sys

import numpy as np


def widen(words: np.ndarray) -> np.ndarray:
    """MLX 3-bit rows (an LSB-first bitstream, 3 bytes a run of 8 codes) to MLX 4-bit words, eight codes a word."""
    b = words.view(np.uint8).reshape(*words.shape[:-1], -1, 3).astype(np.uint32)
    v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    out = np.zeros(v.shape, np.uint32)
    for j in range(8):
        out |= ((v >> (3 * j)) & 7) << (4 * j)
    return out


def header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


def shard(src, dst):
    f, h, base = header(src)
    meta = h.pop("__metadata__", None)
    names = sorted(h, key=lambda k: h[k]["data_offsets"][0])
    out_h, plan, at = {}, [], 0
    for k in names:
        t = h[k]
        s, e = t["data_offsets"]
        shape = list(t["shape"])
        grow = ".switch_mlp." in k and k.endswith(".weight")
        if grow:
            assert t["dtype"] == "U32"
            shape[-1] = shape[-1] * 32 // 3 // 8
        size = int(np.prod(shape)) * 4 if grow else e - s
        out_h[k] = {"dtype": t["dtype"], "shape": shape, "data_offsets": [at, at + size]}
        plan.append((k, s, e, grow, t["shape"]))
        at += size
    if meta is not None:
        out_h["__metadata__"] = meta
    hb = json.dumps(out_h, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as o:
        o.write(struct.pack("<Q", len(hb)))
        o.write(hb)
        for k, s, e, grow, shape in plan:
            f.seek(base + s)
            if not grow:
                left = e - s
                while left:
                    buf = f.read(min(left, 1 << 26))
                    o.write(buf)
                    left -= len(buf)
                continue
            per = int(np.prod(shape[1:])) * 4  # an expert's bytes
            for _ in range(shape[0]):
                w = np.frombuffer(f.read(per), np.uint32).reshape(shape[1:])
                o.write(widen(w).tobytes())
    print(dst, flush=True)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        q = os.path.join(dst, name)
        if name.startswith("model-experts-"):
            shard(p, q)
        elif name == "config.json":
            c = json.load(open(p))
            for key in ("quantization", "quantization_config"):
                if key in c:
                    c[key] = {k: v for k, v in c[key].items() if ".switch_mlp." not in k}
            json.dump(c, open(q, "w"), indent=2)
        elif not os.path.exists(q):
            os.symlink(os.path.realpath(p), q)


if __name__ == "__main__":
    # self-check: eight codes 0..7 packed 3-bit read back in order
    codes = np.arange(8, dtype=np.uint32)
    v = sum(int(c) << (3 * i) for i, c in enumerate(codes))
    w = np.frombuffer(np.array([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, 0], np.uint8)[:3].tobytes() * 4, np.uint32)
    assert all(((widen(w.reshape(1, 3))[0, 0] >> (4 * i)) & 15) == i for i in range(8))
    main()
