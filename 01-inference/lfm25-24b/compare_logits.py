"""KL(ref || test) and top-1 agreement of two [T, vocab] logit dumps (.npy fp16 from ref.py, or raw bf16 .bin).

  python compare_logits.py ref.npy test.npy [--skip N]     # --skip: ignore the first N positions
"""

import argparse
import sys

import numpy as np
import torch


def load(path, vocab=128000):
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path).astype(np.float32))
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).reshape(-1, vocab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()
    r, t = load(a.ref), load(a.test)
    n = min(len(r), len(t))
    if len(r) != len(t):
        print(f"lengths differ: {len(r)} vs {len(t)}, comparing {n}", file=sys.stderr)
    r, t = r[a.skip : n], t[a.skip : n]
    lr, lt = r.log_softmax(-1), t.log_softmax(-1)
    kl = (lr.exp() * (lr - lt)).sum(-1)
    agree = (r.argmax(-1) == t.argmax(-1)).float()
    print(f"positions {len(r)}  KL mean {kl.mean():.5f}  p50 {kl.median():.5f}  max {kl.max():.4f}  top-1 agree {agree.mean():.4f}")


if __name__ == "__main__":
    main()
