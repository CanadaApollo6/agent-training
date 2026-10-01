"""Check prime-rl's bf16 export against the original checkpoint and put back what training dropped.

    python finish_export.py --orig ORIGINAL_BF16 --export EXPORTED_BF16

prime-rl drops the MTP head (mtp.*) on load, so the export lacks it; the original's goes back in unchanged, since
training never touched it. The export must otherwise hold exactly the original's tensors (names, shapes, dtypes), so
the R1s rebuild reads it like the original. The vision tower and the routers were frozen and must be bit-identical.
Prints how far each kind of trained tensor moved (relative RMS change), then copies the original's config, chat template
and tokenizer files over the export's.
"""
import argparse
import collections
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FROZEN = (r"^model\.visual\.", r"\.mlp\.gate\.weight$")
ASSETS = ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
          "vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--export", required=True)
    a = ap.parse_args()
    orig, exp = Path(a.orig), Path(a.export)
    ow = json.loads((orig / "model.safetensors.index.json").read_text())["weight_map"]
    ew_path = exp / "model.safetensors.index.json"
    index = json.loads(ew_path.read_text())
    ew = index["weight_map"]

    mtp = sorted(k for k in ow if k.startswith("mtp."))
    missing = sorted(set(ow) - set(ew) - set(mtp))
    extra = sorted(set(ew) - set(ow))
    assert not missing and not extra, f"missing {missing[:5]}, extra {extra[:5]}"

    moved, problems = collections.defaultdict(list), []
    for k in sorted(ew):
        with safe_open(orig / ow[k], "pt") as f:
            t0 = f.get_tensor(k)
        with safe_open(exp / ew[k], "pt") as f:
            t1 = f.get_tensor(k)
        if t0.shape != t1.shape or t0.dtype != t1.dtype:
            problems.append(f"{k}: {tuple(t0.shape)} {t0.dtype} -> {tuple(t1.shape)} {t1.dtype}")
            continue
        if any(re.search(p, k) for p in FROZEN):
            if not torch.equal(t0, t1):
                problems.append(f"{k}: frozen but changed")
            continue
        d = (t1.float() - t0.float()).pow(2).mean().sqrt() / t0.float().pow(2).mean().sqrt().clamp_min(1e-12)
        moved[re.sub(r"\.\d+\.", ".N.", k.removeprefix("model.language_model."))].append(d.item())
    for p in problems:
        print("PROBLEM", p)
    assert not problems, f"{len(problems)} problems"
    print("relative RMS change per tensor kind (mean, max):")
    for kind, ds in sorted(moved.items()):
        print(f"  {sum(ds) / len(ds):.2e} {max(ds):.2e}  {kind}")

    tensors = {}
    for k in mtp:
        with safe_open(orig / ow[k], "pt") as f:
            tensors[k] = f.get_tensor(k)
    save_file(tensors, exp / "model-mtp.safetensors", metadata={"format": "pt"})
    ew.update({k: "model-mtp.safetensors" for k in mtp})
    ew_path.write_text(json.dumps(index, indent=2))
    for name in ASSETS:
        if (orig / name).exists():
            shutil.copy(orig / name, exp / name)
    print(f"export checked; {len(mtp)} MTP tensors restored")


if __name__ == "__main__":
    main()
