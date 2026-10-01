"""Does prime-rl's renderer (the `renderers` package, config qwen3.6 with thinking preserved) tokenize our harvested
samples exactly as TensorFold served them, with the loss on the same tokens?

    ~/.cache/sd-renderers/.venv/bin/python parity.py data/all.jsonl

For each sample: our ids are the TensorFold render (harvest.py's text) tokenized; theirs come from
renderers.build_training_sample on the sample's messages and tools. Reports token and loss-mask agreement and the first
divergence of any sample that differs.
"""
import json
import sys
from pathlib import Path

from renderers import create_renderer
from renderers.base import build_training_sample
from renderers.configs import Qwen36RendererConfig
from transformers import AutoTokenizer

MODEL = Path.home() / "models/ornith/r1s"


def main(path: str):
    tok = AutoTokenizer.from_pretrained(MODEL)
    renderer = create_renderer(tok, Qwen36RendererConfig(enable_thinking=True, preserve_thinking=True))
    same, stats = 0, {"opener": 0}
    rows = [json.loads(line) for line in open(path)]
    for row in rows:
        enc = tok(row["text"], add_special_tokens=False, return_offsets_mapping=True)
        ours = enc["input_ids"]
        our_mask = [any(lo <= a < hi for lo, hi in row["loss"]) for a, _ in enc["offset_mapping"]]
        theirs = build_training_sample(renderer, row["messages"], tools=row["tools"])
        ids, mask = list(theirs.token_ids), list(theirs.loss_mask)
        n = len(ours)
        # ours ends at the last <|im_end|>; theirs may add the template's trailing newline
        ok_ids = ids[:n] == ours and all(not m for m in mask[n:])
        # the reply opener the server puts in the prompt (<|im_start|>assistant\n<think>\n): prime-rl trains on it,
        # the model never samples it; any other mask difference is a real one
        diff = [i for i in range(min(n, len(mask))) if mask[i] != our_mask[i]]
        opener = [i for i in diff if not our_mask[i] and mask[i] and i + 3 < n and
                  any(our_mask[j] for j in range(i + 1, min(n, i + 6))) and
                  tok.decode(ours[i:next(j for j in range(i + 1, n) if our_mask[j])]).endswith("<think>\n")
                  and "<|im_start|>assistant" in tok.decode(ours[max(0, i - 4):next(j for j in range(i + 1, n) if our_mask[j])])]
        stats["opener"] += len(opener)
        ok_mask = len(diff) == len(opener)
        if ok_ids and ok_mask:
            same += 1
            continue
        k = next((i for i in range(min(n, len(ids))) if ids[i] != ours[i] or mask[i] != our_mask[i]), min(n, len(ids)))
        print(f"DIFF {row['harness']} {row['task']} seg {row['segment']}: ours {n} tokens, theirs {len(ids)}; "
              f"ids {'same' if ok_ids else 'differ'}, mask {'same' if ok_mask else 'differ'}; first at token {k}")
        print("   ours:  ", repr(tok.decode(ours[max(0, k - 8):k + 12])), our_mask[k:k + 3])
        print("   theirs:", repr(tok.decode(ids[max(0, k - 8):k + 12])), mask[k:k + 3])
    print(f"{same}/{len(rows)} samples identical in tokens and in loss mask apart from the reply opener "
          f"({stats['opener']} opener tokens prime-rl would also train on)")


if __name__ == "__main__":
    main(sys.argv[1])
