"""Pick the training set from harvest.py's samples and write it pre-tokenized for prime-rl (prime-rl-pretokenized.patch).

    uv run --with tokenizers --with pyarrow select_data.py data/all.jsonl --per-task 2 --out data/train
    uv run --with tokenizers --with pyarrow select_data.py data/round2.jsonl:4 data/teacher.jsonl:all --out data/train-a

Per task, the --per-task solved rollouts with the fewest model-written tokens: the shortest solves, since runs that
end in budget are mostly unproductive, not unlucky. A rollout brings all its samples (its main conversation, any
compaction segments, prime_agent's review-gate calls). Writes <out>/train.parquet (input_ids, loss_mask, and where each
row came from) and prints what was kept. A source written path:K takes its own per-task cap (K or all).
"""
import argparse
import collections
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

MODEL = Path.home() / "models/ornith/r1s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+", help="harvest.py files, each optionally path:K")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("data/train"))
    a = ap.parse_args()
    keep, by_task = [], set()
    for spec in a.samples:
        path, _, k = spec.partition(":")
        cap = None if k == "all" else int(k) if k else a.per_task
        rollouts = collections.defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            rollouts[(r["task"], r["rollout"])].append(r)
        tasks = collections.defaultdict(list)
        for (task, _), samples in rollouts.items():
            tasks[task].append(samples)
        by_task |= set(tasks)
        keep += [s for task in sorted(tasks)
                 for s in sorted(tasks[task], key=lambda ss: sum(x["loss_tokens"] for x in ss))[:cap]]

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    cols = collections.defaultdict(list)
    for samples in keep:
        for s in samples:
            enc = tok.encode(s["text"], add_special_tokens=False)
            mask = [any(lo <= start < hi for lo, hi in s["loss"]) for start, _ in enc.offsets]
            assert sum(mask) == s["loss_tokens"] and len(enc.ids) == s["tokens"]
            cols["input_ids"].append(enc.ids)
            cols["loss_mask"].append(mask)
            for k in ("task", "harness", "run", "rollout", "segment", "tokens", "loss_tokens"):
                cols[k].append(s[k])
    a.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({**cols, "input_ids": pa.array(cols["input_ids"], pa.list_(pa.int32()))}),
                   a.out / "train.parquet")

    n = len(cols["input_ids"])
    print(f"{len(keep)} rollouts over {len(by_task)} tasks -> {n} samples, {sum(cols['tokens'])} tokens, "
          f"{sum(cols['loss_tokens'])} trained on")
    print("rollouts by harness:", dict(collections.Counter(ss[0]["harness"] for ss in keep)))
    # prime-rl fills each pack in data order and starts a new one when the next sample doesn't fit (round 1: 54
    # estimated, ~53 seen); data.batch_size packs make a step
    packs, cur = 0, 0
    for t in (min(t, 131072) for t in cols["tokens"]):
        packs, cur = (packs + 1, t) if cur + t > 131072 else (packs, cur + t)
    print(f"~{packs + (cur > 0)} packs of 128K per epoch")
    for cap in (32768, 65536, 131072):
        cut = sum(max(0, t - cap) for t in cols["tokens"])
        print(f"  seq_len {cap}: {sum(t <= cap for t in cols['tokens'])}/{n} samples whole, {cut} tokens cut")


if __name__ == "__main__":
    main()
