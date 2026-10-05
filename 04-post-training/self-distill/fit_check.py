"""Did round 2's arms fit their training data? Log-probs of the trained-on tokens under base Ornith, R1s-SD's bf16
weights and both arms' tuned weights.

If arm A's log-prob on the teacher solves barely moved from R1s-SD's, the training didn't take (too weak). If it rose a
lot and the arm still can't solve those tasks, the model fit the traces but can't replay them over 60 turns (a long
rollout leaves the recorded states within a few steps).

    uv run --with tokenizers --with pyarrow fit_check.py build         # locally: data/fitcheck.parquet
    python fit_check.py score --model DIR --name NAME --out FILE       # with vLLM: one record per sample

Sets, all tokenized as trained (loss mask = the model-written tokens):
- teacher_raw: arm A's teacher rows (raw teacher reasoning), teacher_rw: arm B's (Ornith's rewritten reasoning)
- own_trained: 80 of the own-solve rows both arms trained on, own_unseen: 80 own-solve rows from the same harvest
  that weren't selected (the control: a model that only shifted overall moves both own sets alike)
Each record splits the trained tokens into reasoning (before </think>) and tail (</think>, answer, tool calls).
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = Path.home() / "models/ornith/r1s/tokenizer.json"


def build():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    teacher = lambda r: not r["run"].split("--")[1].startswith("ornith")
    a = pq.read_table(HERE / "data/train-a/train.parquet").to_pylist()
    b = pq.read_table(HERE / "data/train-b/train.parquet").to_pylist()
    rng = random.Random(0)
    rows = [{**r, "set": "teacher_raw"} for r in a if teacher(r)]
    rows += [{**r, "set": "teacher_rw"} for r in b if teacher(r)]
    own = [r for r in a if not teacher(r)]
    rows += [{**r, "set": "own_trained"} for r in rng.sample(own, 80)]
    seen = {r["rollout"] for r in a}
    tok = Tokenizer.from_file(str(TOKENIZER))
    unseen = [s for s in map(json.loads, open(HERE / "data/round2.jsonl")) if s["rollout"] not in seen]
    for s in rng.sample(unseen, 80):
        enc = tok.encode(s["text"], add_special_tokens=False)
        mask = [any(lo <= st < hi for lo, hi in s["loss"]) for st, _ in enc.offsets]
        rows.append({"input_ids": enc.ids, "loss_mask": mask, "set": "own_unseen",
                     **{k: s[k] for k in ("task", "harness", "run", "rollout", "segment", "tokens", "loss_tokens")}})
    cols = {k: [r[k] for r in rows] for k in ("input_ids", "loss_mask", "set", "task", "run", "rollout", "segment",
                                               "tokens", "loss_tokens")}
    pq.write_table(pa.table({**cols, "input_ids": pa.array(cols["input_ids"], pa.list_(pa.int32()))}),
                   HERE / "data/fitcheck.parquet")
    for s in ("teacher_raw", "teacher_rw", "own_trained", "own_unseen"):
        sel = [r for r in rows if r["set"] == s]
        print(f"{s:12} {len(sel):4} samples {sum(r['tokens'] for r in sel) / 1e6:6.2f}M tokens "
              f"{sum(r['loss_tokens'] for r in sel) / 1e6:5.2f}M trained on")


def score(a):
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    rows = pq.read_table(a.data).to_pylist()
    llm = LLM(model=a.model, max_model_len=262144, gpu_memory_utilization=0.9, enable_prefix_caching=False,
              max_logprobs=1, language_model_only=True, trust_remote_code=True, max_num_batched_tokens=8192,
              max_num_seqs=16)
    end = llm.get_tokenizer().convert_tokens_to_ids("</think>")
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["input_ids"]) for r in rows], sp)
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            ids, mask, plp = r["input_ids"], r["loss_mask"], o.prompt_logprobs
            assert len(plp) == len(ids)
            rec = {"model": a.name, **{k: r[k] for k in ("set", "task", "run", "rollout", "segment")},
                   "lp_reason": 0.0, "n_reason": 0, "lp_tail": 0.0, "n_tail": 0}
            in_tail = False
            for j in range(1, len(ids)):
                if not mask[j]:
                    in_tail = False         # a new trained span starts in reasoning
                    continue
                if ids[j] == end:
                    in_tail = True
                part = "tail" if in_tail else "reason"
                rec[f"lp_{part}"] += plp[j][ids[j]].logprob
                rec[f"n_{part}"] += 1
            f.write(json.dumps(rec) + "\n")
    print(f"{a.name}: scored {len(rows)} samples -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--data", default=str(HERE / "data/fitcheck.parquet"))
    ap.add_argument("--out")
    a = ap.parse_args()
    build() if a.mode == "build" else score(a)


if __name__ == "__main__":
    sys.exit(main())
