"""Token rows for the EXL3 quantization test, from Nanbeige's own outputs (drafter/data/live).

Writes three safetensors files, each holding `input_ids` [rows, 2048]:
  kl-gen.safetensors    100 rows, one window per chat/math/code answer (thinking included)
  kl-agent.safetensors   40 rows, two windows per pi-harness trajectory (20 trajectories)
  cal-nb.safetensors    250 rows for `convert.py -cd`: 200 answers + 50 windows from the other 37 trajectories
The KL rows and the calibration rows come from different conversations, so a Nanbeige-calibrated quant is not
scored on text it was calibrated on. Conversations are rendered with the model's own chat template.
Every row starts with <|im_start|>, the model's BOS: a window cut from mid-trajectory gets it put in front (its last
token drops). Without it the model has no first-token anchor and KL vs bf16 is 8-9x higher (README, quantization test).
The first run (2026-10-27) built its agent and calibration windows without that token; `bos_rows.py` patched the KL
rows after the fact, which gives exactly what this version writes for kl-agent.

    python build_rows.py MODEL_DIR GEN_JSONL TRAJ_JSONL OUT_DIR
"""
import json
import random
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

LEN = 2048
BOS = 166100  # <|im_start|>


def render(tok, msgs):
    for m in msgs:  # the template iterates tool-call arguments as a mapping
        for c in m.get("tool_calls") or []:
            if isinstance(c["function"]["arguments"], str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    return tok.apply_chat_template(msgs, tokenize=False)


def windows(ids, n, rng):
    """Up to n non-overlapping LEN-token windows from one conversation, at random offsets."""
    if len(ids) < LEN:
        return []
    spans = len(ids) // LEN
    picks = sorted(rng.sample(range(spans), min(n, spans)))
    slack = len(ids) - spans * LEN
    off = rng.randint(0, slack)
    return [[BOS] + ids[off + p * LEN: off + (p + 1) * LEN - 1] for p in picks]


def packed(tok, convs, rows):
    """Concatenate short conversations into LEN-token rows; each long one gives one window."""
    out, buf = [], []
    for msgs in convs:
        ids = tok(render(tok, msgs), add_special_tokens=False)["input_ids"]
        if len(ids) >= LEN:
            out.append(ids[:LEN])
        else:
            buf += ids
            if len(buf) >= LEN:
                out.append(buf[:LEN])
                buf = []
        if len(out) == rows:
            return out
    raise SystemExit(f"only {len(out)} of {rows} rows")


def save(rows, path):
    t = torch.tensor(rows, dtype=torch.long)
    save_file({"input_ids": t}, path)
    print(path, tuple(t.shape))


def main(model_dir, gen_path, traj_path, out_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    rng = random.Random(0)

    gen = [json.loads(line)["conversations"] for line in open(gen_path)]
    rng.shuffle(gen)
    save(packed(tok, gen[:400], 100), f"{out_dir}/kl-gen.safetensors")

    # Trajectory files hold several prefixes of each run (id "<run>-<k>"); keep the longest per run.
    runs = {}
    for line in open(traj_path):
        r = json.loads(line)
        base = r["id"].rsplit("-", 1)[0]
        if base not in runs or len(line) > runs[base][0]:
            runs[base] = (len(line), r["conversations"])
    keys = sorted(runs)
    rng.shuffle(keys)
    kl_runs, cal_runs = keys[:20], keys[20:]
    agent = []
    for k in kl_runs:
        agent += windows(tok(render(tok, runs[k][1]), add_special_tokens=False)["input_ids"], 2, rng)
    save(agent[:40], f"{out_dir}/kl-agent.safetensors")

    cal = packed(tok, gen[400:1400], 200)
    cal_agent = []
    for k in cal_runs:
        cal_agent += windows(tok(render(tok, runs[k][1]), add_special_tokens=False)["input_ids"], 2, rng)
    rng.shuffle(cal_agent)
    save(cal + cal_agent[:50], f"{out_dir}/cal-nb.safetensors")


if __name__ == "__main__":
    main(*sys.argv[1:])
