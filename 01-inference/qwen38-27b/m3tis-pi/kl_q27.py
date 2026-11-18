"""How far each 4-bit build drifts from its BF16 source: per-token KL(BF16 || 4-bit) and top-1 agreement on our own
TB2 conversations (pod/kl.jsonl, from export_texts.py).

Two models stay loaded, each on half the GPUs: R (BF16 reference) and Q (BF16 with the 4-bit build's weights
dequantized in place). Each step swaps weights, so three comparisons need only two loads:

    1. base     vs Vontra 4-bit             the quant error M3TiS #1 already ships with
    2. merged   vs Vontra 4-bit             the adapter's own effect, plus quant error
    3. merged   vs adapted 4-bit (ours)     the quant error of the new build; should match 1

"merged" is base + (alpha/r)·B·A on the 256 LoRA weights, in BF16. KL is reported over all positions and over
assistant positions only (the reply tokens, which is what the model generates), per text set.

    python kl_q27.py --base ~/qwen27 --lora ~/pi-lora --vontra ~/vontra --adapted ~/q27pi-mlx4 \
        --texts ~/pod/kl.jsonl --out ~/out/kl.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from build_mlx4 import dequant


def hf_name(mlx_key):
    """Vontra tensor prefix -> HF parameter name (without .weight)."""
    if mlx_key == "language_model.lm_head":
        return "lm_head"
    return "model.language_model." + mlx_key.removeprefix("language_model.model.")


def quantized(root, only=None):
    """Yield (hf parameter name, dequantized bf16 weight) for every quantized tensor in an MLX checkpoint."""
    for shard in sorted(Path(root).glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if not k.endswith(".scales"):
                    continue
                pre = k.removesuffix(".scales")
                name = hf_name(pre)
                if only is not None and name not in only:
                    continue
                w = dequant(f.get_tensor(pre + ".weight"), f.get_tensor(k), f.get_tensor(pre + ".biases"))
                yield name + ".weight", w.to(torch.bfloat16)


@torch.no_grad()
def load_into(model, items):
    params = dict(model.named_parameters())
    n = 0
    for name, w in items:
        p = params[name]
        if p.shape != w.shape:
            raise SystemExit(f"{name}: model {tuple(p.shape)}, checkpoint {tuple(w.shape)}")
        p.copy_(w.to(p.device))
        n += 1
    return n


@torch.no_grad()
def merge_lora(model, lora_dir):
    cfg = json.loads((Path(lora_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    ab = {}
    with safe_open(Path(lora_dir) / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            mod, which = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
            ab.setdefault(mod, {})[which[0]] = f.get_tensor(k).float()
    params = dict(model.named_parameters())
    for mod, d in ab.items():
        p = params[mod + ".weight"]
        p.copy_((p.float() + scale * (d["B"].to(p.device) @ d["A"].to(p.device))).to(p.dtype))
    return {m + ".weight" for m in ab}


def dict_args(messages):
    """Qwen's template iterates tool-call arguments as a mapping; the exported traces carry OpenAI JSON strings."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [dict(tc) for tc in m["tool_calls"]]}
            for tc in m["tool_calls"]:
                f = tc["function"] = dict(tc.get("function", tc))
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"] or "{}")
                    except json.JSONDecodeError:
                        f["arguments"] = {"raw": f["arguments"]}
        out.append(m)
    return out


def encode(tok, row, max_len):
    ids = tok.apply_chat_template(dict_args(row["messages"]), tools=row["tools"], tokenize=True, add_generation_prompt=False,
                                  enable_thinking=row["thinking"], return_dict=False)
    if hasattr(ids, "keys"):  # some transformers versions return a BatchEncoding regardless
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)[:max_len]
    start, end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    asst_tok = tok.encode("assistant", add_special_tokens=False)
    mask, inside = [], False
    for i, t in enumerate(ids):
        if t == start:
            inside = ids[i + 1:i + 1 + len(asst_tok)] == asst_tok
        mask.append(inside)  # positions whose *next* token is part of an assistant turn (header included)
        if t == end:
            inside = False
    return torch.tensor([ids]), torch.tensor(mask[:-1])


@torch.no_grad()
def compare(r_model, q_model, texts, chunk=1024):
    out = {}
    for name, ids, mask in texts:
        lr = r_model(input_ids=ids.to(r_model.device)).logits[0, :-1]
        lq = q_model(input_ids=ids.to(q_model.device)).logits[0, :-1].to(lr.device)
        kls, agree = [], []
        for i in range(0, lr.shape[0], chunk):
            a = torch.log_softmax(lr[i:i + chunk].float(), -1)
            b = torch.log_softmax(lq[i:i + chunk].float(), -1)
            kls.append((a.exp() * (a - b)).sum(-1).cpu())
            agree.append((a.argmax(-1) == b.argmax(-1)).cpu())
        kl, ag = torch.cat(kls), torch.cat(agree)
        for key, sel in ((name, slice(None)), (name + " (assistant)", mask)):
            o = out.setdefault(key, {"kl": [], "agree": []})
            o["kl"].append(kl[sel])
            o["agree"].append(ag[sel])
        del lr, lq
    rows = {}
    for key, o in out.items():
        kl, ag = torch.cat(o["kl"]), torch.cat(o["agree"])
        rows[key] = {"positions": len(kl), "kl_mean": kl.mean().item(), "kl_p99": kl.quantile(0.99).item(),
                     "top1_agree": ag.float().mean().item()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    for a in ("base", "lora", "vontra", "adapted", "texts", "out"):
        ap.add_argument("--" + a, type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    texts = []
    for line in open(args.texts):
        row = json.loads(line)
        ids, mask = encode(tok, row, args.max_len)
        texts.append((row["set"], ids, mask))
    print(f"{len(texts)} texts, {sum(t[1].shape[1] for t in texts)} tokens", flush=True)

    half = args.gpus // 2

    def load(gpus):
        mem = {g: "46GiB" if g in gpus else 0 for g in range(args.gpus)}
        m = Qwen3_5ForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto",
                                                            max_memory=mem)
        return m.eval()

    t0 = time.time()
    r_model, q_model = load(range(half)), load(range(half, args.gpus))
    print(f"loaded 2 models in {time.time() - t0:.0f} s; Q: {load_into(q_model, quantized(args.vontra))} "
          f"weights from Vontra", flush=True)
    results = {}

    results["1. base vs Vontra 4-bit"] = compare(r_model, q_model, texts)
    print(json.dumps(results, indent=1), flush=True)

    touched = merge_lora(r_model, args.lora)
    results["2. merged vs Vontra 4-bit"] = compare(r_model, q_model, texts)
    print(json.dumps(results, indent=1), flush=True)

    n = load_into(q_model, quantized(args.adapted, only={t.removesuffix(".weight") for t in touched}))
    if n != len(touched):
        raise SystemExit(f"adapted build: replaced {n} of {len(touched)} weights")
    results["3. merged vs adapted 4-bit"] = compare(r_model, q_model, texts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))
    print("KL_DONE")


if __name__ == "__main__":
    main()
