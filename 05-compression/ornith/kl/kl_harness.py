"""KL harness: how far does a quantized Ornith 35B-A3B build drift from bf16, where, and what would fix it?

One bf16 model in memory. A build is simulated by writing the dequantized weights of its quantized tensors over the
bf16 ones:
- always-on path: from the MLX 4-bit checkpoint (4-bit linears, embeddings and LM head; 8-bit routers), the path
  every TensorFold build here uses, or bf16;
- routed experts: re-quantized in-process with quantize_experts.py's imatrix-weighted search, at any width per layer
  and projection (so R1 is 3/3 everywhere, R2 2/3), or MLX's own 4-bit, or bf16.

Scores come from the positions in kl-data.pt where the model wrote the next token itself (its thinking, answers and
tool calls in Q8's recorded agent runs). Per position: KL(bf16 || build) over the full vocabulary, whether the top
token agrees, and the build's log-probability of the token that was actually written.

    python kl_harness.py --bf16 DIR --mlx4 DIR --imatrix FILE --plan builds     # R1, R2, MLX4, mixes
    python kl_harness.py ... --plan layers                                       # per-layer 4-bit upgrades of R1

Results append to results/kl-<plan>.jsonl, one line per build.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "01-inference/speed-hillclimb/tensorfold"))
import quantize_experts as qe  # noqa: E402

LAYERS, PROJS = 40, ("gate_proj", "up_proj", "down_proj")


# -- MLX weights ---------------------------------------------------------------------------------------------------

def unpack(words: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """MLX words [..., k * bits / 32] -> levels [..., k] (an LSB-first bitstream)."""
    by = words.contiguous().view(torch.uint8)
    bit = (by[..., None] >> torch.arange(8, device=by.device, dtype=torch.uint8)) & 1
    bit = bit.reshape(*by.shape[:-1], k, bits).to(torch.int32)
    return (bit << torch.arange(bits, device=by.device, dtype=torch.int32)).sum(-1)


def dequant(words, scales, biases, bits, gs=64, rows=8192):
    """MLX affine: w = s * q + b per group of gs, in fp32 then bf16 (as the kernels see it); chunked over rows."""
    k = scales.shape[-1] * gs
    lead = scales.shape[:-1]
    words, scales, biases = words.reshape(-1, words.shape[-1]), scales.reshape(-1, scales.shape[-1]), biases.reshape(-1, biases.shape[-1])
    out = torch.empty(words.shape[0], k, dtype=torch.bfloat16, device=words.device)
    for i in range(0, words.shape[0], rows):
        q = unpack(words[i:i + rows], bits, k).float().reshape(-1, scales.shape[-1], gs)
        out[i:i + rows] = (scales[i:i + rows].float()[..., None] * q + biases[i:i + rows].float()[..., None]).reshape(-1, k)
    return out.reshape(*lead, k)


class MLX:
    def __init__(self, path: Path):
        self.path = path
        self.map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        q = json.loads((path / "config.json").read_text())["quantization"]
        self.bits = lambda name: (q.get(name) or {}).get("bits", q["bits"])

    def get(self, key):
        with safe_open(self.path / self.map[key], "pt", device="cuda") as f:
            return f.get_tensor(key)

    def weight(self, name):
        """Dequantized bf16 weight of MLX tensor ``name`` (without .weight)."""
        return dequant(self.get(name + ".weight"), self.get(name + ".scales"), self.get(name + ".biases"),
                       self.bits(name))


def fake_rtn(w, bits, rows=8192):
    """MLX-format round-to-nearest of a bf16 matrix [N, K]: (words, scales, biases)."""
    outs = []
    for i in range(0, w.shape[0], rows):
        g = w[i:i + rows].float().reshape(-1, w.shape[1] // 64, 64)
        s, b = qe.rtn(g, bits)
        q = qe.levels(g, s, b, bits).reshape(-1, w.shape[1])
        outs.append((qe.words(q, bits), s[..., 0].bfloat16(), b[..., 0].bfloat16()))
    return tuple(torch.cat(x) for x in zip(*outs))


PARTS = ("embed", "lm_head", "attention", "linear_attention", "shared_expert", "routers")


def part(mlx: str) -> str:
    """Which part of the always-on path an MLX tensor belongs to."""
    if "embed_tokens" in mlx:
        return "embed"
    if mlx.endswith("lm_head"):
        return "lm_head"
    if ".self_attn." in mlx:
        return "attention"
    if ".linear_attn." in mlx:
        return "linear_attention"
    if ".shared_expert." in mlx:
        return "shared_expert"
    assert mlx.endswith((".mlp.gate", ".shared_expert_gate")), mlx
    return "routers"


def hf_name(mlx: str) -> str:
    if mlx == "language_model.lm_head":
        return "lm_head.weight"
    assert mlx.startswith("language_model.model.")
    return "model.language_model." + mlx[len("language_model.model."):] + ".weight"


# -- builds --------------------------------------------------------------------------------------------------------

class Builds:
    """Writes build weights over the bf16 model, keeping the bf16 originals on the CPU to restore."""

    def __init__(self, model, mlx: MLX, bf16_dir: Path, imatrix: str):
        self.model, self.mlx, self.bf16_dir = model, mlx, bf16_dir
        self.params = dict(model.named_parameters())
        self.imp = qe.importance(imatrix)
        self.always_on = sorted({k[:-len(".scales")] for k in mlx.map if k.endswith(".scales") and ".switch_mlp." not in k})
        self.orig = {}                               # hf name -> bf16 CPU copy, taken before the first overwrite
        self.state = {}                              # hf name -> the source now in place ("bf16" when absent)
        self.cache = {}                              # (layer, proj, source) -> (words, s, b) on CPU

    def _put(self, name, value, source):
        p = self.params[name]
        if self.state.get(name, "bf16") == source:
            return
        if name not in self.orig:
            self.orig[name] = p.detach().to("cpu", copy=True).pin_memory()
        with torch.no_grad():
            p.copy_(self.orig[name].to(p.device, non_blocking=True) if source == "bf16" else value)
        self.state[name] = source

    def always_on_path(self, sources):
        """Per part (see PARTS), or one for all: 'bf16', 'mlx4' (the checkpoint's own weights) or 'r<bits>'
        (round-to-nearest of bf16, groups of 64)."""
        for m in self.always_on:
            name = hf_name(m)
            source = sources if isinstance(sources, str) else sources.get(part(m), "bf16")
            if source.startswith("r") and self.state.get(name, "bf16") != source:
                w = (self.orig[name] if name in self.orig else self.params[name].detach()).cuda()
                qw, sc, bi = fake_rtn(w, int(source[1:]))
                self._put(name, dequant(qw, sc, bi, int(source[1:])), source)
            elif not source.startswith("r"):
                self._put(name, None if source == "bf16" else self.mlx.weight(m).to(self.params[name].dtype), source)

    def _expert_q(self, layer, proj, source):
        """(words, scales, biases, bits) for one layer's projection: 'mlx4', 'q<bits>' (the imatrix search) or
        'r<bits>' (plain round-to-nearest)."""
        key = (layer, proj, source)
        if key not in self.cache:
            if source == "mlx4":
                n = qe.BASE_PREFIX.format(layer, proj)
                self.cache[key] = (self.mlx.get(n + ".weight").cpu(), self.mlx.get(n + ".scales").cpu(),
                                   self.mlx.get(n + ".biases").cpu(), self.mlx.bits(n))
            else:
                bits = int(source[1:])
                w = self._bf16_expert(layer, proj)
                imp = None if source.startswith("r") else self.imp[f"blk.{layer}.{qe.IMATRIX[proj]}.weight"]
                (qw, s, b), _ = qe.quantize(w, imp, bits)
                self.cache[key] = (qw, s, b, bits)
        return self.cache[key]

    def _bf16_expert(self, layer, proj):
        name = f"model.language_model.layers.{layer}.mlp.experts." + ("down_proj" if proj == "down_proj" else "gate_up_proj")
        w = self.orig[name] if name in self.orig else self.params[name].detach().cpu()
        if proj == "down_proj":
            return w
        n = w.shape[1] // 2
        return w[:, :n] if proj == "gate_proj" else w[:, n:]

    def experts(self, layer, gate_up, down):
        """Sources per projection: 'bf16', 'mlx4' or 'q2'/'q3'/'q4'."""
        base = f"model.language_model.layers.{layer}.mlp.experts."
        for name, parts in ((base + "gate_up_proj", (("gate_proj", gate_up), ("up_proj", gate_up))),
                            (base + "down_proj", (("down_proj", down),))):
            source = "+".join(s for _, s in parts)
            if self.state.get(name, "bf16") == source:
                continue
            if all(s == "bf16" for _, s in parts):
                self._put(name, None, "bf16")
                continue
            if name not in self.orig:
                self.orig[name] = self.params[name].detach().to("cpu", copy=True).pin_memory()
            pieces = []
            for proj, s in parts:
                if s == "bf16":
                    pieces.append(self._bf16_expert(layer, proj).cuda())
                else:
                    qw, sc, bi, bits = self._expert_q(layer, proj, s)
                    pieces.append(torch.cat([dequant(qw[i:i + 32].cuda().view(torch.int32), sc[i:i + 32].cuda(),
                                                     bi[i:i + 32].cuda(), bits) for i in range(0, qw.shape[0], 32)]))
            self._put(name, torch.cat(pieces, 1) if len(pieces) > 1 else pieces[0], source)

    def apply(self, spec):
        """spec: {"always_on": "bf16"|"mlx4", "experts": [(gate_up, down)] * LAYERS}."""
        self.always_on_path(spec["always_on"])
        for layer, (gu, dn) in enumerate(spec["experts"]):
            self.experts(layer, gu, dn)
        torch.cuda.synchronize()


# -- scoring -------------------------------------------------------------------------------------------------------

@torch.no_grad()
def hidden(model, rows):
    """Final hidden states at the scored positions of every conversation, [N, D] bf16 on the GPU."""
    lm = model.model.language_model
    out = []
    for r in rows:  # noqa: B007
        ids = r["ids"].long().cuda()[None]
        h = lm(input_ids=ids, use_cache=False).last_hidden_state[0]
        out.append(h[r["mask"].cuda()])
    return torch.cat(out)


@torch.no_grad()
def score(ref_h, ref_w, h, w, targets, chunk=2048):
    """KL(ref || build) per position, top-1 agreement, and log-probs of the written tokens under both."""
    kl, agree, lp_ref, lp = [], [], [], []
    for i in range(0, len(h), chunk):
        a = torch.log_softmax((ref_h[i:i + chunk] @ ref_w.T).float(), -1)
        b = torch.log_softmax((h[i:i + chunk] @ w.T).float(), -1)
        kl.append((a.exp() * (a - b)).sum(-1))
        agree.append(a.argmax(-1) == b.argmax(-1))
        t = targets[i:i + chunk, None]
        lp_ref.append(a.gather(1, t)[:, 0])
        lp.append(b.gather(1, t)[:, 0])
    return torch.cat(kl), torch.cat(agree), torch.cat(lp_ref), torch.cat(lp)


def summary(name, kl, agree, lp_ref, lp, secs, extra=None):
    q = torch.quantile(kl.float()[torch.randperm(len(kl), generator=torch.Generator().manual_seed(0))[:100000].to(kl.device)],
                       torch.tensor([0.5, 0.9, 0.99], device=kl.device))
    return {"build": name, "positions": len(kl), "kl_mean": kl.mean().item(), "kl_median": q[0].item(),
            "kl_p90": q[1].item(), "kl_p99": q[2].item(), "top1_agree": agree.float().mean().item(),
            "nll_ref": -lp_ref.mean().item(), "nll": -lp.mean().item(), "seconds": round(secs, 1)} | (extra or {})


def spec(always_on="mlx4", gate_up="q3", down="q3", over=None):
    e = [(gate_up, down)] * LAYERS
    for layer, v in (over or {}).items():
        e[layer] = v
    return {"always_on": always_on, "experts": e}


DEFAULT_IMPL = {}


def impl(model, attn=None, experts=None):
    """Switch the attention or experts kernels (None: the defaults loaded with the model): same weights, different
    summation order. The KL this gives is the floor of bf16 arithmetic itself."""
    model.set_attn_implementation(attn or DEFAULT_IMPL["attn"])
    model.set_experts_implementation(experts or DEFAULT_IMPL["experts"])


def plans(name):
    if name == "check":
        return {"bf16-again": spec("bf16", "bf16", "bf16"), "r1": spec(), "bf16-restored": spec("bf16", "bf16", "bf16")}
    if name == "floor":
        return {"all-r8": spec("r8", "r8", "r8"), "always-on-r8": spec("r8", "bf16", "bf16"),
                "experts-r8": spec("bf16", "r8", "r8"), "experts-r4": spec("bf16", "r4", "r4"),
                "always-on-r4-like-mlx4": spec("mlx4", "bf16", "bf16")}
    if name == "fidelity":                                              # what R1 gets back per byte
        mlx4 = dict.fromkeys(PARTS, "mlx4")
        return {"r1": spec(),
                "r1+embed8": spec(mlx4 | {"embed": "r8"}),
                "r1+shared8": spec(mlx4 | {"shared_expert": "r8"}),
                "r1+attn8": spec(mlx4 | {"attention": "r8"}),
                "r1+linattn8": spec(mlx4 | {"linear_attention": "r8"}),
                "r1+linattn6": spec(mlx4 | {"linear_attention": "r6"}),
                "r1+allattn8": spec(mlx4 | {"attention": "r8", "linear_attention": "r8"}),
                "r1+allattn6": spec(mlx4 | {"attention": "r6", "linear_attention": "r6"}),
                "r1+down4": spec(down="q4"),
                "r1+experts4": spec(gate_up="q4", down="q4"),
                "r1+cheap8": spec(mlx4 | {"embed": "r8", "shared_expert": "r8", "attention": "r8"}),
                "r1+alwayson8": spec(dict.fromkeys(PARTS, "r8") | {"lm_head": "mlx4"})}
    if name == "numerics":                                              # bf16 weights, other kernels
        bf16 = spec("bf16", "bf16", "bf16")
        return {"experts-eager": bf16 | {"impl": {"experts": "eager"}},
                "attention-eager": bf16 | {"impl": {"attn": "eager"}},
                "both-eager": bf16 | {"impl": {"attn": "eager", "experts": "eager"}},
                "defaults-again": bf16}
    if name == "parts":                                                 # one always-on part at MLX 4-bit, rest bf16
        return {f"mlx4-{p}": spec({p: "mlx4"}, "bf16", "bf16") for p in PARTS}
    if name == "builds":
        return {
            "mlx4": spec("mlx4", "mlx4", "mlx4"),                       # the stock MLX 4-bit build
            "r1": spec("mlx4", "q3", "q3"),
            "r2": spec("mlx4", "q2", "q3"),
            "always-on-only": spec("mlx4", "bf16", "bf16"),             # what the always-on path alone costs
            "experts-q3-only": spec("bf16", "q3", "q3"),                # what R1's experts alone cost
            "q4-search": spec("mlx4", "q4", "q4"),                      # 4-bit experts, our search vs MLX's rounding
            "r1-down4": spec("mlx4", "q3", "q4"),
            "r1-gateup4": spec("mlx4", "q4", "q3"),
        }
    if name == "layers":                                                # R1 with one layer's experts at 4 bits
        return {f"r1+L{i}": spec(over={i: ("q4", "q4")}) for i in range(LAYERS)}
    raise SystemExit(f"unknown plan {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--mlx4", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--data", default=str(HERE / "kl-data.pt"))
    ap.add_argument("--plan", default="builds")
    ap.add_argument("--only", nargs="*", help="build names from the plan")
    ap.add_argument("--conversations", type=int, default=0, help="first N conversations only (0: all)")
    ap.add_argument("--r1-hashes", help="r1-hashes.json: sha256 of the local R1 checkpoint's 3-bit expert words")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText

    rows = torch.load(args.data)
    if args.conversations:
        rows = rows[::max(1, len(rows) // args.conversations)][:args.conversations]
    targets = torch.cat([r["ids"][1:][r["mask"][:-1]] for r in rows]).long().cuda()
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.bf16, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"loaded in {time.time() - t0:.0f} s; {len(rows)} conversations, {len(targets)} scored positions", flush=True)
    DEFAULT_IMPL.update(attn=model.config._attn_implementation, experts=model.config._experts_implementation)
    print(f"kernels: {DEFAULT_IMPL}", flush=True)
    builds = Builds(model, MLX(Path(args.mlx4)), Path(args.bf16), args.imatrix)

    t0 = time.time()
    ref_h = hidden(model, rows)
    ref_w = model.lm_head.weight.detach().clone()
    print(f"bf16 reference: {time.time() - t0:.0f} s", flush=True)
    assert len(ref_h) == len(targets)

    out = HERE / "results" / f"kl-{args.plan}.jsonl"
    out.parent.mkdir(exist_ok=True)
    for name, s in plans(args.plan).items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        builds.apply(s)
        impl(model, **s.get("impl", {}))
        prep = time.time() - t0
        h = hidden(model, rows)
        kl, agree, lp_ref, lp = score(ref_h, ref_w, h, model.lm_head.weight, targets)
        res = summary(name, kl, agree, lp_ref, lp, time.time() - t0, {"prep_s": round(prep, 1), "conversations": len(rows)})
        print(json.dumps(res), flush=True)
        torch.save({"kl": kl.half().cpu(), "agree": agree.cpu()}, out.parent / f"kl-{args.plan}-{name}.pt")
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        if name == "r1" and args.r1_hashes:
            want = json.loads(Path(args.r1_hashes).read_text())
            same = sum(hashlib.sha256(builds._expert_q(int(k.split(".")[3]), k.split(".")[6], "q3")[0].view(torch.int32)
                                      .numpy().tobytes()).hexdigest() == v for k, v in want.items())
            print(f"in-process 3-bit experts equal the R1 checkpoint's: {same} of {len(want)}", flush=True)

if __name__ == "__main__":
    main()
