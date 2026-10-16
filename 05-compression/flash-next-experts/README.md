# Can Qwen3.8-Flash-Next lose half its experts?

The question behind a possible M3TiS build of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(125B-A6B, 512 experts per layer, top-10). It only fits a 3090 if about half the experts go and the rest drop to
2 bits (see the reading-spine entry in the top README). This tests the first half of that: drop experts, no
quantization.

**Answer: not without repair.** Halving the experts moves the model's next-token predictions far more than a full
low-bit quantization does, even on our own agent traces, before any 2-bit rounding is added.

## What ran

`hf_job.sh` on HF Jobs h200x8, 2026-10-16, run `experts-1016-h200b`: the BF16 model, 67 minutes, about $45. (An
earlier full-size attempt ran at 179 tokens/s, too slow for the time cap, and was stopped after about $9.)

- `prep_tokens.py` renders our harness traces (DeepSWE, Terminal-Bench 2 and 4, Terminal-Lego; written by Qwen3.8-27B,
  Ornith and DeepSeek runs) with Flash-Next's chat template into 8K-token windows. Tasks are split by name, so the
  held-out windows come from tasks the expert choice never saw. Control is English web text plus Chinese, Spanish and
  Russian Wikipedia.
- `router_stats.py` pass 1 (30 windows, 244K tokens) records, per layer and expert: how often the router picks it,
  the routing weight it gets, and a REAP-style score (weight times the size of the expert's output). It then picks
  the experts to keep, the same number in every layer.
- Pass 2 runs the held-out windows (12, 98K tokens) and control windows (8, 58K tokens) twice: with the full model,
  and with the dropped experts masked out of the router. Masking is what pruning does to the forward pass: the router
  chooses its 10 among the kept experts and renormalizes. It compares next-token distributions.

## Results

Expert usage on the calibration tokens: the 256 most-used experts per layer get **89%** of routing weight (layer 0:
81%; last layers: 95-96%). The top 384 get 98%. On held-out tasks the same 256 get only **82%**, and on control text
**42%**.

Next-token drift on held-out agent traces (KL in nats per token, full model vs pruned):

| Keep per layer | Picked by | KL | Top-1 agrees |
|---|---|---|---|
| 384 (cut 25%) | usage | 0.47 | 84% |
| 256 (cut 50%) | usage | 0.80 | 78% |
| 256 (cut 50%) | REAP score | 0.80 | 78% |

By source, cutting 50% by usage: DeepSWE 0.46 KL (82% agree), Terminal-Bench 2 0.47 (85%), the easy Terminal-Lego
windows 0.06 (92%), and windows full of hard-to-predict tool output (perplexity ~38 even for the full model) 1.4-3.9.

Control text, cutting 50%: KL 1.77, top-1 agreement 49%. Chinese Wikipedia perplexity goes from 8.0 to 99.6, Spanish
from 6.8 to 54, Russian from 3.3 to 27.

For scale: our LFM2.5-24B 4-bit Zig build is 0.14 KL from BF16, and the 4-bit KV-cache gate is ≤0.01. Halving the
experts alone is about 6x the drift of a whole 4-bit build.

## Reading it

- The router is fairly concentrated, but not enough. The last 11% of routing weight carries real information, and the
  experts our traces use shift between tasks: 89% coverage in the calibration tasks falls to 82% on new tasks.
- Choosing by REAP's score instead of plain usage made no difference here.
- Other languages and general knowledge collapse, as expected for an agent-trace calibration.
- One oddity: on the DeepSWE windows, the REAP-pruned model gave the actual text a *lower* perplexity than the full
  model (4.27 vs 4.79), while still disagreeing with it a lot (KL 0.58). The traces were written by other models, so
  "closer to the real next token" isn't the same as "closer to Flash-Next". I haven't looked further.
- Not tested: a repair pass (training the pruned model to match the full one). It might recover some of this, but at
  125B it's a multi-GPU training job, not a quick test.
