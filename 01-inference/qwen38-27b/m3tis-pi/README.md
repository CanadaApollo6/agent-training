# M3TiS #1 pi variant: Qwen3.8-27B + harness-hub's pi adapter, in the M3TiS #1 format

The pilot in `06-agents/harness-evals` (2026-11-17) put harness-hub's pi LoRA at 13.0/19 TB2 tasks with thinking off,
against 12.0 for the base model at medium effort (p 0.57), with 0.70x the tokens. This folder checks whether the
adapter survives M3TiS #1's 4-bit format, and what it costs in speed. Nothing here is published.

## How the build works

`build_mlx4.py` starts from Vontra's MLX 4-bit checkpoint (affine, group 64), the weights M3TiS #1 ships. Of its
weights, the LoRA touches 256: gate, up and down in all 64 layers, and q, k, v and o in the 16 full-attention layers.
For each of these:

1. Merge: base BF16 + 2·B·A, computed in fp32 and rounded to BF16, as PEFT does.
2. Quantize with `mx.quantize` (MLX on CPU) and replace that weight's packed values, scales and biases.

Every other tensor, plus the shard layout and metadata, stays byte-identical to Vontra's.

Recipe check: before replacing anything, the script quantizes 12 *base* weights and compares them with Vontra's.

- 99.93% of groups have the same scale and bias.
- The error is the same to 4 digits: 0.09244 vs 0.09244.
- 1.2% of packed words differ. Each holds one value on a rounding boundary, from CPU vs Metal arithmetic.

So Vontra's file is plain `mx.quantize` output, not a calibrated quant. The new weights are quantized the same way
as the rest. The LoRA changes each weight by 0.18% of its norm at the median, and 0.26% at most.

## KL against BF16 (`kl_q27.py`)

Texts: 32 of our own pilot conversations (`export_texts.py`), up to 6,144 tokens each:

- 16 from the adapter arm (thinking off)
- 16 from the base arm at medium (thinking kept)

Each conversation comes from a different task. Two HF models stay loaded, and the weights are swapped between steps.

Mean KL(BF16 ‖ 4-bit) on assistant tokens, the tokens the model generates:

| Comparison | Adapter-arm texts | Medium-arm texts | Top-1 agreement |
|---|---|---|---|
| 1. base vs Vontra 4-bit: what M3TiS #1 ships with | 0.0299 | 0.0301 | 96.0% / 95.2% |
| 3. merged vs our 4-bit build | **0.0305** | **0.0308** | 96.0% / 95.2% |
| 2. merged vs Vontra 4-bit: adapter effect plus quant error | 0.0450 | 0.0351 | 95.1% / 94.8% |

The new build loses as much to 4-bit as M3TiS #1 already does: 0.0305 vs 0.0299. Running Vontra's weights unchanged
would lose half the adapter's shift on its own texts: 0.0450 vs 0.0305. The re-quantized build keeps it.

### The all-positions numbers, and why they aren't the headline

Over all positions, the mean KL is about 1.46, with a p99 of 20 nats, in every comparison. Almost all of it comes
from tool output and the system prompt: text the model reads but never writes.

I checked whether that was a measurement bug (`kl_outliers.py`, `path_test.py`, `vllm_nll.py`;
`results/hf-vs-vllm-nll.txt`). It isn't:

- Swapping the GPUs gave identical numbers.
- The fla kernels, the pure-torch fallback and eager attention agree.
- An early position's loss doesn't change with text length.
- vLLM, a separate engine, gives the same next-token loss as HF on the same tokens: 2.577 vs 2.580 nats on a
  6,144-token text.

On that text's tool output, BF16 Qwen3.8-27B itself averages 3.2 nats per token. Where the model is that unsure, a
small weight change swings its guesses a long way. That says nothing about the replies it writes. The
assistant-position KL is the number to compare.

## Speed and draft acceptance (`speed_q27.py`)

Setup:

- One RTX A6000 (sm_86, the 3090's architecture, 48 GB).
- The released bundle: `tensorfold-native 1.0.5` with the DFlash2 drafter, `--no-thinking`, 32K window.
- The 1.0.5 log has no timing. Speed is timed from the stream: tokens after the first, over the time from the first
  streamed token to the last.
- Code and chat: 3 runs each after a warm-up. Agent turns: our 12 mid-run pilot turns (2-11K-token prompts), one
  run each.

| Case | Vontra tok/s | Pi build tok/s | Vontra tokens/round | Pi build tokens/round |
|---|---|---|---|---|
| Code, greedy | 277 | 280 | 8.26 | 8.26 |
| Chat, greedy | 148 | 139 | 4.41 | 4.10 |
| Agent turns, greedy (pooled) | 217 | 206 | 6.47 | 6.34 |
| Code, T 0.8 | 315 | 286 | 9.66 | 8.68 |
| Chat, T 0.8 | 135 | 132 | 4.13 | 4.00 |
| Agent turns, T 0.8 (pooled) | 206 | 180 | 6.41 | 6.08 |

The DFlash2 drafter was trained on the base model's outputs. On the adapter, it keeps 0-10% fewer tokens per round.
Agent-turn speed is 5-13% lower. Part of that gap is noise: the two builds write different replies (1,758 vs 1,416
greedy tokens over the 12 turns), and each turn ran once.

These are A6000 numbers. The A6000 has less memory bandwidth than a 3090 (768 vs 936 GB/s), so 3090 speeds should
be somewhat higher for both builds.

## Verdict

The format works: the adapter goes into M3TiS #1's 4-bit weights with no extra quantization loss. Speed stays
within about 10% of the release, on the same drafter. The TB2 gain itself, +1.0 task with p 0.57, isn't significant
yet. That needs more attempts, or the 4-bit build run under pi, before it's worth calling it a variant.

## Files

| File | What it does |
|---|---|
| `export_texts.py` | Writes `pod/kl.jsonl` (32 conversations) and `pod/turns.jsonl` (12 agent turns) from the pilot traces |
| `build_mlx4.py` | Builds the 4-bit weights, with the recipe check. About 9 minutes on 56 CPU cores |
| `kl_q27.py` | Runs the three KL comparisons |
| `speed_q27.py` | Measures speed and acceptance against a running `m3tis-serve.sh` |
| `kl_outliers.py`, `path_test.py`, `vllm_nll.py` | The all-positions KL investigation |
| `setup_pod.sh`, `run_pod.sh` | Pod setup, then build → speed (Vontra) → KL → speed (pi) |
| `results/` | `kl.json`, `speed-*.json`, `M3TIS_BUILD.json` (per-weight stats), logs, revisions, `pods.log` |

Revisions (`results/revisions.txt`):

- Qwen/Qwen3.8-27B 37561816
- harness-hub/Qwen3.8-27B-pi-adapted e786f849
- Vontra/Qwen3.8-27B-MLX-4bit 22d8d538
- z-lab/Qwen3.8-27B-DFlash2 50307d4c
- CanadaApollo6/Qwen3.8-27B-M3TiS-3090 3074a4f3

Pod: 4x A6000 (lambdalabs, $4.36/h), 62 minutes, about $4.50. The built weights weren't kept. Rebuild them with
`build_mlx4.py`.
