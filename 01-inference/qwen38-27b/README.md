# Qwen3.8-27B on the RTX 3090

What a 4-bit build of Qwen3.8-27B looks like on this card: how it fits, how fast it decodes against the bandwidth
ceiling, and how much context is left for real work.

```bash
CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH \
  uv run --project 01-inference/envs/vllm 01-inference/qwen38-27b/bench.py --ctx 8192
```

vLLM JIT-compiles a kernel at startup, so it needs `nvcc` (`/opt/cuda`). Results are in `results/vllm-awq.json`.

The second half compares an EXL3 build in ExLlamaV3, which has its own environment (its wheel pins torch 2.13):

```bash
uv run --project 01-inference/envs/exl3 01-inference/qwen38-27b/bench_exl3.py --ctx 262144 --slots 1 --long 65536
uv run --project 01-inference/envs/exl3 01-inference/qwen38-27b/bench_exl3.py --ctx 32768 --slots 8 --draft 0 --long 16384
```

## The build

[`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4): 4-bit weights in groups of
32, with a bf16 scale and an integer zero point per group. vLLM runs it with Marlin kernels, which work on Ampere.

The architecture is the one from the token walkthrough, scaled up: 64 layers, only every 4th is full attention, and
the other 48 are Gated DeltaNet.

## Where the bytes are (21.0 GB on disk)

| Part | GB | Precision | Read per decode step |
|---|---|---|---|
| MLPs | 9.89 | 4-bit (+1.07 GB of group scales) | all of it |
| DeltaNet layers | 3.30 | 4-bit, small parts bf16 | all of it |
| LM head | 2.54 | **bf16** | all of it |
| Embedding table | 2.54 | **bf16** | one row |
| Full-attention layers | 0.97 | 4-bit | all of it |
| Vision tower | 0.92 | bf16 | none (text only) |
| MTP head | 0.85 | bf16 | only when drafting |

- A decode step streams **16.70 GB** of weights.
- It also reads and writes **302 MB** of DeltaNet state.
- It reads **64 KB of KV cache per context token**. Only the 16 full-attention layers keep a cache; if all 64 did, it
  would be 256 KB.

The LM head alone is 15% of the bytes per token. This build left it in bf16.

**Ceiling at 829 GB/s:**

| Context | Ceiling |
|---|---|
| 0 tokens | 48.7 tok/s (20.5 ms/token) |
| 16K tokens | 45.9 tok/s |
| 32K tokens | 43.3 tok/s |

The setup notes' estimate of ~58 tok/s assumed every weight was 4-bit. The bf16 embedding and LM head are the
difference.

## Measured

| | Result |
|---|---|
| Weights in VRAM | 18.37 GiB |
| Room left for context | ~1 GiB → **10,119 tokens**, shared by all requests |
| Decode, short prompt | **32.3 tok/s** (31.0 ms/token), 66% of ceiling |
| Decode after 7,825 tokens of context | 30.9 tok/s (32.3 ms/token) |
| Prefill | 996 tok/s (7,825 tokens in 7.9 s) |
| Batch 1 / 4 / 8 | 32 / 112 / 117 tok/s total |
| MTP speculative decoding | does not fit: loading the draft head needs 2.37 GiB more |
| Sanity check ("17 sheep, all but 9 run away") | `9` |

Other constraints:
- The desktop holds ~1.2 GiB of VRAM, so vLLM can take at most ~93.5% of the card.
- Settings: prefill chunks of 2,048 tokens, at most 8 sequences.
- A 32K window needs 2.3 GiB of KV and failed; so did 10,240. 8,192 is what fits.

## What it shows

1. **It fits, but barely, and context is the casualty.** About 10K tokens is too small for agent work: a harness
   prompt plus a few tool outputs fills it, and Xiaomi's RL tasks run to 200K. On this card, the model's size decides
   how much it can *read*, not just how fast it runs.
2. **Decode behaves as predicted.** It reaches 66% of the byte-count ceiling, and it barely slows with context
   (32.3 → 30.9 tok/s at 7.8K tokens). That's the hybrid design at work: most layers carry a fixed-size state
   instead of a growing cache.
3. **Prefill is compute-bound.** About 2 × 25B FLOPs per token at 996 tok/s is ~50 TFLOPS, two thirds of the card's
   measured bf16 peak.
4. **Batching is capped by memory, not compute.** Throughput goes from 32 to 112 tok/s at 4 streams, then flattens at
   8, because the ~1 GiB pool can't hold more sequences' state and cache at once.
5. **The bf16 embedding and LM head (5.1 GB) are the waste.** GGUF k-quants quantize them too: Q4_K_M is 16.5 GB with
   everything included. At 5.95 GB, ternary Bonsai 2 27B would have a ~139 tok/s ceiling and leave ~16 GB for
   context. That makes low-bit quantization a *context* win on a 24 GB card, not just a speed win.

## Second build: EXL3 4.0 bpw in ExLlamaV3

[`turboderp/Qwen3.8-27B-exl3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), branch `4.00bpw`: the build the
[MiaAI one-click kit](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install) serves. EXL3 rotates
each weight matrix, then encodes it with a trellis code. Everything is quantized: layers at 4.0 bits per weight, the
LM head at 6.0.

### Where the bytes are (16.9 GB on disk)

| Part | GB | Where it lives |
|---|---|---|
| Layers (MLP, DeltaNet, attention) | 12.23 | GPU, read every step |
| LM head | 0.95 | GPU, read every step (6 bits, not bf16: 2.54 → 0.95) |
| Embedding table | 2.54 | **system RAM**: only one row per token is needed |
| Vision tower | 0.92 | not loaded (text only) |
| MTP head | 0.21 | GPU, read when drafting |

A decode step streams **13.18 GB** of weights, down from 16.70. The ceiling rises from 48.7 to **61.5 tok/s**.

### Measured

Two loads, because the cache has a second cost (below): 262K context with one sequence and MTP, then 32K context with
eight sequences.

| | vLLM AWQ | EXL3 4.0 bpw |
|---|---|---|
| Weights + MTP head in VRAM | 18.37 GiB (MTP doesn't fit) | 13.4 GiB |
| Context that fits | 10,119 tokens, bf16 KV | **262,144 tokens**, int4 KV (4.8 GiB) |
| Decode, short prompt | 32.3 tok/s (66% of ceiling) | 32.6–33.6 tok/s (53–55% of ceiling) |
| Decode after long context | 30.9 tok/s at 7.8K | 31.4 at 20K; **26.0 at 83K** |
| Prefill | 996 tok/s at 7.8K | 939 tok/s at 20K; 714 at 83K |
| Batch 1 / 4 / 8 (32K window) | 32 / 112 / 117 tok/s | 33 / 107 / **185** tok/s |
| MTP drafting, 4 tokens | OOM | **61.6 tok/s** short (43% of drafts accepted); **53.0 at 83K** (69%) |
| Sanity check | `9` | `9` |

Notes:
- The desktop held 1.2 GiB. The 262K load used 20.5 GiB of the card's 24, with 3.5 GiB to spare.
- The 83K-token prompt came from the same filler as vLLM's but was sized for 64K; ExLlamaV3's tokenizer count is the
  one reported.
- The 262K run's batch-1 figure (24 tok/s) isn't reported. It came right after the 83K job, when the generator was
  reshuffling a nearly full cache. The 32K run measures batching cleanly.
- The 262K run's sanity answer was `9`, but generation ran past the end-of-turn token. The stop condition was fixed
  for the 32K run.

### What it shows

6. **Quantize everything, and context comes back.** Moving to 6 bits for the LM head, 4 bits for the KV cache and CPU
   RAM for the embedding frees about 5 GiB. That is the difference between 10K and 262K tokens. The Module 1 lesson
   holds: on 24 GB, bits per weight decide how much an agent can read.
7. **Fewer bytes did not mean faster decode.** EXL3 streams 21% fewer bytes but decodes at the same ~33 tok/s, only
   53% of its ceiling. The trellis code costs arithmetic to unpack, and Ampere pays for it. The byte ceiling is a
   limit, not a prediction; the kernel decides how close you get.
8. **The "fixed-size" state still costs memory per sequence.** Each concurrent sequence needs its own DeltaNet state,
   151 MB. MTP keeps 5 snapshots of it, for rolling back rejected drafts, which makes 728 MiB per sequence. The
   default 16 slots would not fit next to a 262K cache. On this hybrid, memory buys either one very long sequence or
   several shorter ones.
9. **MTP is the speed lever that was missing.** The model's own draft head doubles single-stream decode: 1.9× on
   prose, 2.0× on the long summary, where the text is more predictable and 69% of drafts land. That is why
   speculative decoding is the default in serving kits for agents, which run one long sequence at a time.
10. **Long context costs more than its bytes.** An 83K-token int4 cache adds 1.6 GB per step, which predicts about 10%
    slower decode. It measured 20% slower, because the quantized cache is unpacked inside attention. Prefill also
    slows (939 → 714 tok/s) as each new token attends to more history.

## Going to 2 bits: speed first, quality next

Two ~2-bit builds of the same base model:
- **EXL3 2.0 bpw**: compressed after training, no retraining; 11 GB on disk.
- **Bonsai 2 (PQ2_0)**: trained to be ternary, run in [PrismML's llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
  (prebuilt CUDA 12.8 binary; `01-inference/envs/prism-llama` supplies its CUDA 12 runtime libraries).

| | EXL3 4.0 bpw | EXL3 2.0 bpw | Bonsai 2 PQ2_0 |
|---|---|---|---|
| Weights in VRAM (approx.) | 13.4 GiB | ~8 GiB | 6.7 GiB |
| Decode, batch 1 | 33 tok/s | 40.6 tok/s | **55.4 tok/s** (llama-bench tg128) |
| Decode with MTP ×4 | 61.6 | 62.8 short, 78.6 long | not tested |
| Batch 8, 32K window | 185 tok/s | 186 tok/s | not tested |
| Prefill | 939 tok/s at 20K | 984 tok/s at 10K | 1,154 tok/s (pp512) |
| Sanity check (thinking off, greedy) | `9` | **`8` (wrong)** | `9` (thinking on) |

- Halving the bits bought only 23% more speed in EXL3 (33 → 40.6 tok/s, about 37% of its byte ceiling). Unpacking
  the trellis code is arithmetic, and on Ampere that arithmetic, not memory, sets the pace at 2 bits.
- Bonsai's plain ternary format is cheaper to unpack and runs faster at a similar size.
- The wrong sheep answer is a single greedy sample, not evidence. The long-reasoning test below is what counts. (The
  first sanity checks in `bench_exl3.py` encoded the chat markers as plain text; the table's answers come from
  `reasoning_probe.py`, which encodes them properly.)

**Open question.** Prism reports IQ2_XXS (a 2-bit build with no training) collapsing on long reasoning: AIME26
94.6 → 57.5, MATH-500 99.8 → 84.6. Bonsai 2 holds (95.8, 98.8). Does EXL3 2.0, which adds a rotation and a trellis
code but still no training, collapse too?

**Riel's prediction (2026-09-28): EXL3 2.0 collapses the way IQ2_XXS did.**

### The test: 20 hard math problems, thinking on

```bash
uv run --project 01-inference/envs/exl3 01-inference/qwen38-27b/reasoning_probe.py exl3 --revision 2.00bpw
uv run --project 01-inference/envs/prism-llama 01-inference/qwen38-27b/reasoning_probe.py server --label bonsai2-pq2
```

The problems are 20 level-5 MATH-500 problems (seed 0). Each gets one sample at the Qwen3.8 card's settings
(temperature 1.0, top-p 0.95, top-k 20), a 16K-token cap with thinking included, and an 8-bit KV cache. Grading is
math-verify on the text after `</think>`. Full traces are in `results/reasoning/*.traces.jsonl.gz`. Each run took
13–20 minutes of GPU, with 10–20 minute cool-downs between runs.

| | EXL3 4.0 bpw | EXL3 2.0 bpw | Bonsai 2 (1.72 bpw, trained) |
|---|---|---|---|
| Correct | **18/20** | 12/20 | 16/20 |
| Wrong answer | 0 | **4** | 0 |
| Hit the 16K cap without answering | 2 | 4 | 4 |
| Looping traces (>20% repeated lines) | 0 | **2** | 0 |
| Median tokens, on the 10 problems all three solved | 717 | **1,367** | 939 |
| Wall time, 8 at once | 12.8 min | 14.9 min | 20.4 min |

Paired against 4.0: 2.0 lost 7 problems and gained 1 (exact McNemar p = 0.07); Bonsai lost 3 and gained 1 (p = 0.63).

**Verdict on Riel's prediction ("EXL3 2.0 collapses like IQ2_XXS"): right in direction and in kind, but not yet
proven.**
- A 30-point drop (90% → 60%) is the size of a collapse. For comparison, IQ2_XXS fell 15 points on all of MATH-500.
  But 20 problems give p = 0.07: strong evidence, not proof.
- The failures have the collapse signature. There are loops (one algebra line written 42 times), arithmetic slips
  that turn into spirals of re-checking until the cap, and reasoning that runs twice as long on the problems it
  still solves.
- It's not *only* long reasoning. 2.0 also got two short problems wrong on plain arithmetic (2 + 7 + 17 = "26", a
  product of 32 instead of 64). Errors happen at every token; long chains just give them more chances to compound.

**Bonsai holds.** It has no wrong answers and no loops. All four of its failures are clean reasoning that ran out
of room: it thinks at length by design, and one trace had the right fraction just before the cap. A bigger budget
would probably recover some of them. Trained ternary at 1.72 bpw beats rounding to 2.0 bpw, as Prism claims, though
on this sample it doesn't reach 4.0.

**Confound, found 2026-09-28.** The Bonsai GGUF's chat template defaults to reasoning effort `xhigh`, which adds a
system line: "Please think carefully through the task, validate key assumptions, consider plausible alternatives…".
The EXL3 prompts had no system line, which is the template's `medium`. So Bonsai was told to think longer than the
EXL3 builds, and its four cap-limited failures may partly come from that line. It's being rerun with
`--template-kwargs '{"reasoning_effort": "medium"}'` (label `bonsai2-pq2-medium`) for a matched comparison.

**For the Ornith quantization plan:** rounding alone (post-training quantization) is safe near 4 bits and breaks
reasoning at 2. To get below ~3 bits and keep reasoning intact, the model has to be *trained* for its low-bit
weights (QAT or distillation), which is what Bonsai did.

### Where is the cliff? EXL3 3.0 bpw

turboderp's KL vs bf16 is 0.05 at 4.0, 0.11 at 3.0 and 0.35 at 2.0, and 0.35 broke reasoning.

**Riel's prediction (2026-09-28): about 16–18/20 at 3.0, fewer loops than 2.0.** Close to 4.0 but not quite there, a
marked step up from 2.0, and 3.0 is the cutoff for rounding without training.

| | EXL3 4.0 | **EXL3 3.0** | EXL3 2.0 | Bonsai 2 |
|---|---|---|---|---|
| Correct | 18/20 | **19/20** | 12/20 | 16/20 |
| Wrong answers | 0 | **0** | 4 | 0 |
| Hit the cap | 2 | 1 | 4 | 4 |
| Looping traces | 0 | **0** | 2 | 0 |
| Median tokens on the 10 problems all four solved | 717 | 844 | 1,367 | 939 |

- Paired, 3.0 vs 2.0: 3.0 solved 7 problems that 2.0 missed, and 2.0 solved none that 3.0 missed (exact McNemar
  p = 0.016). That is the first significant gap in this series.
- 4.0 vs 3.0: 1 vs 2 in each direction, which is noise.

**Verdict on Riel's prediction: right.** 3.0 is indistinguishable from 4.0 on this probe, with no loops, and it is a
marked step up from 2.0. **The cliff for rounding without training is between 3 and 2 bits per weight** (KL 0.11
holds; KL 0.35 breaks). One sample per problem at temperature 1.0 means 18 vs 19 is luck, not 3.0 beating 4.0.
Its one sign of strain: two answers came in just under the cap (15,670 and 16,317 tokens), where 4.0 took 4,296 and
ran out.
