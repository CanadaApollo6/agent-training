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
EXL3 builds, and its four cap-limited failures may partly come from that line. So it was rerun with
`--template-kwargs '{"reasoning_effort": "medium"}'` (label `bonsai2-pq2-medium`) for a matched comparison.

| Bonsai 2 PQ2_0 | Correct | Wrong | Hit the cap | Loopy | Median tokens, 13 problems both runs solved |
|---|---|---|---|---|---|
| xhigh (template default) | 16/20 | 0 | 4 | 0 | 898 |
| medium (matches EXL3's prompt) | 14/20 | 3 | 3 | 0 | 1,192 |

- **The confound wasn't the cause.** Without the "think carefully" line, Bonsai didn't reason shorter (if anything
  longer) and still hit the cap 3 times. The xhigh line didn't manufacture the runaways.
- **The bigger lesson is noise.** 5 of 20 problems changed outcome between two runs of the same model (3 one way, 1
  the other, p = 0.63). Sampling at temperature 1.0 with one sample per problem moves a score by ±2–3 problems between
  runs, the same size as several gaps we've been reading. The 9B ladder's losses all go one way (3–0, 4–0), which
  is worth more than a two-way swap, but a firm call needs more samples per problem.
- Matched comparison against EXL3 4.0: Bonsai lost 5 and gained 1 (p = 0.22). "Trained ternary holds up, but short
  of 4-bit" stands. The 2.0-bpw comparison (12/20 with 4 wrong and 2 loops) stands too.

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

## Third engine: TensorFold with the DFlash2 draft model (2026-10-07)

Goal (Riel, 2026-10-07): as fast as possible on the 3090, with context you can actually work in, without losing
intelligence.

[TensorFold](../tools/TensorFold) (0.3.6.3, our sm86 patch) runs the MLX 4-bit build
([`Vontra/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit), 4-bit in groups of 64, 15 GB)
with [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), a 5-layer draft model. The
draft model guesses a block of upcoming tokens in one go, and the big model checks all of them in one pass. The big
model keeps or rejects every guess, so the output is the big model's own; drafting doesn't change what it says.

```bash
./bench_tf.sh mlx4-nodraft --no-drafts
TF_DRAFT_PACKED=1 TF_BUDGET_GIB=40 LONG="16000 32000 56000" ./bench_tf.sh mlx4-dflash2-64k --context 65536
```

### Getting it to start on 24 GB

Out of the box it refuses: its memory planner says zero tokens of context fit. The planner is written for
128 GB machines and over-counts in three places:

| What it counts | Planner | Real |
|---|---|---|
| Draft model | 7.2 GiB (4 bytes a weight) | ~1 GiB: it is packed to 4-bit as it loads |
| Fixed scratch space | 3.7 GiB | not visible in use |
| Context cache per token | 256 KB (4 copies at 4 bytes) | 64 KB held (bf16, one copy) |

With drafts off it allows only an 8K window, and then the card has ~6 GB unused. `serve_kernels.py` gained two
switches: `TF_DRAFT_PACKED=1` counts the draft model at its packed size, and `TF_BUDGET_GIB` overrides the planner
outright, so the window is set by `--context` and checked by measuring real memory with long prompts.

### Measured

Decode speed, median of 3 (TensorFold's public bench: a code prompt and a chat prompt, 256 tokens):

| | No drafts | With DFlash2 | EXL3 4.0 + MTP×4 (before) |
|---|---|---|---|
| Chat, greedy | 36.2 tok/s | **102** tok/s | 61.6 |
| Chat, sampled (temp 1) | 35.4 | **81–104** | – |
| Code, greedy | 33.1 | **143–148** | – |
| Code, sampled | 35.2 | **159–192** | – |

(Two runs each for the drafted cells; ranges are run-to-run. The host was busy, load average ~15.)

Long prompts (with drafts, 64K window; a list of records, then "what is the code for record N?"):

| Prompt | Prefill | Recall | Peak card memory (desktop's 1.4 GB included) |
|---|---|---|---|
| 20,707 tokens | 27 s (758 tok/s) | right | 21.3 GB |
| 41,741 tokens | 59 s (704 tok/s) | right | **24.0 GB of 24.5** |
| ~73K tokens | – | – | refused: longer than the 64K window |

### What it shows

11. **The draft model is the speed lever, and it's big.** 3× on chat and 4–5× on code over plain decoding, and
    1.7× over EXL3's MTP drafting. Plain decoding is 35 tok/s, 64% of this build's ~55 tok/s byte ceiling. Drafting
    goes past the ceiling because one read of the weights checks many tokens.
12. **Context is now the limit.** ~42K tokens fills the card. Memory grows ~128 KB per prompt token, twice the cache
    itself, so something during prompt processing holds a second copy. EXL3 fit 262K by caching keys and values in
    4 bits and keeping the embedding table in system RAM; this engine does neither yet.

### The missing memory: kept prompt states pin old buffers

A first guess was the cache's growth rule: it doubles when full, so a 42K prompt sits in a 64K-row buffer. Growing in
2K-row steps instead (`TENSORFOLD_KV_STEP=2048`) changed nothing: the 41.7K prompt still peaked at 24.05 GB.

The real cause: the engine keeps up to 4 prompt states (at message starts and the prompt's end) so a follow-up
request can resume instead of reprocessing. A kept state holds the cache buffer that existed when it was taken. When
the cache outgrows that buffer, or a new unrelated prompt starts, the old buffer can't be freed. (`--prompt-cache-gib`
doesn't limit this: it is the MLX path's setting.)

`TENSORFOLD_ONE_BUFFER=1` (in `tensorfold-27b-context.patch`, against TensorFold 0.3.6.3; the forward.py hunks include
the sm86 patch's lines) does two things:
- A fresh prompt allocates its whole window at once, so the buffer never moves and every kept state shares it.
- A prompt that doesn't resume a kept state drops the kept states of earlier prompts first.

One conversation at a time keeps its resume points; that's the agent case.

```bash
TENSORFOLD_ONE_BUFFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_DRAFT_PACKED=1 TF_BUDGET_GIB=200 \
  LONG="32000 48000" ./bench_tf.sh onebuf-80k --context 81920
```

| | Before | One buffer |
|---|---|---|
| Window | 64K | **80K** |
| Longest prompt run | 41,741 tokens | **63,045 tokens** (recall right, 95 s prefill at 666 tok/s) |
| Peak card memory | 24.0 GB at 41.7K, rising | 24.07 GB at both 41.7K and 63K: set at load |
| Decode, chat / code | 102 / 143–192 | 92–105 / 132–153 (same within noise) |

The 80K window leaves ~0.5 GB spare, so this is about the bf16 ceiling for this build.

### 8-bit context storage: 144K window, no measurable loss

The cache now stores each key and value as an 8-bit whole number plus one 16-bit scale per 32 values (8.5 bits a value
instead of 16), the scheme ExLlamaV3 uses. `TENSORFOLD_KV_BITS=8` turns it on (`cuda/kernels/kvq.py` in the patch).
The writes quantize rows as they go in. The three attention kernels (prompt, decode, draft check) turn them back into
16-bit as they read them, inside the kernel, so no 16-bit copy of the cache ever exists.

```bash
TENSORFOLD_KV_BITS=8 TENSORFOLD_ONE_BUFFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_DRAFT_PACKED=1 \
  TF_BUDGET_GIB=200 LONG="48000 100000" ./bench_tf.sh kv8-144k --context 147456
```

| | 16-bit cache | 8-bit cache |
|---|---|---|
| Window | 80K | **144K** |
| Longest prompt run | 63,045 tokens | **132,209 tokens** (recall right) |
| Peak card memory | 24.07 GB | 24.06 GB |
| Prompt reading | 666 tok/s at 63K | 642 tok/s at 63K, 501 tok/s at 132K (264 s) |
| Decode, chat / code | 92–105 / 132–153 | 98–117 / 124–157 (same within noise) |

**The intelligence check** (`kv_kl.py`): the same model reads one long real document (this repo's READMEs, then
TensorFold's source) with each cache. At three depths it scores the next 256 tokens through the decode path. The
table compares the full next-token probabilities of the two runs (`results/kv-kl-8bit.md`):

| Depth into the document | KL, mean | KL, worst token | Same top choice | Perplexity, 16-bit / 8-bit |
|---|---|---|---|---|
| 1,024 | 0.0007 | 0.004 | 99.2% | 8.91 / 8.87 |
| 16,384 | 0.0008 | 0.018 | 96.9% | 3.12 / 3.10 |
| 49,152 | 0.0003 | 0.005 | 99.6% | 1.81 / 1.81 |

That is well under what 4-bit weights cost (KL in the hundredths), and perplexity doesn't move. The top-choice
differences are near-ties, where the worst token's KL is still under 0.02. The check stops at 49K because the 16-bit
cache can't go much deeper on this card. `tests/test_kvq.py` checks the kernels themselves: they match a plain
PyTorch calculation on the same 8-bit cache to 0.2%.

### 4-bit context storage: the full 262K window

`TENSORFOLD_KV_BITS=4` stores each group of 32 values in 4 bits plus one scale (4.5 bits a value, 144 bytes a head
row). Before rounding, each group is mixed by a 32-wide Hadamard rotation (ExLlamaV3's scheme), which spreads the
odd very large value across the group so it doesn't wreck the group's precision.

The cache keeps values in their rotated form. The rotation preserves dot products and is its own inverse, so the
kernels never un-rotate the cache. The wrappers rotate the queries (and the new tokens' keys and values) once on the
way in, and rotate the output back once on the way out. A first version un-rotated every block as it was read and
lost a third of the decode speed (76 chat, 84 code).

```bash
TENSORFOLD_KV_BITS=4 TENSORFOLD_ONE_BUFFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_DRAFT_PACKED=1 \
  TF_BUDGET_GIB=200 LONG="100000" ./bench_tf.sh kv4r-262k --context 262144
```

| | 16-bit | 8-bit | 4-bit |
|---|---|---|---|
| Window | 80K | 144K | **262K** (the model's native maximum) |
| Longest prompt run, recall right | 63K | 132K | 132K |
| Peak card memory | 24.07 GB | 24.06 GB | 23.82 GB |
| Prompt reading at 132K | – | 501 tok/s | 515 tok/s |
| Decode, chat / code (tok/s) | 92–105 / 132–153 | 98–117 / 124–157 | 103–114 / 144–147 |

**Intelligence check** (`kv_kl.py`, `results/kv-kl-4bit.md`):

| Depth | KL mean | KL, worst token | Same top choice | Perplexity, 16-bit / 4-bit |
|---|---|---|---|---|
| 1,024 | 0.0034 | 0.030 | 96.9% | 8.91 / 8.92 |
| 16,384 | 0.0055 | 0.045 | 96.9% | 5.71 / 5.74 |
| 49,152 | 0.0018 | 0.029 | 99.2% | 2.12 / 2.12 |

That is 3–7× the 8-bit cache's drift, but still a tenth or less of what 4-bit weights cost, and perplexity moves by
under 0.5%.

Notes:
- The first 4-bit comparison looked catastrophic (KL 14 from 16K on). The cause was the test, not the cache: the
  document is this repo's READMEs, and this README had changed between the 16-bit run and the 4-bit run.
  `kv_kl.py` now reads a frozen token file (`results/kv_kl_tokens.pt`), and refuses to compare runs that read
  different tokens. The 8-bit numbers above come from a matched pair of runs on the earlier document.
- The first 4-bit version also got the recall check wrong on a 159K-token prompt (2784 for 4950). That test is
  thousands of near-identical records with one exact lookup and no thinking. No 16-bit or 8-bit window fits
  159K, so nothing yet says whether 4-bit storage or the length itself caused it.
- Not yet run: a recall check near 250K.

**Math probe with the 4-bit cache (2026-10-07): 15/20, against 18/20 for EXL3 4.0.** Same 20 problems, same
sampling, 262K window (`probe_tf.sh tf-kv4-262k`, `results/reasoning/tf-kv4-262k.json`, 17.7 min).

| | EXL3 4.0 bpw | TensorFold, MLX 4-bit weights + 4-bit cache |
|---|---|---|
| Correct | 18/20 | 15/20 |
| Wrong answer | 0 | 0 |
| Ran out of room (16K tokens) | 2 | 5 |
| Median / mean tokens | 985 / 4,162 | 1,881 / 6,094 |

- The 3 extra misses are all run-outs on problems EXL3 solved (intermediate_algebra/2015 and 558, geometry/686);
  it never gave a wrong answer. Thinking ran longer on most hard problems.
- This doesn't yet say the 4-bit cache costs anything. Three misses, all one way, from one sample each at
  temperature 1.0 is weak evidence (p = 0.25). Two things also changed besides the cache: the weights (MLX 4-bit
  group-64 vs EXL3 4.0) and the engine's sampler.
- **Separated (same day): the cache is not the cause.** The same probe on TensorFold with the 16-bit cache (80K
  window, `tf-kv16-80k`, 15.4 min) also scored **15/20**, 5 run-outs, 0 wrong, median 1,761 tokens. The two
  TensorFold runs miss 4 of the same 5 problems; each solved one the other didn't (precalculus/902 for 4-bit,
  intermediate_algebra/582 for 16-bit), which is sampling noise.

| Problem | EXL3 4.0 | TF, 4-bit cache | TF, 16-bit cache |
|---|---|---|---|
| intermediate_algebra/2015 | 9,176 ✓ | run-out | run-out |
| intermediate_algebra/558 | 11,443 ✓ | run-out | run-out |
| geometry/686 | 1,489 ✓ | run-out | run-out |
| precalculus/902 | 8,359 ✓ | 12,588 ✓ | run-out |
| intermediate_algebra/582 | run-out | run-out | 10,133 ✓ |
| geometry/880 | run-out | run-out | run-out |

  So the 4-bit cache is free on this test; the gap to EXL3 (18 vs 15) is the weights (MLX 4-bit, group 64, vs EXL3
  4.0) or TensorFold's sampler, and both TensorFold runs think longer (mean ~6,000 tokens vs 4,162). With one
  sample per problem, 3 problems is still within luck (p = 0.25 each way). Telling weights from sampler apart would
  mean EXL3-format weights in TensorFold (not supported) or more samples per problem.

### EXL3 weights in TensorFold (2026-10-07)

TensorFold 0.3.6.3 also reads turboderp's EXL3 packs of this model (`turboderp/Qwen3.8-27B-exl3`, branches 4.00/3.50/3.00bpw,
6-bit head) with the same DFlash2 drafter. On the 3090, every request first crashed with an illegal memory access.
`compute-sanitizer` showed a 2-byte read at address 0 inside `linear_kernel<8, 2, 4>`, which is the bias of a
layer that has none. The source guards that load with `if (bias)`, but the 4-warp version of the kernel compiled for
sm86 reads it anyway. Layers with 8 warps are fine. The fix (`tensorfold-exl3-sm86-bias.patch`) passes zeros
instead of no bias. Adding zero changes no output. After the fix, every layer type passes at 1, 3, 8, 9 and 16 rows, and the model answers
normally.

Speed with the 4-bit cache, 64K window (`bench_tf.sh`, tokens/s, median of 3):

| Build | Loaded | Code, sampled | Code, greedy | Chat, sampled | Chat, greedy |
|---|---|---|---|---|---|
| MLX 4-bit (262K window) | 17.2 GB | 147 | 144 | 103 | 114 |
| EXL3 4.00bpw | 18.3 GB | 169 | 137 | 81 | 92 |
| EXL3 3.50bpw | 16.5 GB | 170 | 133 | 90 | 94 |
| EXL3 3.00bpw | 15.3 GB | 163 | 173 | 90 | 89 |

- **Fewer bits doesn't make EXL3 faster here.** Its decode is limited by unpacking the trellis code, not by reading
  memory, so 3.0 runs at the same speed as 4.0. EXL3 is a bit faster on code and about 15-20% slower on chat than the
  MLX build.
- What EXL3 3.0 offers is room and maybe quality: it scored 19/20 on the math probe under ExLlamaV3, and it loads 3 GB
  smaller than EXL3 4.0.

**A mixed 3/4-bit MLX build** (`rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX`):
- Bits are allocated by activation-weighted error: 257 layers at 3-bit, 9 at 2-bit, the rest at 4-bit, group 64.
- Its card shows it matching the 4-bit build on MMLU-Pro, GSM8K and HumanEval+.
- It loads and answers correctly, but runs at **8-13 tok/s**.
  - TensorFold's fast CUDA kernel takes only 4-bit/group-64 layers (`QLinear.fast`).
  - Every 2- and 3-bit layer goes to the generic Triton reader (`cuda/kernels/affine.py`), which is built for
    correctness, not speed.

A fast 3-bit kernel wouldn't buy much:

| | MLX 4-bit | Mixed 3.5 |
|---|---|---|
| Weights read per step (language model, no embedding) | 14.42 GB | 12.90 GB |
| Time to read them at 936 GB/s | 15.4 ms | 13.8 ms |

- Without drafts a token takes 27.6 ms (36.2 tok/s), so weight reading is ~56% of a step.
- 10.5% fewer bytes is at most ~6% faster without drafts, and less with them: verify steps add compute that the
  weight bits don't touch.
- Uniform 3-bit would read ~22% fewer bytes, ~10% faster at best, and its card reports a 6.6-point MMLU-Pro loss.

**Verdict: on this engine, weight bits are no longer the decode bottleneck.**
- Lower-bit weights buy memory (EXL3 3.0 saves 3 GB), not speed.
- The open question is quality, not speed: the math probe gap (TensorFold 15/20 vs EXL3-in-ExLlamaV3 18/20).
  It follows the engine, not the weights: see below.

### Math probe: EXL3 weights in TensorFold (2026-10-07)

`probe_exl3_pair.sh`: the same 20 problems, the same sampling (temperature 1.0, top-p 0.95, top-k 20, one try each,
16K cap), on TensorFold with the 4-bit cache and the DFlash2 drafts. The request's sampling settings reach the server
(`server/request_options.py`). ExLlamaV3 ran the same settings through its own sampler.

| Run | Engine | Weights | Correct | Run-outs | Wrong | Median tokens |
|---|---|---|---|---|---|---|
| `exl3-4.00bpw` | ExLlamaV3 | EXL3 4.0 | 18 | 2 | 0 | 985 |
| `exl3-3.00bpw` | ExLlamaV3 | EXL3 3.0 | 19 | 1 | 0 | 1,676 |
| `tf-kv16-80k` | TensorFold | MLX 4-bit | 15 | 5 | 0 | 1,761 |
| `tf-kv4-262k` | TensorFold | MLX 4-bit | 15 | 5 | 0 | 1,881 |
| `tf-exl3-4.00` | TensorFold | EXL3 4.0 | 15 | 4 | 1 | 2,447 |
| `tf-exl3-3.00` | TensorFold | EXL3 3.0 | 17 | 2 | 1 | 2,600 |

- **The same EXL3 4.0 weights score 18 in ExLlamaV3 and 15 in TensorFold.** The MLX weight files are not the
  cause. The gap follows the engine.
- All six runs agree on the 14 easy problems. The difference is in six hard ones: 902, 2015, 582, 686, 880 and 558.
  - ExLlamaV3 solved 9 of 12 of those (75%) and TensorFold 7 of 24 (29%), Fisher p ≈ 0.014.
  - TensorFold loses mostly to run-outs: it thinks to the 16K cap.
- TensorFold also thinks longer on the problems it does solve: median 1.8-2.6K tokens against 1.0-1.7K.
- **Suspect: sampling with drafts.** At temperature 1.0, a draft-and-verify engine stays faithful only if it accepts
  draft tokens with the exact rejection rule over the top-k/top-p distribution. A rule that leans greedy or skips the
  truncation would shift what the model writes. Longer, looping thinking fits that.
  - The test: the same probe with drafts off (the request's `draft` field, or no drafter). That's about 50 minutes
    at 36 tok/s.
  - If it returns to ~18, the draft sampler is the bug. If it stays at ~15, look at the base sampler or the
    numerics.
- 3 bits held up again (17 vs 15 at 4.0 on this engine; 19 vs 18 in ExLlamaV3). With one try per problem, that is
  noise, not a 3-bit advantage.

### Next: where more room comes from

| Change | Frees | Then |
|---|---|---|
| ~~8-bit cache~~ | done: 144K window, see above | |
| ~~4-bit cache with the 32-wide rotation~~ | done: 262K window, see above | |
| Embedding table to system RAM | ~0.7 GB | not needed now: 262K is the model's maximum |
| Mixed 3/4-bit weights (EXL3 3.0 held 19/20 on the math probe) | ~3–4 GB | faster decoding: every token reads all the weights |

Each further step gets the same checks: `kv_kl.py` plus the 20-problem math probe.
