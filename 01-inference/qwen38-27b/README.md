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
