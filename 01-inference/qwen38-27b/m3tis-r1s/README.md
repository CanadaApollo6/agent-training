---
license: mit
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - rtx-3090
  - cuda
  - moe
  - 3-bit
  - mtp
  - speculative-decoding
  - tensorfold
  - long-context
  - m3tis
---

# Ornith-1.5-35B-A3B · M3TiS · RTX 3090

**Ornith 1.5 35B-A3B at ~545 tokens/s for code and ~400 for chat, with the full 262K-token context, on one 24 GB
RTX 3090.**

M3TiS is a series of builds tuned for the RTX 3090. This second release is a new quantization of
[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (we call it R1s) plus a server built from
[TensorFold](https://github.com/machinegenieorg/TensorFold)'s native Zig engine. Everything is in this repo: 15 GB
of weights and a 10 MB server.

- **The experts are 3-bit, the rest 4-bit.** All 40 layers' routed experts are 3-bit, which is what makes the model
  15 GB. The always-on parts (attention, linear attention, shared expert) are 4-bit, rounded with an importance-weighted
  search instead of plain rounding. That cut the damage measured against the original by 12% at no cost in size or
  speed (below).
- **Drafting uses Ornith's own built-in draft head (MTP),** so there's no second model to download. The 35B checks every
  guess, so the output is the same as without it.

## Quick start

```bash
pip install -U huggingface_hub            # for `hf download`
hf download CanadaApollo6/Ornith-1.5-35B-A3B-M3TiS-3090 --local-dir m3tis-ornith
cd m3tis-ornith && chmod +x m3tis-serve.sh bin/tensorfold-native
./m3tis-serve.sh
```

It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (`/chat/completions`, `/completions`, `/models`).
`PORT=…` and `CONTEXT=…` change the port and window, and extra flags pass through to `tensorfold-native serve`
(`--no-thinking`, `--temperature`, `--api-key`, …).

## Measured on one RTX 3090 (2026-10-13)

These numbers come from the files in this repo, started with `./m3tis-serve.sh`: the 262K window, a single user, and
256-token replies. Each figure is the median of 3 runs.

| Tokens/s | Greedy | Sampled (T=1.0) |
|---|---|---|
| Code (write a Fibonacci function) | **545** | 480 |
| Chat (explain GPUs, no thinking) | **403** | 367 |

| Long prompts | Read | Reply speed |
|---|---|---|
| 21K tokens | 3.7 s (5,590 tokens/s) | |
| 45K tokens | 9.8 s | **274** tokens/s drafted, 186 without drafts |
| 52K tokens | 12.2 s (4,300 tokens/s) | |
| 199K tokens | 95 s (2,080 tokens/s) | |

- **First token:** 17–25 ms on short prompts.
- **Recall:** every long prompt hid a number in its middle, and the model found it each time, including at 199K.
- **Memory:** the card peaked at 17.2 GB with the full 262K window, including ~1.3 GB used by the desktop. The server
  wants ~21.3 GiB free at the 262K window (weights, its working space and a 1 GiB reserve), or ~18.8 GiB at
  `CONTEXT=131072`. If other programs use the card, it says how much is missing and suggests a smaller window.
- **Measured on a desktop in use,** so expect a few percent more on an idle machine.

## Is it still the same model?

**The draft head and other users change nothing.** Every output token is checked by the 35B. One user, drafts off,
and three users at once all give the same replies. Short and long (14K, 40K, 45K) prompts give the same replies
with and without drafts.

**The quantization is the lossy part.** Here is how much it costs, measured in several ways:

- **Next-token predictions vs the original (bf16).** This is KL divergence over the full vocabulary. It was scored on
  287K positions the model wrote itself (thinking, answers and tool calls) in 57 recorded agent conversations.

  | Build | KL vs bf16 (lower is better) |
  |---|---|
  | Swapping kernels on bf16 itself (noise floor) | ~0.04 |
  | **This build (R1s)** | **0.229** |
  | Same experts with the always-on parts plainly rounded at 4-bit | 0.260 |
  | Ornith's MLX 4-bit, everything 4-bit (smaller sample: 6 conversations) | ~0.22 |

  Plainly rounding only the always-on parts costs about as much as the 3-bit experts. Our search recovers most of the
  gap to an all-4-bit build without its extra 4 GB.
- **Agent tasks.** Terminal-Bench 2 (20 tasks, of which 19 are scorable) ran under the pi harness, 3 attempts each.
  This build solved **10.0 of 19** on average, against **10.3** for Ornith's official Q8_0. The task-by-task
  difference is not significant (p = 1.0). Its solved runs used 1.28× Q8_0's tokens (95% CI 0.92–1.86).
- **Under Claude Code.** On 10 DeepSWE tasks with one attempt each, it solved 2, against 1 for bf16. On all 89
  Terminal-Bench 2 tasks it solved 33 under Claude Code, 35 under prime_agent and 40 under pi.

The agent evals ran these weights on TensorFold's Python engine. The Zig server in this repo runs the same weights
with a different summation order. On our test prompts, greedy replies drift apart after 13 to 63 tokens. That kind
of difference measures at the ~0.04 noise floor above.

## Requirements

- An NVIDIA RTX 3090 (sm_86, 24 GB). Other sm_86 cards with 24 GB should work but are untested.
- Linux x86-64 with glibc 2.28 or newer and an AVX2 CPU (Intel 2013+, AMD 2015+).
- An NVIDIA driver with CUDA 13 support (R580 or newer, expected). Tested on 615.71.09.
- No CUDA toolkit, Python or PyTorch: the server is self-contained, and its kernels are compiled in.

## How the weights were made

The starting point is the bf16 [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). The
weights use MLX's affine format with group size 64:

| Part | Bits | How |
|---|---|---|
| Routed experts (gate, up, down; all 40 layers) | 3 | Importance-weighted search |
| Attention, linear attention, shared expert | 4 | Importance-weighted search |
| Embeddings, lm_head, MTP draft head | 4 | From [Ornith's MLX 4-bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit), unchanged |
| Routers | 8 | From Ornith's MLX 4-bit, unchanged |
| Norms, convolutions | 16 | Unchanged |

The search picks each group's scale and offset to minimize error, weighted by how much each input channel is used on
real text. Those weights come from [bartowski](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF)'s importance
matrix. `quantize-report.json` lists each expert tensor's error next to plain rounding. The search's error is about a
third lower (median 0.69×, best 0.47×).

## What's different from upstream TensorFold

The server is built from TensorFold PR [#608](https://github.com/machinegenieorg/TensorFold/pull/608)'s Zig engine. It
carries the [first M3TiS release](https://huggingface.co/CanadaApollo6/Qwen3.8-27B-M3TiS-3090)'s RTX 3090 changes
plus these, which add this model family to the engine (source: SOURCE.md):

- 3-bit and 4-bit expert kernels, with the MTP draft head drafting 3 tokens ahead.
- CUDA graphs for one-user rounds, and a faster word pick and expert router for long prompts.
- A sliding window for the draft head's own memory, so drafting stays cheap at long contexts.

## Files

| Path | What |
|---|---|
| `m3tis-serve.sh` | Starts the server on the weights in this folder |
| `bin/tensorfold-native` | The server (10 MB, static except libc; loads `libcuda` at run time) |
| `config.json`, `model-*.safetensors`, `mtp-4bit.safetensors`, tokenizer files | The weights (15 GB, MLX affine format) |
| `quantize-report.json` | Each expert tensor's error, ours vs plain rounding |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/` | TensorFold's licenses for the server (Apache-2.0) |

## Credits and license

The weights are MIT, as is [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) by Ornith AI.
The server is Apache-2.0, as is TensorFold by its authors. The importance matrix is by bartowski. M3TiS is an
independent build by [CanadaApollo6](https://huggingface.co/CanadaApollo6) and is not affiliated with any of them.
