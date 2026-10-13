---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - Vontra/Qwen3.8-27B-MLX-4bit
  - z-lab/Qwen3.8-27B-DFlash2
pipeline_tag: text-generation
tags:
  - rtx-3090
  - cuda
  - speculative-decoding
  - dflash2
  - tensorfold
  - 4-bit
  - long-context
  - m3tis
---

# Qwen3.8-27B · M3TiS · RTX 3090

**Qwen3.8-27B at ~210 tokens/s for code and ~145 for chat, with the full 262K-token context, on one 24 GB RTX 3090.**

M3TiS is a series of builds tuned for the RTX 3090. This first release contains no new weights. It is a
server build of [TensorFold](https://github.com/machinegenieorg/TensorFold)'s native Zig engine with 3090-specific
changes, plus the compiled 3090 kernels and a one-line launcher. It runs these existing weights:

- [`Vontra/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit): the model, 4-bit (~15 GB)
- [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2): a small draft model that
  guesses several tokens ahead. The 27B checks every guess, so the output is the same as without it.

## Quick start

```bash
pip install -U huggingface_hub            # for `hf download`
hf download CanadaApollo6/Qwen3.8-27B-M3TiS-3090 --local-dir m3tis
cd m3tis && chmod +x m3tis-serve.sh bin/tensorfold-native
./m3tis-serve.sh                          # first run downloads ~19 GB of weights
```

It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (`/chat/completions`, `/completions`, `/models`).
`PORT=…` and `CONTEXT=…` change the port and window, and extra flags pass through to `tensorfold-native serve`
(`--no-thinking`, `--temperature`, `--api-key`, …). Thinking is on by default, as in Qwen's chat template.

## Measured on one RTX 3090 (2026-10-13)

The 262K window, 4-bit context store, 256-token replies, median of 3, single user:

| Tokens/s | Greedy | Sampled (T=1.0) |
|---|---|---|
| Code (write a Fibonacci function) | **208** | 202 |
| Chat (explain GPUs, no thinking) | **147** | 135 |
| Plain decoding, no draft model (for reference) | 45 | |

- **First token:** 31–57 ms on short prompts.
- **Long prompts:** a 192K-token prompt read in 7 minutes (449 tokens/s), then recalled a code planted in its
  middle. Peak card use was 19.6 GB.
- **Measured on a desktop in use:** a video was playing on the same card. On an idle machine the best run so far was
  235 code / 160 chat, so expect roughly 10% more with nothing else on the GPU.

## Is it still the same model?

- **The draft model changes nothing.** Every output token is checked by the 27B. Our test suite gives the same tokens
  with and without drafts, and the same tokens as TensorFold 0.6.6's Python engine (one user and three at once).
  With the 4-bit store, drafted output matches plain decoding with that store.
- **The 4-bit context store is the one lossy part.** Past tokens are kept in 4 bits after a fixed rotation, which
  is what fits 262K in 24 GB. Measured against a 16-bit store, next-token predictions drift by KL 0.002–0.006,
  and perplexity stays within 0.5%. To use a 16-bit store instead, set `TF_QWEN27_KV_BITS=16` and a window of
  about 32K.
- The 27B weights are Vontra's 4-bit MLX-format quantization, used as is.

## Requirements

- An NVIDIA RTX 3090 (sm_86, 24 GB). Other sm_86 cards with 24 GB should work but are untested. The desktop can use up
  to ~2 GB of the card.
- Linux x86-64 with glibc 2.28 or newer and an AVX2 CPU (Intel 2013+, AMD 2015+).
- An NVIDIA driver with CUDA 13 support (R580 or newer, expected). Tested on 615.71.09.
- No CUDA toolkit, Python or PyTorch: the binary is self-contained, and the kernels are precompiled.

## What's different from upstream TensorFold

Built from TensorFold PR [#608](https://github.com/machinegenieorg/TensorFold/pull/608)'s Zig engine with these
changes (source: SOURCE.md):

- RTX 3090 support: matmul fallbacks for pre-Hopper cards and 16-byte aligned staging (upstream PR
  [#3](https://github.com/machinegenieorg/TensorFold/pull/3)), and work buffers sized for one user.
- A 3090-tuned 4-bit matmul kernel.
- The 4-bit context store with a CUDA attention kernel for long contexts.
- Weight tensors packed into shared blocks: one allocation per tensor wasted 1.1 GB.
- Past 128K: 1024-token prompt chunks, with the word table kept in host RAM.
- Short prompts use the decode kernels, for a fast first token.

## Files

| Path | What |
|---|---|
| `m3tis-serve.sh` | Downloads the weights, sets the 4-bit store, starts the server |
| `bin/tensorfold-native` | The server (7 MB, static except libc; loads `libcuda` at run time) |
| `share/tensorfold/cuda/sm86/` | Precompiled sm_86 kernels (1.7 MB) |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/` | TensorFold's licenses (Apache-2.0) |

## Credits and license

Apache-2.0, as are all the parts: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team;
the 4-bit weights by Vontra; DFlash2 by z-lab / Inco AI; TensorFold by its authors. M3TiS is an independent build
by [CanadaApollo6](https://huggingface.co/CanadaApollo6) and is not affiliated with any of them.
