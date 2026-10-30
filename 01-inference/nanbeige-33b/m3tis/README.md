---
license: apache-2.0
base_model:
  - Nanbeige/Nanbeige4.3-33B
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - nanbeige
  - exl3
  - speculative-decoding
  - dflash
  - cuda
  - rtx-3090
  - tool-calling
  - tensorfold
  - m3tis
---

# M3TiS-33B

**Nanbeige4.3-33B with its own draft head, on one 24 GB card: 73 tokens/s on code, 2.7× llama.cpp on the same card,
with an 82K-token window.**

[Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) is one of the strongest open agent models near
30B, and on a home GPU it is slow. M3TiS-33B is our rebuild of how it runs, so it feels like a different model to use.
Everything is in this repo: 16.8 GB of weights, a 4 GB draft head and a 44 MB server.

- **A draft head trained for this model.** A small 5-layer network guesses the next 8 tokens, and the 33B checks them
  all in one pass. We trained it on 36,000 of Nanbeige's own replies. On code it lands about 5 tokens per pass.
- **Weights quantized on Nanbeige's own text.** EXL3 at 3.5 bits per weight, calibrated on its chat and agent output.
  On 19 Terminal-Bench 2 tasks it solved as many as the full bf16 model (below).
- **A server built for both.** A port of [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine with
  kernels for EXL3, the draft head, and a 4-bit context store that fits 82K tokens next to the weights.

Measured on an NVIDIA A10 (below), the same Ampere generation as the RTX 3090 with 24 GB, but about two-thirds of the
3090's memory bandwidth. RTX 3090 measurements will be added here.

## Quick start

```bash
pip install -U huggingface_hub            # for `hf download`
hf download CanadaApollo6/M3TiS-33B --local-dir m3tis-33b
cd m3tis-33b && chmod +x m3tis-serve.sh bin/tensorfold-native
./m3tis-serve.sh
```

It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (`/chat/completions`, `/completions`, `/models`).
`PORT=…` and `CONTEXT=…` change the port and window, and extra flags pass through to `tensorfold-native serve`
(`--temperature 0`, `--no-thinking`, `--api-key`, …).

Requests with no sampling settings use Nanbeige's recommended temperature 1.0 and top_p 0.95. Both sampled and greedy
(`"temperature": 0`) replies use the draft head. Thinking is on by default, as in Nanbeige's chat template. Replies
reason in `reasoning_content` before the answer in `content`, and `max_tokens` counts both.

## Speed

Measured on 2026-10-30 on an NVIDIA A10 (24 GB, sm_86) with the files in this repo, started with `./m3tis-serve.sh`.
That is a single user, the 81,920-token window, and the 4-bit context store. Short prompts use 512-token replies. Each
figure is the median of 3 runs after a warm-up.

The comparison is [llama.cpp](https://github.com/ggml-org/llama.cpp) (commit 792fcf4, built for sm_86) on the same
card, running bartowski's [Q3_K_M GGUF](https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF)
(14.8 GiB, the closest size to these weights) with flash attention, measured by `llama-bench`. At 16K and 61K
it uses a 4-bit (q4_0) cache, like this build.

| Reply speed, tokens/s | M3TiS-33B, greedy | M3TiS-33B, Nanbeige's sampling | llama.cpp Q3_K_M |
|---|---|---|---|
| Code (a Fibonacci function, with thinking) | **73** | 61 | 27 |
| Chat (GPU matrix multiplication, explained) | **40** | 34 | 27 |
| Tool call (weather in two cities, 89 tokens) | **66** | 61 | 27 |
| After a 16K-token prompt (a summary reply) | **22** | | 18 |
| After a 61K-token prompt (a summary reply) | **12** | | 9 |

The draft head is where the speed comes from. Without it, this server decodes at 17 tokens/s, slower than llama.cpp.
Its gain is largest on code, where its guesses land most often, and it holds up at long context.

**Prompt reading is slower than llama.cpp,** about 2× on long prompts.

| Reading a prompt | M3TiS-33B | llama.cpp Q3_K_M |
|---|---|---|
| 16K tokens | 48 s | not measured |
| 61K tokens | 273 s | 138 s |
| The next turn of the same conversation | 8 s after 16K, 14 s after 61K: only the part that changed is read | |

The first token of a short prompt arrives in 0.17 s.

## Is it still the same model?

**The weights.** These are Nanbeige's weights at 3.5 bits. The quantization moves next-token predictions by a KL
divergence of 0.036 on Nanbeige's chat replies and 0.066 on its agent turns, against the bf16 original. To check
that this does not cost agent skill, we ran 19 Terminal-Bench 2 tasks, 3 attempts each, with the pi coding agent,
on exllamav3:

| Build | Tasks solved per attempt (of 19), mean of 3 |
|---|---|
| Nanbeige4.3-33B bf16 (vLLM) | 11.0 |
| **These weights (exllamav3)** | **11.3** (10, 12, 12) |

The difference is noise (p = 1.0). The 3.5-bit build used 1.21× the tokens per solved task. This server runs the same
weights as exllamav3 does. On a 128-token greedy reply it matched exllamav3 token for token, and their next-token
predictions differ by a KL of 0.00005.

**The draft head.** The 33B checks every guess before it keeps it, so drafting does not change what the model decides.
With the 16-bit context store, drafted greedy output matched plain greedy output token for token.

**The 4-bit context store is lossy.** It is what fits 82K tokens on the card. Against the 16-bit store, on nine long
conversations (20K, 60K and 100K tokens), it moves predictions by a mean KL of 0.013 to 0.16 on seven, and by 0.9 and
1.6 on two hard ones. That matches exllamav3's own 4-bit cache, which scores 1.07× ours on the same nine. For full
precision with shorter conversations, use `TF_LLAMA_KV_BITS=16 CONTEXT=16384 ./m3tis-serve.sh`.

## Memory

With the 81,920-token window, the card peaked at 20.4 GB: 15.1 GiB of weights and draft head, 4.6 GiB of context. The
server keeps 2 GiB free for a desktop. If other programs use the card, it says what is missing, and a smaller
`CONTEXT` fits. Each 1K tokens of window takes about 57 MB with the 4-bit store and 200 MB with the 16-bit one.

## Tool calls

The server turns Nanbeige's tool calls into OpenAI `tool_calls` with JSON arguments, in whole replies and streamed,
and keeps the markup out of `content`. The tool-call replies above finished with `finish_reason: "tool_calls"`.

## Limits

- **One user at a time** (`--parallel 1`). A second request waits for the first.
- **Prompt reading is about 2× slower than llama.cpp** (above).
- **Text only.**
- **Only sm_86 cards.** The kernels are compiled for the RTX 3090 / 3090 Ti, A10, A40, RTX A5000 and RTX A6000
  generation. Only the A10 is tested so far.

## Requirements

- An NVIDIA sm_86 card with 24 GB or more.
- Linux x86-64 with glibc 2.28 or newer and an AVX2 CPU (Intel 2013+, AMD 2015+).
- An NVIDIA driver with CUDA 13 support (R580 or newer). Data-center cards on an older driver can use NVIDIA's
  `cuda-compat-13-4` package: `LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat ./m3tis-serve.sh`. The A10 numbers above
  ran that way, on driver 570.
- No CUDA toolkit, Python or PyTorch. The server is self-contained, and its kernels are compiled in.

## Files

| Path | What |
|---|---|
| `m3tis-serve.sh` | Starts the server on the weights in this folder |
| `bin/tensorfold-native` | The server (44 MB, static except libc; loads `libcuda` at run time) |
| `config.json`, `model-*.safetensors`, `quantization_config.json` | The EXL3 3.5-bit weights |
| Tokenizer files, `generation_config.json` | Nanbeige's, unchanged |
| `draft/` | The DFlash2 draft head (bf16; the server packs it to 4 bits when it loads) |
| `LICENSE` | Apache-2.0, for the weights and the server |
| `NOTICE`, `SOURCE.md`, `THIRD_PARTY_NOTICES.md`, `LICENSES/` | Attribution, build source, and TensorFold's licenses |

## How it was made

Every step has its recipe and receipts in
[agent-training/01-inference/nanbeige-33b](https://github.com/CanadaApollo6/agent-training/tree/main/01-inference/nanbeige-33b).
In short:

- **The draft head.** 36,000 replies generated by Nanbeige4.3-33B, then 2 epochs with SpecForge plus a short cooldown.
  On H200s it accepts 2.9 to 4.0 tokens per pass greedy.
- **The quant.** exllamav3 1.6.5, 3.5 bits per weight with a 6-bit output head, calibrated on Nanbeige's own chat and
  agent text, which beat the stock calibration (0.066 vs 0.073 KL on agent text).
- **The server.** Checked on an A100 against exllamav3 at every stage, including predictions, greedy output,
  the OpenAI wire format, drafting, the 4-bit store, and prompt reading up to 100K tokens. The checks are in the
  source (SOURCE.md).

## Credits and license

The model is [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) by the Nanbeige team, under the
Apache License 2.0. The quantized weights and the draft head are CanadaApollo6's derived work under the same license.
The server is built from [TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0, by its authors.
M3TiS is an independent build by [CanadaApollo6](https://huggingface.co/CanadaApollo6), not affiliated with
Nanbeige or TensorFold.
