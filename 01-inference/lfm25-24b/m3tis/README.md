---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model:
  - LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
tags:
  - liquid
  - lfm2.5
  - rtx-3090
  - cuda
  - moe
  - 4-bit
  - tool-calling
  - speculative-decoding
  - tensorfold
  - m3tis
---

# LFM2.5-24B-A2B · M3TiS · RTX 3090

**Liquid AI's LFM2.5 24B-A2B at ~330 tokens/s, with tool calls and its full 131K-token context, on one 24 GB
RTX 3090.**

M3TiS is a series of builds tuned for the RTX 3090. This third release runs
[LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) on a server built from
[TensorFold](https://github.com/machinegenieorg/TensorFold)'s native Zig engine, with kernels written for this model's
mix of short convolutions, attention and 64 small experts. Everything is in this repo: 13.7 GB of weights and a 10 MB
server.

- **The weights are Liquid's own 4-bit MLX build,** unchanged, plus one added file: the shared embedding and output
  table at 4 bits instead of 6, because the server's kernels read 4-bit tables.
- **Greedy replies draft from the text itself.** The model has no draft head, so the server guesses the next tokens by
  finding where the last few tokens appeared earlier, and the 24B checks every guess. This pays most when a reply
  repeats its prompt, like rewriting code: 347 instead of 277 tokens/s on a 14.7K-token rewrite.
- **It does agent work like Liquid's 8-bit build.** On 19 Terminal-Bench 2 tasks it solved 5.0 a run against 5.7 for
  Liquid's Q8_0 GGUF, within noise (below).

## Quick start

```bash
pip install -U huggingface_hub            # for `hf download`
hf download CanadaApollo6/LFM2.5-24B-A2B-M3TiS-3090 --local-dir m3tis-lfm
cd m3tis-lfm && chmod +x m3tis-serve.sh bin/tensorfold-native
./m3tis-serve.sh
```

It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (`/chat/completions`, `/completions`, `/models`).
`PORT=…` and `CONTEXT=…` change the port and window, and extra flags pass through to `tensorfold-native serve`
(`--temperature 0`, `--no-thinking`, `--api-key`, …).

Requests with no sampling settings use Liquid's: temperature 0.1, top_k 50, and a repetition penalty of 1.05 over the
last 64 tokens, applied as llama.cpp does. `"temperature": 0, "repetition_penalty": 1` in a request, or
`./m3tis-serve.sh --temperature 0 --repetition-penalty 1`, gives plain greedy replies, which are the ones that draft.
`"temperature": 0` alone keeps the penalty, and a penalized reply doesn't draft.

## Measured on one RTX 3090 (2026-10-16)

These numbers come from the files in this repo, started with `./m3tis-serve.sh`: the full 131K window, a single
user, 512-token replies. Each figure is the median of 3 runs after a warm-up.

| Tokens/s | Greedy (drafted) | Greedy with the penalty | Liquid's settings | T 1.0 |
|---|---|---|---|---|
| Code (write a Fibonacci function, with thinking) | **332** | 322 | 317 | 316 |
| Chat (explain GPU matrix multiplication) | **324** | 322 | 315 | 317 |
| Tool call (weather in two cities, 79 tokens) | **336** | 323 | 318 | 318 |

Greedy is `temperature 0, repetition_penalty 1`. Liquid's settings are the defaults above (T 0.1, top_k 50, penalty
1.05), and T 1.0 keeps the penalty.

| Long prompt (14.7K tokens, rewrite this code) | |
|---|---|
| Read | 7.0 s (2,100 tokens/s) |
| Reply speed | **347** tokens/s greedy (drafted), 277 with Liquid's settings |
| The next turn of the same conversation | 0.24 s to read, not 7.0 s: only the new tokens are read |

- **First token:** 34 ms on a short prompt.
- **For comparison,** llama.cpp's `llama-bench` on the same card gives 270 tokens/s for Liquid's QAD-Q4_0 GGUF and
  256 for Q4_K_M (128-token replies, empty context). llama.cpp reads prompts faster: 4,700–6,200 tokens/s on 2K-token
  batches.
- **Memory:** the card peaked at 18.1 GB with the full 131K window, including ~1.6 GB used by the desktop. If other
  programs use the card, the server says how much is missing and suggests a smaller `CONTEXT`.
- **Measured on a desktop in use,** so expect a few percent more on an idle machine.

## Tool calls

LFM2.5 writes tool calls as Python, like `[get_weather(city='Paris'), get_weather(city='Tokyo')]`, between
`<|tool_call_start|>` and `<|tool_call_end|>`. The server turns them into OpenAI `tool_calls` with JSON arguments, in
whole replies and streamed as deltas, and keeps the markup out of `content`. Several calls in one reply, calls with
prose around them, and JSON-style calls all work. A call to a tool the request didn't offer stays as text.

Agent loops are fast on later turns. LFM2.5's chat template drops the thinking of earlier turns, so each new turn
changes text the server already read. The server keeps snapshots of its state every 512 tokens and resumes from the
last one before the change, instead of re-reading the whole conversation. A resumed turn gives the same reply as a
cold read of the same prompt.

## Is it still the same model?

**Drafting changes no decision of the model, but greedy text can still differ.** Every guess is checked by the 24B.
But a check of several tokens at once runs different kernels than one token at a time, and their sums round a
little differently. At a near-tie that can pick the other token, so drafted and plain greedy replies can split
(after ~950 characters on our code prompt). With both paths forced to round alike, drafted output matched plain
greedy token for token on all four test prompts. Sampled replies don't draft.

**The quantization is the lossy part.** Next-token predictions vs the original bf16 model (KL divergence over the
full vocabulary, lower is better), on 443 positions of a code answer:

| Build | KL vs bf16 |
|---|---|
| Liquid's MLX 4-bit as published (6-bit embedding table), reference maths | 0.135 |
| Same, 4-bit embedding table (these files), reference maths | 0.152 |
| **These files on this server** | **0.14** (0.137–0.145 across the speed work) |

The 4-bit table costs about 0.017. The server's fp32 sums land slightly closer to bf16 than the reference maths, so
the files on this server score about like Liquid's published build. This is a small sample from one answer.

**Agent tasks: within noise of Liquid's 8-bit build.** We ran 19 Terminal-Bench 2 tasks with the pi
coding agent as the harness, 3 attempts each, up to 60 turns and 60 minutes per task, with both
builds at Liquid's settings. The reference was Liquid's Q8_0 GGUF on llama.cpp.

| Build | Solved per attempt (of 19) | Mean | Tokens per solved task vs Q8_0 |
|---|---|---|---|
| Liquid's Q8_0, llama.cpp | 7, 6, 4 | 5.7 (30%) | — |
| **These files on this server** | 3, 7, 5 | **5.0 (26%)** | 1.40× (95% CI 0.98–1.86) |

The difference in solves is noise (paired by task, p = 0.75). The 4-bit build takes more turns to reach the same
solves. This model is built for tool automation more than coding in a shell; Terminal-Bench is a hard test of the
latter. Method and tables: [agent-training/06-agents/harness-evals](https://github.com/CanadaApollo6/agent-training/tree/main/06-agents/harness-evals).

## Limits

- **One user at a time** (`--parallel 1`). A second request waits for the first.
- **`frequency_penalty` and `presence_penalty` are accepted but not applied.** `repetition_penalty` is applied.
- **Slower prompt reading than llama.cpp,** about 2–3× (above).
- **Text only.**

## Requirements

- An NVIDIA RTX 3090 (sm_86, 24 GB). Other sm_86 cards with 24 GB should work but are untested.
- Linux x86-64 with glibc 2.28 or newer and an AVX2 CPU (Intel 2013+, AMD 2015+).
- An NVIDIA driver with CUDA 13 support (R580 or newer, expected). Tested on 615.78.08.
- No CUDA toolkit, Python or PyTorch: the server is self-contained, and its kernels are compiled in.

## What's different from upstream TensorFold

The server is built from TensorFold PR [#608](https://github.com/machinegenieorg/TensorFold/pull/608)'s Zig engine
(closed unmerged on 2026-10-15; nothing here is in upstream TensorFold yet). It
carries the [first](https://huggingface.co/CanadaApollo6/Qwen3.8-27B-M3TiS-3090) and
[second](https://huggingface.co/CanadaApollo6/Ornith-1.5-35B-A3B-M3TiS-3090) M3TiS releases' changes plus these,
which add LFM2.5 to the engine (source: SOURCE.md):

- Kernels for the short convolution, the expert router, and attention over the filled part of the cache only.
- Expert matmuls where warps split the work and add up once at the end, and a CUDA graph per token step.
- Prompt-lookup drafting with a draft length that adapts to how often guesses are accepted.
- State snapshots so a conversation's next turn resumes instead of re-reading, and Liquid's tool-call format.
- Liquid's repetition penalty (1.05 over the last 64 tokens, from `generation_config.json`). Without it, long agent
  runs could loop.

## Files

| Path | What |
|---|---|
| `m3tis-serve.sh` | Starts the server on the weights in this folder |
| `bin/tensorfold-native` | The server (10 MB, static except libc; loads `libcuda` at run time) |
| `config.json`, `model-*.safetensors`, tokenizer files, `chat_template.jinja` | Liquid's 4-bit MLX weights, unchanged |
| `embed4.safetensors` | The embedding and output table at 4 bits (added; the shards' own 6-bit table, 0.2 GB, goes unused) |
| `LICENSE` | The LFM Open License v1.0, for the weights |
| `NOTICE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/` | Attribution, and TensorFold's licenses for the server (Apache-2.0) |

## Credits and license

The weights are [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) by Liquid AI, under the
**LFM Open License v1.0** (`LICENSE`). Its main condition: **commercial use is licensed only to companies with less
than US$10 million in annual revenue.** Non-commercial and research use has no revenue limit. Read the license
before you build on it. The server is Apache-2.0, as is TensorFold by its authors.
M3TiS is an independent build by [CanadaApollo6](https://huggingface.co/CanadaApollo6) and is not affiliated with
Liquid AI or TensorFold.
