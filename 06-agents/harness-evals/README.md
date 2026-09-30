# Harness evals: pi vs prime_agent

Does the agent harness change what a self-hosted model can do? Prime Intellect's
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 runs the same model and tasks under different harnesses.
- **pi** gives the model one tool, a bash shell.
- **prime_agent** gives it a persistent Python REPL. It also layers Prime's "continual harness" machinery on top: a
  running digest of session memories, plus an automatic review every 25 turns that decides whether to write lessons
  down (`/refine`).

This is the harness Prime says models should be trained against.

Setup:
- **Tasks:** 20 Terminal-Bench 2 tasks (`tasks.txt`, a seed-0 sample), one attempt each. Each task gets its own Prime
  sandbox VM.
- **Limits:** 60 turns and 30 minutes per task.
- **Model:** served by llama-server (PrismML's llama.cpp build) on a rented A6000 through an SSH tunnel. It uses the
  model card's sampling settings and 8-bit KV, with 8 slots shared by both harnesses running at once (4 tasks each).
- **Scripts:**
  - `./run_pilot.sh <pi|prime_agent> <label>` runs one harness.
  - `uv run summarize.py outputs/primeintellect/<run>...` prints the per-task table and a paired comparison.

## Ornith 1.5 9B, Q8_0

| Harness | Solved | Finished on its own | Hit 30 min | Hit 60 turns | Model time | Tool time | Output tokens |
|---|---|---|---|---|---|---|---|
| pi | 6/20 | 8 | 8 | 3 | 5.9 h | 0.7 h | 374K |
| prime_agent | 6/20 | 8 | 11 | 0 | 6.2 h | 1.3 h | 401K |

One task (qemu-startup) failed to start its sandbox under both harnesses.

Paired on the same tasks:
- pi solved compile-compcert and financial-document-processor, which prime_agent missed.
- prime_agent solved mteb-retrieve and pypi-server, which pi missed.
- Both solved git-leak-recovery, kv-store-grpc, modernize-scientific-stack and nginx-request-logging.
- 2 against 2 gives exact McNemar p = 1.

**No difference between the harnesses at this size.** The tie itself matters less than why so many runs never finished:

- **The 30-minute limit was really a token budget.** Each stream decoded at about 17 tokens/s (8 streams sharing one
  A6000), and model time was 83–90% of each harness's total. So 30 minutes buys roughly 30K output tokens. Apart
  from the three stuck runs below, every timed-out run used 25–32K. The pilot measures "what can the 9B solve in
  ~30K tokens," and the harness is second-order.
  - Consequence: wall-clock limits make results depend on server load and serving speed. The next runs should either
    fix a token budget or report the tokens/s alongside the score.
- **prime_agent's tool is easier to hang.** In two runs the last tool call blocked until the time limit:
  - kv-store-grpc: the model started its server with output piped back, then read the pipe to the end. A running
    server never ends the stream (1,367 s lost). The task still passed, since the server was up.
  - financial-document-processor: an OCR call hung (826 s).

  A bash tool with per-command timeouts is harder to wedge like this. prime_agent's `tool_timeout` (600 s) didn't
  catch either one.
- **prime_agent spends model time on itself.** The every-25-turns review is its own model call. On protein-assembly,
  the run ended inside one (616 s). That overhead only pays off if the model uses what gets written down, which is
  what harness-targeted training is meant to teach.

## Ornith 1.5 35B-A3B, Q8_0

Same settings, same server setup.

| Harness | Solved | Finished on its own | Hit 30 min | Hit 60 turns | Errors | Model time | Tool time | Output tokens |
|---|---|---|---|---|---|---|---|---|
| pi | 11/20 | 12 | 5 | 2 | 1 | 4.1 h | 0.6 h | 230K |
| prime_agent | 7/20 | 9 | 8 | 0 | 3 | 5.5 h | 1.1 h | 349K |

Paired on the same tasks:
- pi solved build-pov-ray, financial-document-processor, hf-model-inference, mteb-retrieve and
  portfolio-optimization, which prime_agent missed.
- prime_agent solved git-multibranch, which pi missed.
- 5 against 1 gives p = 0.22.

**The bigger model nearly doubles pi's score (6 → 11), and pi now leads, though not significantly.** Some of
prime_agent's deficit is plumbing, not the model:
- **financial-document-processor:** prime_agent tried to look at an invoice image. The server had no vision
  projector, so the request failed with an error. pi read the same images with OCR from the shell and solved it.
- **compile-compcert:** the prime_agent daemon crashed mid-run.
- **hf-model-inference:** the REPL sat in a wait loop on a pip install for 26 minutes. pi solved it in 10 calls.
- **pypi-server:** the model finished and wrote its summary, but the session didn't return until the time limit
  (21 minutes of tool time). It still passed.

Setting aside the three that failed on plumbing, pi still has three solves prime_agent doesn't (build-pov-ray,
mteb-retrieve, portfolio-optimization), against one the other way. pi also got there on 34% fewer tokens.

**The MoE didn't decode faster under load.** It runs at 104 tokens/s for a single stream, but with 8 streams sharing
the card, each got about 15 tokens/s, no better than the 9B. Each stream routes its tokens to different experts, so
batching them saves little weight reading. The 30-minute limit was the same ~30K-token budget for both models.

## Ornith 35B-A3B builds under pi: Q8_0, Q4_K_M, R2, R1

The same 20 tasks, three attempts per build, pi, 60 turns, 60 minutes, 4 tasks at a time, sampling at temperature
1.0 / top_k 20 / top_p 0.95. qemu-startup fails before the agent starts on every build (HarnessError), so the scores
are out of 19.
- **Q8_0 and Q4_K_M:** llama.cpp on rented pods.
- **R2 and R1:** the low-bit builds on the 3090 with patched TensorFold.
  - R2: gate/up experts 2-bit, down 3-bit, 13 GB.
  - R1: all experts 3-bit, 15 GB.

`compare_builds.py` makes every table here from the saved runs. Its paired test matches builds task by task: it
permutes the build labels within each task and asks whether the gap in per-task solve rates beats what swapping
labels gives.

| Build | Solved per attempt (of 19) | Mean | vs Q8, task by task | Median output tokens a task |
|---|---|---|---|---|
| Q8_0 | 9, 10, 12 | 10.3 (54%) | — | 9.2K |
| Q4_K_M | 12, 10, 11 | 11.0 (58%) | +3.5 points, p = 0.80 | 11.9K |
| R2 | 7, 7, 9 | 7.7 (40%) | −14 points (worse on 7 tasks, better on 2), p = 0.10 | 17.2K |
| R1 | 12, 8, 9 | 9.7 (51%) | −3.5 points (worse on 5, better on 4), p = 0.82 | 18.5K |

**Going from 2 to 3 bits on gate/up bought back most of R2's loss.** R1 is within noise of Q8. On nginx-request-logging
R2 went 0 for 3, writing a log format that runs fields together, and R1 went 3 for 3. Three attempts over 19 tasks can't
show a gap much smaller than ~15 points, so "within noise" means R1 has no large deficit, not that it's equal.

**R1 still writes much more than Q8.** On tasks both builds solved, compare the median output tokens of the solved
runs, then take the geometric mean across tasks:

| Build | Solved-run tokens vs Q8 | 95% CI | Longer on | Output tokens per call |
|---|---|---|---|---|
| Q4_K_M | 1.18× | 0.93–1.55 | 7 of 13 tasks | 382 |
| R2 | 1.84× | 1.21–2.78 | 9 of 11 | 495 |
| R1 | 1.67× | 1.21–2.36 | 10 of 12 | 477 |

Q8 writes 373 tokens per call. The extra bit fixed accuracy but not length. On the 3090 that is ~1.7× the wall-clock
per solved task. custom-memory-heap-crash is the clearest case:
- Q8 solved it 3 of 3 in 10–33K tokens.
- R1 solved it once in 68K. The other two attempts ran into the turn limit at 50K and 109K.

**How the failures end:**

| Build | Gave a wrong answer | Hit 60 turns | Hit 60 min |
|---|---|---|---|
| Q8_0 | 14 | 12 | 0 |
| Q4_K_M | 13 | 11 | 0 |
| R2 | 19 | 11 | 4 |
| R1 | 10 | 14 | 4 |

R2 mostly fails by doing the wrong thing. R1 mostly fails by running out of room.

The timeouts are not long thinking. They hit only git-multibranch and rstan-to-pystan, and only on the local builds.
In each, the model's own calls took at most 13 of the 60 minutes. The rest went outside model calls, to a tool command
or sandbox step that never returned. That is still worth a look, but it's not about length.

**Engine vs bits: what's ruled out.** Both long builds run on TensorFold, and both short ones on llama.cpp. Three checks:
- **Sampling is the same.** Both servers default to temperature 1.0, top_k 20, top_p 0.95 with no min_p, and pi sends
  only `max_tokens`.
- **Prompts are the same** (`length_replay.py prompts`, no GPU). Both engines render earlier turns' thinking into the
  history:
  - TensorFold's recorded prompt sizes match renders that keep the thinking.
  - llama-server reports only uncached prompt tokens. Its cache hits match the thinking kept in 38 of 40 turns, and
    miss by ~10K tokens every time if it were dropped.
- **Single-turn reasoning doesn't get longer on TensorFold.** On the reasoning probe, matched on problems both builds
  solved:

  | Build | Engine | Length vs Q8 | 95% CI |
  |---|---|---|---|
  | MLX 4-bit | TensorFold | 1.07× | 0.97–1.19 |
  | R1 | TensorFold | 1.09× | 0.89–1.30 |
  | Q4_K_M | llama.cpp | 1.17× | 1.05–1.30 |
  | experts 2-bit | llama.cpp | 1.32× | 1.20–1.47 |

  So the 1.7× only shows up in agent runs.

**Replay at fixed context** (`length_replay.py replay`, 13 minutes on the 3090):
- 38 recorded turns from Q8's runs and 38 from R1's, 2 per task, prompts up to 32K tokens.
- Each was sent again to R1 twice, and only the next turn was generated.

| Prefixes from | R1's replayed turn vs the recorded turn | 95% CI |
|---|---|---|
| Q8's runs | 1.02× | — |
| R1's own runs (control) | 0.85× | 0.66–1.11 |
| Q8 relative to the control | 1.20× | 0.91–1.67 |

- **At Q8's contexts, R1 writes about what Q8 wrote.** It's 1.02× raw, and 1.20× measured against the control, which
  is within noise.
- **At R1's own contexts, R1 now writes less than it did in the live eval,** at 0.85×. The median turn is 266 tokens
  replayed vs 362 recorded. The prompts match token for token (median difference 0).

In the replay, every prompt is prefilled from scratch. In the live eval, most turns continued from state in the cache.
**Live cache vs cold prefill, bit for bit** (`cache_check.py`, `run_cache_check.sh`, under 2 minutes of GPU):
- 6 of R1's eval conversations were driven through the running server for 6 turns each, taking turns so they pushed
  each other out.
- 3 of the conversations were greedy and 3 sampled with fixed seeds.
- 30 of the 36 requests resumed from the RAM swap, bringing back 127–3,182 tokens each, on top of prefixes kept on
  the GPU.
- Then every prompt was generated again in-process from an empty cache.

**36 of 36 replies came out identical, token for token, and every prompt matched too.** The serving path doesn't
change R1's output, so the 0.85× control was sampling noise.

What this covers: contexts up to ~6K tokens, requests one at a time. The eval ran contexts to 50K+ with 4 agents
queued. The mechanisms are the same, but those sizes weren't run.

**Where R1's extra length comes from.**
- Not sampling, prompts or the engine.
- Not wordier turns at the same context: 1.02× Q8's length at Q8's own contexts.
- It comes from where R1's runs go. Its own runs reach states where it writes more per turn (477 vs 373 tokens a
  call), and it takes more of them: detours, error recovery, and running out of turns.
- That points to decision quality from the lower bits. Compressing the model's thinking wouldn't address it, since
  R1 thinks about as much as Q8 does when they face the same context.

## R1s: the always-on path from the imatrix search

The KL harness (`05-compression/ornith/kl/`) found that R1 loses most of its fidelity in the always-on path (attention,
linear attention, shared expert), not in the experts. R1s is R1 with those matrices re-quantized at the same 4 bits by
the imatrix-weighted search. Only the numbers change; size, format and kernels stay the same. It cut KL against bf16
from 0.260 to 0.229, and on the 3090 it decodes at 95–104% of R1's speed. Same eval as above, on the 3090, three
attempts (attempt 2 was re-run after the server exited mid-run with no error in its log). `compare_builds.py` makes these
tables:

| Build | Solved per attempt (of 19) | Mean | vs Q8, task by task | Median output tokens a task |
|---|---|---|---|---|
| Q8_0 | 9, 10, 12 | 10.3 (54%) | — | 9.2K |
| R1 | 12, 8, 9 | 9.7 (51%) | −3.5 points, p = 0.82 | 18.5K |
| R1s | 11, 10, 9 | 10.0 (53%) | −1.8 points (worse on 3 tasks, better on 4), p = 1.0 | 12.6K |

| Build | Solved-run tokens vs Q8 | 95% CI | Longer on | Output tokens per call (median run) |
|---|---|---|---|---|
| R1 | 1.67× | 1.21–2.35 | 10 of 12 tasks | 478 |
| R1s | 1.28× | 0.92–1.86 | 7 of 13 | 406 |

(Q8 is 374 tokens per call on the same measure. The table above says 373 because of rounding.)

| Build | Gave a wrong answer | Hit 60 turns | Hit 60 min |
|---|---|---|---|
| R1 | 10 | 14 | 4 |
| R1s | 10 | 14 | 3 |

- **Solves hold.** At 10.0 of 19, R1s is within noise of Q8 and R1, as predicted (~10).
- **Most of R1's extra length is gone.** Solved runs went from 1.67× Q8's tokens to 1.28×, and that CI now includes
  1.0. Tokens per call fell from 478 to 406. This is better than predicted (~1.5×).
- **That fits the diagnosis above.** R1's length came from where its runs went, which is decision quality, not wordier
  turns. Fixing the always-on path is a fidelity fix, and it shortened the runs without touching the thinking.
- **On the 3090, a solved task costs ~20% less time than on R1.** It needs 0.77× the tokens at 95–104% of the decode
  speed.
- **The hour-long hangs are still there.** 5 runs hit 60 minutes: custom-memory-heap-crash twice, git-multibranch
  twice and rstan-to-pystan once. Two of them still passed the verifier, so the model had finished and something after
  it never returned.
- **custom-memory-heap-crash is the task that got worse:** Q8 solved it 3 of 3, R1 1 of 3 and R1s 0 of 3. It's the
  one to examine first.

## So far

- **Harness choice:** pi is the safer default for self-hosted models today. prime_agent's continual-harness features
  cost tokens, and its REPL and daemon add failure modes. Its image tool assumes a vision-capable endpoint. None of
  this says what a model trained on prime_agent would do, which is Prime's actual claim.
- **Model:** 35B-A3B over the 9B for agentic work (11 vs 6 under pi).
- **Local build:** R1s (all experts 3-bit, imatrix-searched 4-bit always-on path, 15 GB) is the local build. It is
  within noise of Q8 on solves (10.0 vs 10.3 of 19). It writes 1.28× Q8's tokens on the same solved tasks, down from
  R1's 1.67×; the engine was ruled out, and fidelity closed most of the gap.

## Next

- Untrained harness baselines, one model per GPU: Ornith 35B, Qwen3.8 27B and Ornith 9B at Q8, each under pi and
  prime_agent. They run on TB2 (3 attempts), DeepSWE (10 tasks) and Terminal-Bench 4 (20 tasks). This is the "before"
  for harness training.
- Find what hangs git-multibranch, rstan-to-pystan and custom-memory-heap-crash for the full hour on the local runs.
- More tasks or several attempts each before calling a harness winner. pi's lead is suggestive, not significant.
- Fix the budget in tokens rather than wall-clock, or serve with vLLM, which batches MoE decode far better than
  llama.cpp, so the time limit stops being a hidden token cap.
- The funnel ends at DeepSWE (`primeintellect/deep-swe`) for finalists: the 35B under pi first.
