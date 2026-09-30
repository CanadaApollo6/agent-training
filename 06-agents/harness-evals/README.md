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

In the replay, every prompt is prefilled from scratch. In the live eval, most turns continued from state in the cache:
rows the model had generated itself, restored prefixes and RAM swap-ins. The engine is meant to make those exactly equal
to a fresh prefill; the unit tests check that on a tiny model, not on R1. The control's interval still includes 1,
so this is a lead, not a finding.

The decisive check is greedy and bit-exact. Run a multi-turn conversation through the live cache path, then send the
same last prompt cold. Any divergence in tokens is a bug in the serving path.

## So far

- **Harness choice:** pi is the safer default for self-hosted models today. prime_agent's continual-harness features
  cost tokens, and its REPL and daemon add failure modes. Its image tool assumes a vision-capable endpoint. None of
  this says what a model trained on prime_agent would do, which is Prime's actual claim.
- **Model:** 35B-A3B over the 9B for agentic work (11 vs 6 under pi).
- **Local build:** R1 (all experts 3-bit) is the local build. It is within noise of Q8 on solves, where R2 is 14
  points behind. It writes ~1.7× Q8's tokens on the same solved tasks, and whether that comes from the bits or from
  the engine is still open.

## Next

- Greedy multi-turn check on R1: live cache/swap path vs a cold prefill, token for token. If they match, R1's extra
  length comes from compounding over whole runs, not from wordier turns or the engine.
- Find what hangs git-multibranch and rstan-to-pystan for the full hour on the local runs.
- More tasks or several attempts each before calling a harness winner. pi's lead is suggestive, not significant.
- Fix the budget in tokens rather than wall-clock, or serve with vLLM, which batches MoE decode far better than
  llama.cpp, so the time limit stops being a hidden token cap.
- The funnel ends at DeepSWE (`primeintellect/deep-swe`) for finalists: the 35B under pi first.
