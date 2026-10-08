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

## R1s on full TB2, with a bigger budget, and under Claude Code

These runs used Prime A6000 pods, with R1s served by TensorFold at 131K context.
- Full TB2 (89 tasks, 60 turns, 1 hour): pi 40, prime_agent 35.
- `sort_failures.py` sorted the unsolved runs into four causes:

  | Cause | prime_agent | pi |
  |---|---|---|
  | setup | 9 | 2 |
  | budget | 22 | 22 |
  | harness misuse | 1 | 0 |
  | task | 22 | 25 |

- Misuse is a tax rather than a cause of failure. 148 of 3,088 prime_agent calls went to tools that don't exist, mostly
  `bash`.

**Big-budget rerun.** Each harness's 22 budget failures were rerun once at 200 turns and 4 hours (`run_bigbudget.sh`).
- pi rescued 5, for 45/89. prime_agent rescued 7, for 42/89.
- **Most of prime_agent's extra rescues weren't budget.** A rescue counts as budget-driven only if the run went past 60
  turns or 1 hour. By that test, budget explains 4 rescues for each harness.
  - prime_agent's other 3 (break-filter-js-from-html, crack-7z-hash, kv-store-grpc) and pi's tune-mjcf finished
    within the old limits. They are second-attempt luck.
- **prime_agent overflowed the context twice** (make-doom-for-mips, path-tracing-reverse). Each time the prompt reached
  ~123K tokens plus a 16K reply against the 131K cache. The model can edit its own context, but it didn't do so before
  the server refused.
- pi stayed under ~111K throughout.

**Claude Code** (`run_claude_code.sh`). This tests whether Ornith 1.5's Claude Code training survived compression.
- Endpoints: TensorFold got `/v1/messages` (`claude-code.patch`). The bf16 original ran on vLLM behind
  `messages_proxy.py`, which applies the same translation.
- Setup: DeepSWE, 10 tasks, one attempt each, 100 turns, 90 minutes.
- **R1s solved 2 (fd, ofetch); bf16 solved 1 (fd), p = 1.0.** Compression cost nothing visible.
- This beats R1s's 0/16 on DeepSWE under pi and prime_agent: the model does best in the harness it was trained in.
- Budget bound both models:
  - every bf16 run hit the 100-turn cap;
  - 8 of 10 R1s runs hit the 90-minute timeout. R1s decoded 7–15 tokens/s per stream with 4 at once at 100K+
    context on one A6000, against vLLM's 40–49 on two.

**Claude Code on full TB2** (same limits as pi and prime_agent: 60 turns, 1 hour; two A6000 pods, 4 at once each).
- **R1s solved 33/89**, against pi 40 and prime_agent 35. The prediction was "north of 45".
- Against pi, Claude Code solved 5 tasks pi missed and missed 12 pi solved (McNemar p ≈ 0.14). Against prime_agent the
  split was 8 to 10.
- Only 2 tasks were solved by Claude Code alone (configure-git-webserver, tune-mjcf). All three harnesses together
  solved 50.
- One rollout (model-extraction-relu-logits) hung past its 1-hour timeout for 4.4 hours. It was killed and scored as
  a failure: the same hang as the local runs.
- The harness the model was trained in is not the best one on TB2. On DeepSWE it was (2 vs 0).

## Teacher bake-off: who solves what Ornith never has (2026-10-02)

`teacher_bakeoff.sh` runs hosted models through Prime Inference under prime_agent on the TB2 training tasks no harness
has ever solved with Ornith (`datagen/never.txt`, 27). Each teacher gets one attempt per task at the eval caps.

Ten of the 27 put an image in the opening prompt: chess-best-move, code-from-image, extract-moves-from-video,
gcode-to-text, install-windows-3-11, path-tracing, path-tracing-reverse, raman-fitting, sam-cell-seg and
video-processing. TensorFold serves Ornith text-only, so those can't become training data. The comparison uses the
other 17 (`datagen/never_text.txt`), at high reasoning effort with 32K tokens per call.

| teacher | solved / 17 | list-price cost |
|---|---|---|
| Qwen3.8-Max | 8 | ~$10 |
| DeepSeek V4.1 Flash | 7 | ~$1 |
| GLM-5.3 | 3 | ~$5 |

- Together DeepSeek and Qwen solved 10 tasks: caffe-cifar-10, circuit-fibsqrt, dna-insert,
  feal-linear-cryptanalysis, gpt2-codegolf, mteb-leaderboard, polyglot-rust-c, prove-plus-comm, reshard-c4-data and
  write-compressor. Every GLM solve was also a DeepSeek solve.
- Riel predicted GLM-5.3 would be the best solver. It came last under these caps:
  - Three rollouts spent a whole 32K-token call thinking and never acted.
  - Six ended with a confident "done, fully tested" that the hidden tests failed.
  - Z.ai's 88% on TB2.1 was measured under Claude Code with 6-hour runs and 64K tokens per turn.
- The first round, at the harness's default 16K tokens per call and GLM's default max effort, is invalid: GLM 1/27,
  DeepSeek 7/27. Eleven GLM rollouts ended on an empty, length-capped first reply, which prime_agent takes as done.
  Hosted thinking models need their effort and per-call cap set explicitly.
- These are single attempts on 17 tasks, so DeepSeek vs Qwen is a tie. GLM vs DeepSeek is 0 tasks to 4 (p = 0.125).

## FrogNano-4B on the pilot (2026-10-02)

[FrogNano](https://huggingface.co/microsoft/FrogNano-4B-2609), Microsoft's RL-only Qwen3.5-4B coding agent, was served
with vLLM 0.30 on the 3090 at its card's settings (temperature 0.6, 8K tokens per turn, 131K context). It ran under
the usual caps, 4 at a time per harness.

- **pi: 7/20.** FrogNano's own 31.1% on full TB2, in its Leaf harness with 150 steps, would be ~6. R1s gets ~12.
- **prime_agent: 3/20, not a clean number.** The vLLM engine died at 22:38 with no traceback, and no OOM or GPU
  fault in the kernel log. The rollouts in flight on tasks 6, 13, 15, 17 and 19 failed with connection errors. They
  were not rerun.

## Reply cap under prime_agent: 16K vs 32K tokens per call (2026-10-05)

R1s-SD on the held-out 20, 3 attempts per cap, 4 at a time, servers with the context clamp. Same 60 turns and
1-hour rollouts.

| | 16K | 32K |
|---|---|---|
| Solved (3 attempts) | 26/60 (9, 9, 8) | 31/60 (11, 9, 11) |
| Runs ending on a length-cut reply | 7 (12%) | 7 (12%) |

- 32K is worth about 1.7 solves an attempt and is the setting for every prime_agent run since.
- The doubled cap did not cut the length-cut endings. A reply that hits the cap before any tool call ends the run as
  `agent_completed`; at 32K the model thinks for 32K tokens instead of 16K. Examples: decoding sqlite pages byte by
  byte in its head (sqlite-db-truncate), redesigning the interpreter in prose (schemelike-metacircular-eval). These
  are 7 of the 11 (16K) and 7 of the 10 (32K) runs that finished on their own and were wrong, all on hard tasks.
- The earlier ~35% figure (of runs that finished on their own) came from older servers without the clamp; on these
  runs it is ~20% at both caps.
- The lever is a thinking budget per reply, or turning a cut reply into "continue, and act" instead of an ending.

## Plan first, then execute (2026-10-05)

Can R1s-SD do better when "work out what to do" is split from "do it"? `plan_eval.py`: each attempt, a fresh plan per
task from one tools-free call, then a normal prime_agent run with the plan in the task prompt. Held-out 20, 3 attempts,
same settings as the 32K baseline above.

- **blind:** the plan comes from the task text alone ("imagine an IPython REPL: what actions would you take"), and the
  run is told to carry it out.
- **look:** a 10-turn read-only inspection run first. Its transcript goes into the planning call, and the run follows
  the plan but rewrites the remaining steps when a step fails.

| | Attempts | Solved | Sandbox or server failures (qemu-startup not counted) |
|---|---|---|---|
| 32K baseline | 11, 9, 11 | 31/60 | 3 |
| blind | 10, 10, 10 | 30/60 | 8 |
| look | 12, 8, 5 | 25/60 | 8 |

- No gain. Counting only runs that weren't killed by the sandbox or server: baseline 31/57, blind 30/52, look 25/52.
- Most failures in the plan arms were Prime terminating sandboxes mid-run (`The sandbox has been terminated`) while
  the agent was doing something harmless. That was the evening's infrastructure, not the plans.
- The one task that moved: sqlite-db-truncate, which needs a look at the damaged file before any plan makes sense.
  look 2/3, baseline 1/3, blind 1/3.
- With the lessons loop (root README, RLTL;DR) this closes the in-context tricks. Text in the prompt, whether a lesson
  or a plan, doesn't change what R1s-SD does once it is working.

## Three harness fixes through a proxy: 87 tasks, off vs on (2026-10-06)

`fix_proxy.py` sits between prime_agent and the R1s-SD server and applies three fixes, each aimed at a measured way
R1s-SD loses runs. **cutoff:** a reply that hits the 32K cap before any tool call is retried with thinking off.
**check:** the first time the model says it's done, it is asked to re-check every output the tests will look at.
**image:** a picture in the conversation becomes a text note instead of the server's error. Test (`fixes_full.sh`):
TB2 without the two qemu tasks (87), 2 tries per task, fixes off (the same proxy passing everything through) vs on, two
HF a10g-large servers per side, 60 turns, 1-hour runs. Runs lost to infrastructure were rerun once
(`lost_runs.py`, `fixes_rerun.sh`). Analysis: `fixes_analysis.py` → `results/fixes_test.json`.

| | Fixes off | Fixes on |
|---|---|---|
| Usable runs (of 174) | 161 | 150 |
| Solved | 61 (37.9%) | 67 (44.7%) |
| Mean per-task solve rate, 82 tasks with usable runs on both sides | 39.0% | 43.9% |
| Same, without the 7 heavy tasks (77 tasks) | 40.9% | 46.8% |

- **Probably a small gain, not proven.** Paired by task: +4.9 points (95% CI −3.7 to +13.4, p = 0.34); without
  the heavy tasks +5.8 (−3.2 to +14.9, p = 0.27). Fixes on did better on 15 tasks, off on 11–12, and 51–55 tied. Two
  tries per task can't resolve a 5-point effect.
- **The image fix carries all of it.** On the 24 tasks where the model looked at a picture: off 13/44 usable runs
  (30%), on 17/36 (47%). Without fixes, 11 runs ended on an image request and none was solved. On the other 59 tasks
  (no pictures, not heavy, usable runs on both sides) the per-task rate is the same: 43.2% vs 43.2%.
- **check: maybe a little.** With fixes off, runs where the model said it was done were solved 52/78 (67%). With fixes
  on, runs where check fired were solved 56/74 (76%). The model always acted on the check.
- **cutoff rescues runs, not solves.** It fired in 25 usable runs (66 times); 2 of those were solved. Without fixes,
  10 runs ended on a cut-off reply. A model that thinks itself into the cap is usually on a task it was going to fail.
- **Prime sandbox terminations are now the main noise.** Of 174 runs per side, 31 (off) and 47 (on) were lost the
  first time, mostly "The sandbox has been terminated". The one rerun of each recovered 41 of the 78; 37 (47%) were
  lost again. Lost runs aren't random (heavy tasks lose more), so the usable runs lean toward lighter tasks.
  - Seven tasks (compile-compcert, gpt2-codegolf, path-tracing, mteb-retrieve, portfolio-optimization,
    torch-pipeline-parallelism, train-fasttext) lose their sandbox in most runs on both sides. They compile, train
    or render inside 2–4 GB.
  - The sandboxes do get each task's requested CPUs and memory, and the account balance was fine. Prime records
    no reason for a termination.
  - For RL on Prime sandboxes this loss rate needs fixing or routing around first.
- Incidents, none affecting the comparison:
  - A fixes-off run hung after its 13:27 timeout, holding its server until it was stopped at 16:50; it counts as
    lost.
  - A 16:45 wifi drop and a 17:33 network blip ended 9 runs; they were rerun.
  - A duplicate launch (`fixes_big.sh`, removed in 097331fb) doubled the load on all four servers from 07:05 to 09:03.
- Spend: GPU servers about $91 (61 server-hours), including the duplicate launch (about $12) and the reruns (about
  $16). Prime sandboxes are extra.

## How R1s-SD uses prime_agent's features (2026-10-07)

prime_agent gives the model one tool, `ipython`, a Python REPL. The rest of the harness lives inside the REPL as Python
calls, described in a long system prompt:
- `edit(...)` for file changes;
- `compact` for trimming its own context;
- `rlm.spawn(...)` for sub-agents;
- `bash(...)` handles for background shell jobs;
- `rlm.harness` and `refine` for memory, skills and self-refinement;
- skills such as `websearch` and `attach_image`.

`feature_audit.py` finds these calls in every saved R1s-SD prime_agent rollout: 1,653 rollouts, of which 1,439 are
usable, over 441 tasks and 12 experiments. Full tables: `results/feature_audit.md`.

| Feature | Runs that used it | Solved with / without, same tasks |
|---|---|---|
| Context compaction | **0** | – |
| Sub-agents | **0** | – |
| Memory, skills, refine | 1% | 69% / 46% (13 tasks) |
| `edit` helper | 11% | 53% / 42% (104 tasks) |
| `bash()` handles (the intended way to run commands) | 6% | about the same |
| `subprocess` / `os.system` (the prompt says not to) | 82% | – |
| Polling with `sleep` (the prompt says not to) | 20% | 33% / 34% |

- **It calls tools that don't exist.** 67% of runs call a `bash` tool, or call `websearch` or `edit` as if they were tools, at
  least once. Each call gets back "Tool bash not found": 1,864 turns, 4% of all turns. That's the pi / Claude Code
  habit leaking through.
- **It never manages its context.** 228 runs passed 100K tokens of prompt (the limit is 131K) and none compacted.
  The bigger the context, the lower the solve rate. Harder tasks also run longer, so this is not cause and effect.

| Peak prompt | Runs | Solved | Ended: done / turn cap / time limit |
|---|---|---|---|
| under 32K | 551 | 53% | 484 / 16 / 51 |
| 32-64K | 367 | 47% | 264 / 50 / 53 |
| 64-100K | 293 | 35% | 129 / 109 / 55 |
| over 100K | 228 | 20% | 65 / 125 / 38 |

- **How runs end:** done 942 (59% solved); 60-turn cap 300 (13%); time limit 197 (13%). 155 runs (11%) ended on a reply
  cut off by the length cap, and none of those was solved.

**What it means:** R1s-SD uses prime_agent as a plain Python sandbox and ignores the features that make prime_agent
different from pi. RL can't teach a feature the model never tries: with 0 compactions and 0 sub-agents, there is no good
example for the reward to reinforce. Those two features need a demonstration first, from coached runs or teacher runs. They
also need tasks that require them. The cheap wins fit RL as it is: stop calling missing tools (each one costs a turn and
appears in most runs), use `edit`, and act before the length cap.

## So far

- **Harness choice:** pi is the safer default for self-hosted models today. prime_agent's continual-harness features
  cost tokens, and its REPL and daemon add failure modes. Its image tool assumes a vision-capable endpoint. None of
  this says what a model trained on prime_agent would do, which is Prime's actual claim.
- **Model:** 35B-A3B over the 9B for agentic work (11 vs 6 under pi).
- **Local build:** R1s (all experts 3-bit, imatrix-searched 4-bit always-on path, 15 GB) is the local build. It is
  within noise of Q8 on solves (10.0 vs 10.3 of 19). It writes 1.28× Q8's tokens on the same solved tasks, down from
  R1's 1.67×; the engine was ruled out, and fidelity closed most of the gap.
- **Proxy fixes:** keep the image fix; it is the clear win (30% → 47% on picture tasks). check is cheap and maybe
  helps; cutoff mostly turns cut-off failures into ordinary failures. Together about +5 points per task, not proven at
  2 tries. Prime sandbox terminations (half the reruns lost again) must be dealt with before RL on Prime sandboxes.

## Next

- Untrained harness baselines, one model per GPU: Ornith 35B, Qwen3.8 27B and Ornith 9B at Q8, each under pi and
  prime_agent. They run on TB2 (3 attempts), DeepSWE (10 tasks) and Terminal-Bench 4 (20 tasks). This is the "before"
  for harness training.
- Find what hangs git-multibranch, rstan-to-pystan and custom-memory-heap-crash for the full hour on the local runs.
- More tasks or several attempts each before calling a harness winner. pi's lead is suggestive, not significant.
- Fix the budget in tokens rather than wall-clock, or serve with vLLM, which batches MoE decode far better than
  llama.cpp, so the time limit stops being a hidden token cap.
- The funnel ends at DeepSWE (`primeintellect/deep-swe`) for finalists: the 35B under pi first.
