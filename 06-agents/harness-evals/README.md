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

Setting aside the three that failed on plumbing, pi still has three solves prime_agent doesn't (build-pov-ray, mteb-
retrieve, portfolio-optimization), against one the other way. pi also got there on 34% fewer tokens.

**The MoE didn't decode faster under load.** It runs at 104 tokens/s for a single stream, but with 8 streams sharing
the card, each got about 15 tokens/s, no better than the 9B. Each stream routes its tokens to different experts, so
batching them saves little weight reading. The 30-minute limit was the same ~30K-token budget for both models.

## So far

- **Harness choice:** pi is the safer default for self-hosted models today. prime_agent's continual-harness features
  cost tokens, and its REPL and daemon add failure modes. Its image tool assumes a vision-capable endpoint. None of
  this says what a model trained on prime_agent would do, which is Prime's actual claim.
- **Model:** 35B-A3B over the 9B for agentic work (11 vs 6 under pi).

## Next

- More tasks or several attempts each before calling a harness winner. pi's lead is suggestive, not significant.
- Fix the budget in tokens rather than wall-clock, or serve with vLLM, which batches MoE decode far better than
  llama.cpp, so the time limit stops being a hidden token cap.
- The funnel ends at DeepSWE (`primeintellect/deep-swe`) for finalists: the 35B under pi first.
