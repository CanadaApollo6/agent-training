# RL smoke test: Ornith-1.5-9B, GRPO, prime_agent on Terminal-Bench 2 (2026-10-07)

Does the whole RL loop run end to end? The model serves rollouts, the agent works in Prime sandboxes, the tasks'
tests give rewards, and the trainer updates the weights and sends them back to the model server.

- **Trainer:** [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) at `81052082`. GRPO-style groups of 4
  attempts per task, 16 attempts per step, 3 steps, learning rate 1e-6.
- **Tasks and agent:** 63 Terminal-Bench 2 training tasks, with the 6 heaviest left out. The agent is prime_agent:
  up to 60 turns, 90K tokens per attempt, and 32K per reply.
- **Hardware:** a Hugging Face Job on h200x4 at $20/h. 2 GPUs run vLLM and 2 run the trainer.
- **Network:** the sandboxed agent reaches the model through a Prime tunnel. `tunnel_check.py` showed an HF Job can
  open one.

```bash
CONFIG=ornith9b-tb2-smoke-h200x4.toml FLAVOR=h200x4 ./hf_rl_job.sh submit
./watch.sh <job id> <run>          # job log, finished steps, error count, stage changes
```

## Result: it works

| Step | Time | Average reward | Attempts cut off by the length cap | Turns per attempt | Trainer throughput |
|---|---|---|---|---|---|
| 1 | 14m 25s | 0.50 | 50% | 36.0 | 1,032 tok/s (includes warm-up) |
| 2 | 9m 50s | 0.75 | 25% | 36.8 | 4,012 tok/s |
| 3 | 8m 44s | 0.44 | 44% | 30.3 | 4,616 tok/s |

- **The loop runs.** Each step collected its 16 attempts and trained on them. The new weights reached vLLM over
  NCCL, and the next step's attempts used them: the off-policy lag never went past 2 steps.
- **The model server and trainer agree.** The KL between vLLM's and the trainer's token probabilities was
  0.002–0.006, so the trainer is learning from the same probabilities the model actually sampled from.
- **Trainer memory:** the peak was 70.4 GiB per GPU, with 131K-token rows. That is half an H200, so there is
  headroom for longer attempts or bigger batches.
- **The rewards mean nothing yet.** Three steps at lr 1e-6 can't move a model; the swings are which tasks each step
  drew.
- **Prime sandboxes still drop out.** 20 tracebacks over the run: sandboxes terminated mid-attempt, plus "Bad Gateway"
  on exec. Those attempts were retried (up to 2 times). Step 3 still lost 5% of its attempts.
- **Many attempts hit the 90K-token cap** (25–50%). On a real run, that cap and the turn limit decide how much of
  each attempt the model learns from.
- **No sandboxes leaked:** none with the run's label were left after the job ended.

## Cost

| | Minutes | Cost |
|---|---|---|
| Three failed starts (missing `prime_kernels`, missing `vllm-router`, NCCL broadcast sized before `dp` was set) | ~11 | ~$4 |
| The run (setup ~1 min, model load ~17 min, 3 steps, save) | ~42 | ~$14 |
| **Total** | | **~$18** |

The a100x8 version queued for over an hour and was cancelled before it ran. Sandbox fees aren't included.

Once warm, a step of 16 attempts takes about 9–10 minutes, which is **~$3 per step** on h200x4. Rollouts are the
bottleneck: the trainer finishes its part in a fraction of that.

## Fixes it took (in `hf_rl_job.sh` and the h200x4 config)

1. `uv sync` needs the `kernels` extra (`prime_kernels`) and the `disagg` extra (`vllm-router`).
2. On Hopper, `attn = "auto"` picks FlashAttention 3, which the job doesn't have, so the config forces
   `flash_attention_2`.
3. `rl` sizes the NCCL weight broadcast before `[deployment]` sets vLLM's data parallelism. Set
   `[inference.vllm] data_parallel_size` explicitly, or vLLM's workers have no broadcast receiver.
4. HF Jobs reject a timeout written as `4h30m` (use `270m`) and single-letter env names.

## The 35B fit check: R1s-SD on h200x8 (2026-10-07)

Can the real model, R1s-SD (Ornith 1.5 35B-A3B after self-distillation, bf16), run RL on one 8×H200 node? Same
tasks and harness as the 9B smoke test; 32 attempts per step in groups of 8; 3 steps
(`ornith35b-sd-rl-fitcheck-h200x8.toml`). 4 GPUs run vLLM (one 67 GB copy each), 4 train (experts split 4 ways,
each 128K row split 4 ways). The trainer keeps fp32 master weights, keeps the expert routing frozen, and replays the
expert choices vLLM made (router replay).

```bash
CONFIG=ornith35b-sd-rl-fitcheck-h200x8.toml FLAVOR=h200x8 TIMEOUT=120m MODEL_SRC=ornith-sd-bf16 ./hf_rl_job.sh submit
```

| Step | Attempts collected in | Reward | Cut off by the length cap | Trainer update | Trainer peak memory | vLLM-trainer KL |
|---|---|---|---|---|---|---|
| 1 | 26m 16s (includes warm-up) | 0.61 | 48% | ~7.5 min | 134.1 GiB | 0.0010 |
| 2 | 10m 20s | 0.60 | 20% | 4m 13s | 133.9 GiB | 0.0009 |
| 3 | 4m 40s | 0.20 | 53% | 5m 18s | 133.8 GiB | 0.0010 |

- **It fits, with little room.** The trainer peaks at ~134 of 140 GiB per GPU. Each step logged one allocator warning
  (a 20 MB mapping failed, then succeeded after the cache was freed). The peak is the same every step, because
  rollouts are packed into fixed 128K rows: longer attempts add rows, not memory per row.
- **Router replay works.** The vLLM-trainer mismatch is ~0.001, half the 9B smoke test's without it.
- **Speed:** the trainer reaches 10K tokens/s (MFU ~70%) once warm. Collecting attempts is the bottleneck. Steps 2
  and 3 were fast partly because attempts finished during step 1 were already waiting.
- **Prime sandboxes still drop out:** 13 attempts lost to "The sandbox has been terminated" in ~47 minutes (retried).
- **Setup:** ~10 min to install, 7 min to copy the 67 GB checkpoint from the bucket, ~3 min to load the trainer.
- **Cost:** 66 minutes, ~$44. Stopped while it was saving the final checkpoint (fp32 weights plus optimizer state,
  ~420 GB), which a fit check doesn't need.

**For the real run:** steady state is roughly 10–13 minutes per step of 32 attempts, so 50 steps is about 9–11 hours,
**~$360–450** on h200x8 plus sandbox fees. Raising the number of attempts in flight above 64 should shorten steps
while vLLM's GPUs still have spare capacity. The rewards above mean nothing yet: 3 steps at lr 1e-6.

## Before the real run: does RL actually help R1s-SD? (plan, 2026-10-07)

The fit check proves the loop runs on the 35B, not that it teaches anything. Riel's bar before ~$400: evidence that
RL moves R1s-SD.

**Room to learn (from existing rollouts, free).**
- On Terminal-Lego (187 tasks, two tries each), R1s-SD's average solve rate is 48%, and 58% of tasks are solved at
  least once.
- RL mostly turns "sometimes" into "usually", so the near-term prize on tasks like these is about +10 points.
- 37 tasks are solved exactly once in two tries: those give the clearest training signal.

**One problem to fix before any paid step: the length cap ends half the attempts.**
- In the fit check, 20–53% of attempts per step reached the 90K-token attempt cap. It sits at 90K so that one more
  32K reply still fits the 131K window.
- Those attempts score whatever the tests give a half-finished sandbox, usually 0. Much of the signal would then be
  "be shorter", not "solve it".
- Options:
  - Compaction, if the coaching test shows R1s-SD uses it when told: context stays small, so the cap rarely bites.
  - A 262K window: the trainer's 128K rows would need more splitting, so memory must be rechecked.
  - A 16K reply cap, which lets attempts run to ~110K. It cost 1.7 solves per attempt in the evals.
- Lost sandboxes are already handled: the config reruns a Prime-killed attempt up to twice and drops it from training
  after that, and drops timed-out attempts too.

**The pilot: a go/no-go run that isn't wasted.**
- **Tasks:** a fixed pool of 16 tasks R1s-SD solves sometimes. Each step trains on 4 of them in groups of 8 attempts.
  Over 12 steps, every task comes up 3 times, so its solve rate can be watched rising (or not).
- **Held-out check:** the held-out 20 at step 0 and step 12, on the run's own model server (prime-rl's eval setting).
  The before and after are then the same engine and settings.
- **Learning rate:** probably 2e-6 rather than the smoke tests' 1e-6, so 12 steps can show a trend.
- **Go:** the pool's solve rate clearly up (about +10 points or more from its first pass to its third) and held-out not
  down. The full run then resumes from the pilot's checkpoint, so the pilot's spend counts toward it.
- **Also required for go: no reward hacks.** Vals AI's MiMo audit (root README, reading spine) showed RL learning
  to dig hidden answers out of Git objects, file timestamps, build caches and upstream repos.
  - Before the pilot, check each of the 16 tasks' images for a reachable answer: Git history, leftover build output,
    network access to the upstream project.
  - After it, run the same scan used on our 1,653 rollouts over the pilot's rollouts.
  - A rising solve rate that comes from a loophole is a no-go.
- **Weights moving:** log the share of bf16 weights that change each step. NeMo-DCR measured 0.6–1.2% per GRPO step
  across six models. Near zero means the updates round away, and the run can't be learning.
- **No-go:** the pool is flat (the recipe doesn't teach; fix it before spending more), or the pool is up but held-out
  is down (memorising; needs a bigger pool).
- **Cost:** ~$8–9 a step on h200x8, plus setup and load (~$15) and the checkpoint save: **about $120–140**, plus
  sandbox fees.
