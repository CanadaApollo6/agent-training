# Self-distillation, round 1

Full fine-tune of bf16 Ornith 1.5 35B-A3B on its own shortest Terminal-Bench 2 solves, rebuilt as R1s-SD (the R1s
recipe applied to the tuned weights). Then R1s and R1s-SD were compared on the held-out 20-task pilot (`tasks.txt`).

## Pipeline

1. `harvest.py` collects every solved TB2 rollout (pi, prime_agent, Claude Code) and renders it exactly as TensorFold
   served it. It skips the pilot tasks, which are held out.
2. `select_data.py` keeps the two shortest solves per task: 71 rollouts, 122 samples, 5.65M tokens, 2.49M loss tokens.
   The harness mix is pi 33, prime_agent 20, Claude Code 18.
3. `stage.sh` copies everything to an 8-GPU pod. `pod_train.sh setup|train|export|rebuild` runs prime-rl (pinned at
   1bddcf6, plus `prime-rl-pretokenized.patch`) with `sft.toml`.
4. `finish_export.py` checks the bf16 export against the original. The frozen vision tower and routers must be
   bit-identical. Training moves only 0.1-0.2% RMS per tensor. It also rounds back to bf16 the DeltaNet tensors
   prime-rl keeps in fp32, and restores the MTP head.
5. `requant_rest.py` plus the R1s scripts rebuild the 3-bit routed experts, the 4-bit always-on path and the rest
   (verified byte-identical to R1s when run on the original weights).

## Training

| | |
|---|---|
| Hardware | 8×A100 80GB (driver 570, CUDA 13 through `cuda-compat-13-0`) |
| Packing and parallelism | 128K packs, cp=8 Ulysses, ep=8 |
| Run | 26 steps (2 epochs), lr 5e-6 cosine |
| Speed and memory | ~55 s/step, 68 GB peak |
| Loss | 0.50 → 0.32 |

The whole pod run took 2h17m, about $51.

Things that broke on the way, all fixed in the scripts:
- **Activation-checkpointing replay:** it handed an int tensor to a float op, because DeltaNet's context-parallel path
  copies to the CPU. The fix is `SD_RECOMPUTE_CPU_COPIES`.
- **Converter crash:** the converter writes the weights, then dies saving assets.
- **fp32 tensors in the export:** prime-rl exports `A_log` and the gated norm as fp32.
- **Rebuild environment:** the rebuild scripts need prime-rl's uv project.

## Result: no measurable change

pi, 60 turns, 1-hour rollouts, 4 at a time, 3 attempts each, on two A6000 pods (2026-10-01):

| | R1s | R1s-SD |
|---|---|---|
| solved per attempt | 13, 12, 10 (mean 11.7) | 12, 12, 12 (mean 12.0) |
| solved at least once in 3 | 14/20 | 14/20 |
| mean turns, same task and attempt solved by both (n=31) | 19.5 | 22.4 |
| failures at max turns / timeout | 8 / 2 | 6 / 1 |

Across the 3 attempts, R1s-SD did better on 3 tasks and worse on 3 (sign test p = 1). Riel predicted 10 → ~11; the
measured shift is +0.3 per attempt, well inside run-to-run noise (R1s alone ranged 10-13).

Task 16 failed with a pi install harness error on every run of both builds. Task 18 needs image input, which the
TensorFold server doesn't serve. So 18 tasks really discriminate.

Self-distillation on the model's own shortest solves didn't make it shorter: turns on shared solves went up, not down.

**Why this is unsurprising:**
- The dose was small: 122 samples, 26 steps, weights moved 0.1-0.2%.
- The data is behaviour the model already has.
- The pilot tasks were held out, so any gain had to transfer from the other TB2 tasks.

**What to change in round 2:**
- More data per task.
- Teacher traces as well as self traces.
- A larger dose.
- An eval with more power, such as full TB2's 89 tasks.

# Round 2: expert-iteration data (prime_agent only)

R1s-SD sampled prime_agent rollouts on the 69 TB2 training tasks (never the pilot), at the eval caps (60 turns,
1 hour). `run_datagen.sh` ran 3 shards on three lambdalabs A100-40GB pods (2026-10-02, about $58 including a failed
massedcompute probe).

- **Tasks solved before:** 42 tasks, 8 attempts each, 334 rollouts. 175 were solved (52%), on 36 tasks. Two rollouts
  hung and were discarded.
- **Never-solved tasks:** 27 tasks, 2 attempts each. The first shard went 0/18, so the rest were skipped.

`harvest.py --runs "../../06-agents/harness-evals/outputs/**/*r1s-sd-dg-*/**/traces.jsonl" --out data/round2.jsonl`
gives 175 solves, 331 samples and 5.05M loss tokens. On 8 tasks these are the first prime_agent solves; no task is
solved for the first time by any harness.

| solves of 8 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| tasks | 5 | 4 | 3 | 3 | 3 | 5 | 8 | 5 |

| shortest per task | rollouts | samples | loss tokens |
|---|---|---|---|
| 2 | 67 | 118 | 1.65M |
| 3 | 94 | 163 | 2.37M |
| 4 | 118 | 210 | 3.12M |
| all | 175 | 331 | 5.05M |

## How off-policy is teacher data?

`offpolicy.py` scores harvested samples under bf16 Ornith 35B, served by vLLM on one H100 pod
(`pods/offpolicy_pod.sh`, about $5). For every token the agent wrote, it records the log-probability Ornith gives it,
split into reasoning (before a turn's `</think>`) and action (answer text and tool calls). The scored set was the 61
teacher samples from the bake-off and 80 random round-2 samples of Ornith's own. Per-sample results are in
`results/offpolicy.jsonl`.

| source | samples | written tokens | logprob/tok | ppl | reasoning | action | under e^-5 |
|---|---|---|---|---|---|---|---|
| own (R1s-SD, round 2) | 80 | 1.10M | -0.41 | 1.51 | -0.49 | -0.20 | 0.4% |
| Qwen3.8-Max | 31 | 0.48M | -0.79 | 2.21 | -1.02 | -0.35 | 2.9% |
| DeepSeek V4.1 Flash | 24 | 0.33M | -0.88 | 2.40 | -1.09 | -0.32 | 3.2% |
| GLM-5.3 | 6 | 0.10M | -0.91 | 2.48 | -1.15 | -0.36 | 3.7% |

Teacher data is clearly off-policy, mostly in the reasoning.
- **Reasoning:** teacher reasoning costs about twice the nats per token of Ornith's own. 48 of 50 teacher samples fall
  below the 10th percentile of own samples (own median -0.53, teacher median -1.26).
- **Actions:** these are closer but still off. Teachers average -0.32 to -0.36 against -0.20, and 27 of 55 samples fall
  below the own 10th percentile.
- **Surprising tokens:** tokens Ornith gives under 0.7% are 7-9× as common in teacher data.

This is the regime where Finetuning with Sampling finds plain SFT forgets more and generalizes worse. The agent-scale
version of its fix is to keep each teacher tool call and let Ornith rewrite the reasoning that leads to it.
