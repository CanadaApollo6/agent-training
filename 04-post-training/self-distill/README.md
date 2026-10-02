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
