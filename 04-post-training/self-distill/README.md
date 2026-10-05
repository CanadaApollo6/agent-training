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

## Teacher reasoning rewritten by Ornith (arm B data)

`rewrite.py` keeps every teacher action and replaces the reasoning before it, turn by turn. At each turn Ornith samples
4 drafts from the context so far, and the draft under which the teacher's tail (`</think>`, answer text, tool calls)
is most likely is kept. The teacher reasoning is never shown to Ornith, and earlier turns are already rewritten when a
later one is drafted. The script runs vLLM in-process because the scores need prompt log-probs with the prefix cache
on, which the OpenAI server doesn't allow. With the cache on, vLLM returns log-probs only for the uncached suffix; they
matched an uncached run exactly.

The run covered 129 teacher samples (47 solves on 11 tasks: DeepSeek's 27, Qwen's 20 at the time) on one 8×A100-80GB
pod, with 4 engines at TP=2 and `pods/rewrite_pod.sh` (2026-10-03, about 1h55m, ~$43).

| per token, nats | teacher reasoning | Ornith's chosen draft |
|---|---|---|
| reasoning | -1.12 (1.50M tokens) | -0.53 (1.04M tokens) |
| teacher tail after it | -0.39 | -0.50 |

- **Coverage:** the drafts replaced the teacher's reasoning on 1,473 of 1,499 turns. The other 26 kept it because no
  draft closed within 8,192 tokens.
- **Reasoning:** it now sits where Ornith's own does (-0.49 on round-2 data).
- **Tail:** the teacher's actions became somewhat less likely, by a median of 0.09 nats per tail token (90th
  percentile 0.30). The draft explained the action better than the teacher's reasoning on 234 turns. Teacher
  reasoning often writes out the command it then runs, and Ornith's drafts don't.
- **Size:** the data comes to 1.84M loss tokens out of 4.93M (raw: 2.06M of 5.15M), and every sample fits in 128K.

Rewritten samples are in `data/teacher_rewritten.jsonl` (gitignored), per-turn scores in `results/rewrite.jsonl`.

A second pass on one H100 PCIe (about 1h50m, ~$6) rewrote Qwen's last 8 solves: 41 samples, 394 turns, two of them
the first solves of regex-chess and train-fasttext. Short on KV memory, vLLM sometimes returned prompt log-probs that
didn't line up with its cached-token count. `rewrite.py` rescores those uncached (18 times in this pass). With both
passes, the teacher set is 55 solves on 13 tasks, 170 samples and 1,893 turns. Teacher reasoning was replaced on 1,861
turns: reasoning went from -1.14 to -0.54 nats/token, and the teacher tail from -0.38 to -0.48.

| arm | data | rollouts | tokens | trained on | packs/epoch | steps (2 epochs) |
|---|---|---|---|---|---|---|
| A | own K=4 + raw teacher | 173 | 14.58M | 5.78M | ~140 | 70 |
| B | own K=4 + rewritten teacher | 173 | 14.23M | 5.43M | ~135 | 68 |

## Round 2 training and held-out result: neither arm helps

Both arms trained on one Hugging Face Jobs `a100x8` (Prime had no 8-GPU node), back to back with round 1's settings,
then both R1s builds were rebuilt (`hf_job.sh`, 2026-10-04, 2h25m, ~$48). Arm A ran 70 steps in 68 min, arm B 68
steps in 59 min, about 56 s/step as in round 1.

Evaluation used round 1's protocol: pi, the held-out 20, 60 turns, 1-hour rollouts, 4 at a time, 3 attempts. Each
build was served by TensorFold on its own HF `a10g-large` (`../../06-agents/harness-evals/hf_serve.sh`, ~3h20m each,
~$15 for all three). R1s-SD was rerun on the same hardware as the baseline.

| | R1s-SD (round 1) | arm A: + raw teacher | arm B: + rewritten teacher |
|---|---|---|---|
| solved per attempt | 12, 9, 10 (mean 10.3) | 7, 8, 10 (mean 8.3) | 9, 8, 10 (mean 9.0) |
| solved at least once in 3 | 13/20 | 13/20 | 12/20 |
| runs ending at 60 turns (of 60) | 7 | 12 | 12 |
| tasks better / worse than R1s-SD | | 3 / 7 (sign test p = 0.34) | 1 / 4 (p = 0.38) |

- **Riel predicted B beats A.** B is ahead by 0.7 per attempt, but it was better on 5 tasks and worse on 4 (p = 1):
  a tie.
- **Neither arm beats R1s-SD.** Both are lower, by 1.3 and 2.0 solves per attempt, though neither gap is significant.
  R1s-SD itself scored 12, 12, 12 on A6000s in round 1, so its 9-12 range here is ordinary noise.
- **One shift shows in both arms.** Twice as many runs hit the 60-turn cap (12 vs 7). Mean turns barely moved (26.4,
  26.7 vs 26.4), so it isn't longer runs overall. The arms quit easy tasks about as fast and grind longer on tasks
  they don't solve. Teacher solves are longer than Ornith's own (more turns per solve), which may be what they taught.
- **What it means for the data:** the 12 teacher-only tasks are not in the held-out 20. These numbers say the teacher
  traces did not transfer to new tasks, under pi. Whether they help on the tasks they came from (the 12 teacher-only
  and 36 self-solved tasks), and under prime_agent where the data was collected, is still unmeasured.
