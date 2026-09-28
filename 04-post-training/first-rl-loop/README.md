# A first RL loop: GRPO on 3-digit addition

Qwen3.5-0.8B, full fine-tune on the 3090. `grpo.py` does the training and the evaluation. Raw numbers are in
`results.json`. One run takes about 15 minutes, most of it evaluation.

## The loop

Each step:

1. Take 32 random problems like `964 + 494 =`.
2. Sample 8 answers per problem at T = 1.
3. Reward an answer with 1 if its first number is the sum, otherwise 0.
4. Compute each answer's advantage: its reward minus the mean reward of its group of 8.
5. Take one gradient step that pushes above-average answers up and below-average ones down.

There's no KL penalty and no reference answers. The learning rate is 2e-6, weights are fp32 and compute is bf16.

## Evaluation

Every score is measured on held-out problems that are never trained on:
- 100 problems of 3-digit addition (the training task)
- 50 of 4-digit addition
- 50 of 2-digit multiplication
- 50 word problems

Greedy uses the top token. pass@k is the chance that at least one of k samples (T = 0.8) is right, estimated from
256 samples per problem.

## Prediction

Riel predicted that "any of 16 right" would stay the same: *RL amplifies latent capability; it does not raise the
capability ceiling.* This is the position of Yue et al. (2025), *Does Reinforcement Learning Really Incentivize
Reasoning Capacity in LLMs Beyond the Base Model?*

## Results

| Task | Step | Greedy | pass@1 | pass@16 | pass@256 | Distinct answers in 32 samples |
|---|---|---|---|---|---|---|
| 3-digit addition (trained) | 0 | 1% | 21.5% | 96.6% | 100% | 12.4 |
| | 5 | 99% | 99.3% | 100% | 100% | 1.1 |
| | 100 | 96% | 95.3% | 97.0% | 98% | 1.1 |
| 4-digit addition | 0 | 0% | 22.2% | 94.8% | 100% | 10.6 |
| | 5 | 98% | 97.6% | 98.4% | 100% | 1.1 |
| | 100 | 96% | 95.4% | 96.0% | 96% | 1.1 |
| Multiplication | 0 | 74% | 44.6% | 91.5% | **100%** | 10.6 |
| | 5 | 84% | 80.5% | 90.5% | 96% | 2.7 |
| | 100 | 70% | 67.9% | 76.5% | **82%** | 1.5 |
| Word problem | 0 | 10% | 3.7% | 31.8% | **80%** | 21.8 |
| | 5 | 12% | 4.1% | 32.6% | 78% | 20.3 |
| | 100 | 8% | 6.6% | 28.7% | **54%** | 13.7 |

P(first token is ` ?`) on addition went from 58% to 0% by step 5. The training reward went from 12.5% to 97.7% by
step 5.

## What it shows

1. **The prediction held: nothing new was learned.** Addition pass@16 went 96.6% → 100% → 97.0%. The model could
   already add. RL removed the "worksheet" habit (` ?`), and greedy went from 1% to 99% in 5 steps. That's
   amplification: the right path was already in the samples, and RL made it the top choice.
2. **Early RL transfers the format fix.** At step 5, 4-digit addition (never trained on) also jumped from 0% to 98%
   greedy, and multiplication pass@1 went from 45% to 81%. What transferred was "answer with a number", not
   arithmetic skill: multiplication pass@16 stayed flat (91.5% → 90.5%).
3. **Continued RL lowers the ceiling.** From step 5 to step 100, the training task was already solved, but training
   kept pushing. On the untrained tasks, pass@256 fell: multiplication 100% → 82%, word problems 80% → 54%. The
   number of distinct answers fell too (word problems 21.8 → 13.7). The model narrowed onto fewer paths and lost
   correct answers it used to sample. This is the Yue et al. result, reproduced in miniature.
4. **Saturated training hurts even the trained task.** From step 40 on, the training reward drifted between 90% and
   100%, and held-out addition slipped from 99% to 96%. A task with no headroom gives almost no learning signal
   (every group all right means zero advantage), yet the remaining updates still move the weights.

## Lessons for Smart Data

- Pick base models by pass@k on the client eval. RL can only amplify what some sample already gets right.
- Train on tasks with headroom. The addition task was solved in 5 steps and harmful after that.
- Keep a regression suite of tasks you are *not* training on, and watch pass@k at large k, not just greedy. Stop
  when the held-out suite starts falling.
- Regularization exists for exactly this reason: a KL penalty toward the starting model, or early stopping. This
  run used neither.

## Caveats

- There is one seed, and the evaluation sets are small (50–100 problems, so a few points either way are noise).
  The multiplication and word-problem drops (9 and 13 problems out of 50) are well beyond that.
- The answers are 8 tokens long with no reasoning, and it's a 0.8B model. Longer reasoning chains and longer
  training are where the "RL does raise the ceiling" counter-evidence (for example NVIDIA's ProRL) comes from.
