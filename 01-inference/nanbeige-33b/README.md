# Nanbeige4.3-33B

Dense 33.1B Llama-architecture thinking model (Nanbeige / BOSS Zhipin, 2026-10-24, Apache-2.0). Reading-spine entry:
the top-level README. Terminal-Bench 2 pilot: `06-agents/harness-evals/README.md`.

**Status (2026-10-26): draft head trained and cooled down.** Held-out acceptance 2.9–4.0 tokens per verify (greedy),
2.7–3.8 sampled at the agent's settings. Single-user speed on an H200 with SGLang: 2.55× on HumanEval, 2.05× on
MT-Bench (greedy); 2.39× and 1.78× sampled. Next: a serving path for Riel's machines. On our 19-task TB2 pilot it averaged 11.0 of 19 (lookup run failed) vs
R1s's 10.0, within noise (p ≈ 0.7), at 2.71× R1s's tokens per solve. That didn't meet the gate, but Riel chose to build
the draft head anyway: if the model is as good as it looks, the speed-up is worth it.

## Draft head plan (2026-10-25)

Shape: a DFlash2 block drafter, the same kind our Qwen3.8-27B build runs (z-lab's Qwen3.8-27B-DFlash2), not an MTP
head. That one beat EXL3's MTP drafting by 1.7× on the 3090, and the engine's drafter path already speaks it. It
guesses 8 tokens at once from five of the target's layers (4, 15, 26, 38, 49 of 52), with 5 small layers of its own.
Width 7168 to match the target (it shares the target's embeddings and output head, frozen), MLP 11008 (half the
target's) so it stays ~1.85B parameters, ~1 GB at 4 bits next to a 17 GB target on the 24 GB card. Config:
`drafter/specforge/nanbeige4.3-33b-dflash2.json`.

Training: SpecForge (sgl-project, rev fddcb4a), which trains DFlash2 online. SGLang servers run the target over each
sample and hand its layer outputs to the trainer, so nothing big is stored on disk. Data is Nanbeige's own writing:

- 36K prompts' responses (`drafter/gen_responses.py`, thinking included, agent sampling, 4K-token cap), made on a
  4× H200 spot pod with FP8 weights and KV, one server per GPU (`drafter/serve_gen.sh`).
- The 57 pilot agent runs cut into 451 windows of ~7K tokens (`drafter/specforge/traj_to_specforge.py`): its real
  tool-calling turns, the text it will draft most.

Mask token `<s>` (id 1): the chat format never uses it (its BOS is `<|im_start|>`). Chat format registered in
SpecForge as `nanbeige-thinking` (`register_template.py`).

## Draft head training (2026-10-26)

Data: 35,999 responses (94.5M tokens, one on-demand H200, ~$49 with a lost spot pod) + 451 agent windows = 36,450
samples. Prime only had single-GPU pods, so the SGLang capture server (FP8) and the trainer share one H200
(`drafter/specforge/allow_shared_gpu.py` drops SpecForge's separate-GPU check). Smoke: 60 steps, loss 19.7 → 7.5,
~2.3 samples/s. Full run (`full-1gpu.yaml`): 2 epochs, 2,278 steps of 32 samples, constant lr 5e-4, ~2.9 samples/s,
7 h. The training pod cost ~$44 (8.2 h with setup and the final eval). `drafter/train_watch.sh` copies each checkpoint's weights home; `drafter/train_status.py` draws progress.

Gotchas: SpecForge's JSON loader needs one schema for every row (`uniform_schema.py`; verified token-identical); the
in-flight low watermark must cover one optimizer step; a crashed run blocks port 30000 for a minute (`train_run.sh`
waits). Serving needs SGLang main: SpecForge's pinned 0.5.18 has no `DFlash2DraftModel` (`eval_accept.sh`).

Held-out check of the step-250 weights (11% of training) on an H100, FP8 target, block 8, thinking on, greedy, 16
prompts per set (`drafter/results/eval-step250/`). Acceptance = tokens per verify step; z-lab's Qwen3.8-27B-DFlash2,
trained on far more data, gets 4.1–5.5 on the same sets.

| Set | GSM8K | MATH-500 | HumanEval | MBPP | MT-Bench |
| :-- | --: | --: | --: | --: | --: |
| Acceptance, step 250 | 2.09 | 2.00 | 2.00 | 2.31 | 1.67 |
| Acceptance, final (step 2278) | **3.85** | **3.69** | **3.26** | **3.72** | **2.80** |

Single user (10 prompts), step 250 on an H100: HumanEval 78.6 vs 49.4 tok/s without the draft (1.59×), MT-Bench 61.9
vs 49.5 (1.25×).

Final weights (`drafter/results/eval-step2278/`, 64 prompts per set, on the training H200): single user, HumanEval
233.5 vs 95.8 tok/s (**2.44×**), MT-Bench 187.5 vs 96.0 (**1.95×**). Greedy HumanEval output came out the same length
with and without the draft (10,469 tokens), as lossless drafting should. Acceptance still rose through the last
epoch, so more data or a third epoch would likely help; the gap to z-lab's 4.1–5.5 is mostly chat and code.
The exported draft (`specforge export --to hf`, 3.8 GB) is in `drafter/data/draft-step2278/`; all ten checkpoints'
weights are in `drafter/data/live/train-full/ckpt/`.

**Learning-rate cooldown (2026-10-26 evening).** The main run held lr at 5e-4 throughout. `cool-1gpu.yaml` warm-starts
from step 2278 (fresh Adam; the optimizer state went with the pod) and trains 570 more steps, about half an epoch,
with lr cosine-decayed from 5e-4 to 0. It took 1.8 h; the pod cost ~$15. Every set improved
(`drafter/results/eval-cool-step570/`, same 64 prompts per set). "Sampled" is temperature 1.0, top_p 0.95, no top-k
limit: the agent's settings. `specforge benchmark` defaults to top-k 1, which is greedy, so `eval_accept.sh` passes
`--top-k -1` explicitly.

| Set | GSM8K | MATH-500 | HumanEval | MBPP | MT-Bench |
| :-- | --: | --: | --: | --: | --: |
| Greedy, step 2278 | 3.85 | 3.69 | 3.26 | 3.72 | 2.80 |
| Greedy, after cooldown | **4.01** | **3.88** | **3.42** | **3.90** | **2.88** |
| Sampled, after cooldown | 3.78 | 3.57 | 3.30 | 3.63 | 2.71 |

| Single user, H200 | Draft | No draft | Speed-up |
| :-- | --: | --: | --: |
| HumanEval, greedy | 244.0 tok/s | 95.9 | **2.55×** (was 2.44×) |
| MT-Bench, greedy | 196.6 | 96.0 | **2.05×** (was 1.95×) |
| HumanEval, sampled | 227.2 | 95.2 | **2.39×** |
| MT-Bench, sampled | 169.9 | 95.4 | **1.78×** |

Greedy output was token-for-token the same length with and without the draft on both sets. The current best
draft is `drafter/data/draft-cool-step570/` (3.8 GB, sha256 0b36cfc0…); its three checkpoints are in
`drafter/data/live/train-cool/ckpt/`. Further gains need more data, especially multi-turn chat; see the
plan in the step-2278 notes above.

## Draft head (started, stopped)

The model has no draft head, so the drafting that speeds our Qwen3.8-27B build doesn't carry over. Plan if this is
picked up again: train a small head shaped like Qwen's built-in drafter (so the Zig engine's draft/verify path carries
over) on Nanbeige's own outputs, then measure acceptance.

- `drafter/prep_prompts.py`: 36K prompts (code 50%, chat 30%, math 20%) from Magicoder-Evol-Instruct,
  OpenCodeInstruct, UltraChat and NuminaMath. `prompts-12k.jsonl` is a third of each, shuffled. Both are gitignored (37 MB); the script rebuilds them.
- `drafter/gen_responses.py`: Nanbeige's responses, thinking included, at the agent sampling (T 1.0, top_p 0.95).
  Resumable, sharded across pods.
- `drafter/data/gen-k0.jsonl`, `gen-k1.jsonl`: the 639 responses made before the pilot verdict (1.4M tokens, mean
  2.2K, 195 cut at the cap; the first 58 at an 8K cap, the rest at 4K).

Throughput lesson: for this model, bulk generation is limited by the KV cache, not the weights. At BF16 KV an H200
holds ~50 four-thousand-token requests and makes ~1.5K tok/s. FP8 KV (`KV_DTYPE=fp8 serve_nanbeige_pod.sh`) doubles
the pool to 619K tokens. At 256 parallel requests vLLM still filled it and churned (evict, recompute, 3.4K → 0.1K
tok/s), so cap concurrency near pool ÷ (prompt + max tokens), ~128 here. Expect ~3K tok/s an H200, so the 12K set
is roughly 30M tokens, ~3 pod-hours (~$16).
