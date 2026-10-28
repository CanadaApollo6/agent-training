# Nanbeige4.3-33B

Dense 33.1B Llama-architecture thinking model (Nanbeige / BOSS Zhipin, 2026-10-24, Apache-2.0). Reading-spine entry:
the top-level README. Terminal-Bench 2 pilot: `06-agents/harness-evals/README.md`.

**Status (2026-10-26): draft head trained and cooled down.** Held-out acceptance 2.9–4.0 tokens per verify (greedy),
2.7–3.8 sampled at the agent's settings. Single-user speed on an H200 with SGLang: 2.55× on HumanEval, 2.05× on
MT-Bench (greedy); 2.39× and 1.78× sampled. Quantization test (2026-10-27): EXL3 3.5 bpw calibrated on
Nanbeige's own text drifts KL 0.036 (chat) / 0.066 (agent) from bf16; 4.0 bpw 0.023 / 0.044. TB2 at 3.5 bpw
(2026-10-27): 11.3 of 19 vs BF16's 11.0 (p = 1.0), at 1.21× the tokens per solve. Next: the TensorFold port. On our 19-task TB2 pilot it averaged 11.0 of 19 (lookup run failed) vs
R1s's 10.0, within noise (p ≈ 0.7), at 2.71× R1s's tokens per solve. That didn't meet the gate, but Riel chose to build
the draft head anyway: if the model is as good as it looks, the speed-up is worth it.

## TensorFold port plan (2026-10-28)

Worktree `01-inference/tools/TensorFold-llama`, branch `nanbeige`, from upstream main 31ce9eb3 (TF 1.1.8). Local
commits only until Riel OKs a fork push or an upstream issue. Upstream's rules apply: files ≤600 lines, drafted output
must equal serial output, new kernels go in `docs/kernel-contract.md`.

The tree has changed since the paper estimate below. The port is a new `llama` family modeled on `families/kolibri1`.
It reuses the shared operators `cuda/exl3.zig` (dense EXL3 matmul), `qk_rows`, `tree_attention` and norms. Qwen3.5's
path doesn't fit: it hard-codes head dim 256, partial RoPE and a gated attention, and its EXL3 path takes MoE only.

| Phase | What | Check |
| :-- | :-- | :-- |
| P1 | `llama` family, plain decode: EXL3 weights, bf16 KV | Per-layer and full-model logits vs exllamav3 on the same 3.5-nb weights; greedy text matches |
| P2 | `tensorfold-native serve`: chat template, reasoning, qwen3_coder tool calls | `smoke_serve.py` passes; BOS `<\|im_start\|>` at position 0 |
| P3 | DFlash2 draft head (`draft-cool-step570`) | Drafted greedy equals serial greedy; speed vs plain |
| P4 | fp8 KV for ~59K context, speed tuning | Long-context KL vs bf16 KV; tok/s |
| P5 | TB2 pilot on the engine, M3TiS #4 packaging | Score vs the 11.3 / 11.0 pilots; release only with Riel's OK |

## RTX 3090 fit and TensorFold port (paper estimate, 2026-10-26)

`fit_3090.py` (no GPU): Nanbeige is dense, and all 52 layers use full attention, so weights and KV cache both crowd a
24 GB card. KV costs 208 KiB a token at bf16 (fp8 halves it). Speeds assume a TensorFold port reading weights at
~730 GB/s, a 9-row verify at 1.15× a plain step, a 3 ms draft pass, and our sampled acceptance (code 3.30, chat
2.71). Treat them as ±25%.

| Weights | VRAM | Context left (bf16 / fp8 KV) | Plain | Drafted @2K code / chat | @32K, fp8 KV |
| :-- | --: | --: | --: | --: | --: |
| MLX 4-bit g64 (stock format) | 16.7 GiB | 12K / 25K | 40 | 104 / 86 | doesn't fit |
| EXL3 4.0 bpw | 15.1 GiB | 20K / 41K | 44 | 114 / 94 | 97 / 79 |
| EXL3 3.5 bpw | 13.4 GiB | 29K / 59K | 50 | 127 / 104 | 106 / 87 |
| EXL3 3.0 bpw | 11.6 GiB | 38K / 77K | 58 | 144 / 118 | 118 / 97 |

For comparison, measured on this 3090: R1s (Ornith 35B-A3B) drafted 180 code / 167 chat on TensorFold. So a working
port at 3.0–3.5 bpw would run at roughly 0.6–0.8× R1s's speed. With Nanbeige's 2.71× tokens per TB2 solve, it would
take about 3.5–4.5× R1s's wall-clock per task, for a TB2 score within noise of R1s's.

The stock 4-bit format doesn't leave room for agent-length context, so the port needs a custom quantization.
EXL3 (trellis, fractional bit rates) is the natural fit: TensorFold already ships EXL3 dense kernels (`exl3_linear.cu`,
used by its deepseek_v41 family). Whether 3.0–3.5 bpw keeps Nanbeige's quality is unmeasured, and on a 1-point TB2
lead, that decides everything.

What the port involves (TensorFold 1.1.x tree, `tools/TensorFold-speed`):
- **New `llama` family.** Plain GQA attention (56 query / 8 KV heads, head dim 128, full RoPE θ 1.2e8), RMSNorm,
  SwiGLU, 52 identical blocks. It's the simplest shape in the tree, with no GDN or MoE. It can reuse the qwen27
  family's full-attention, sampler, prefix cache and server pieces, and the EXL3 linears.
- **Drafter.** `families/qwen27/dflash/config.zig` already describes DFlash2 by config, and our draft passes its
  shape checks (7168 wide, 52 target layers, taps 4/15/26/38/49, mask 1). Today `requireStock()` refuses anything but
  z-lab's 27B drafter. Lift that gate, test the kernels at our sizes, and capture the five tap layers from the new
  engine. The draft can be stored at 4 bits; that can only change acceptance, never the output.
- **Quantization.** Convert with exllamav3 on a pod at 3.0 / 3.5 / 4.0 bpw. Gate on KL vs bf16 (the LFM2.5 pattern),
  then a TB2 pilot.
- **Server.** Add Nanbeige's chat template and tool-call format to `tensorfold-native serve`. Check that drafted
  greedy output equals serial greedy.

Suggested order: measure the quantization quality first (pod, ~$15–25: EXL3 conversions, KL, a TB2 subset at the
best bit rate). That gates the engine work. If 3.0–3.5 bpw holds the score, the port is worth building; if not,
stop there.

## Quantization test: EXL3 KL vs bf16 (2026-10-27)

`quant/` (pod nb-quant1, one H200, ~4h, ~$23). exllamav3 1.6.5 converted Nanbeige at 3.0, 3.5 and 4.0 bpw (head
6 bits, mul1 codebook) and once more at 3.5 bpw calibrated on Nanbeige's own text (`3.5-nb`: 200 chat answers and
50 agent windows from `drafter/data/live`, via `convert.py -cd`). `model_diff.py` scored each against bf16 on three
sets: 100 rows of Nanbeige's own chat/math/code answers with thinking (`gen`), 40 windows from 20 pi-harness
trajectories (`agent`), and 100 rows of wikitext-2. Scored and calibration rows come from different conversations.
Every row is 2048 tokens. Full table: `quant/results/summary.tsv`.

| mean KL vs bf16 (top-1 agreement) | 4.0 | 3.5-nb | 3.5 | 3.0 |
| :-- | --: | --: | --: | --: |
| own chat answers | 0.023 (95.6%) | **0.036 (94.5%)** | 0.043 (93.9%) | 0.078 (91.6%) |
| own agent runs | 0.044 (94.7%) | **0.066 (93.4%)** | 0.073 (93.0%) | 0.122 (90.5%) |
| wikitext-2 | 0.051 (91.6%) | 0.109 (87.5%) | 0.093 (88.6%) | 0.161 (84.6%) |
| files on disk (incl. 2.4 GB bf16 embedding, kept in RAM) | 18 GB | 16 GB | 16 GB | 14 GB |

What we learned:
- **The start token matters a lot.** The first agent and wikitext runs started mid-text with no `<|im_start|>` (the
  model's BOS) at position 0. Without it every quant fell apart on a minority of tokens: agent KL 0.39 at 4.0 bpw,
  0.84 at 3.0, while the median stayed small. With the start token restored, the same rows score 8–9× lower (the table
  above). Real requests always start with `<|im_start|>system`, so the table is the realistic one. The engine port must
  keep that token at position 0, and its KV entry is a candidate for extra precision.
- **exllamav3 runs Nanbeige correctly.** On the same 8 wikitext rows, bf16 perplexity is 18.05 under exllamav3 and
  18.09 under HF transformers (`quant/hf_ppl.py`). Raw wikitext is just far from what this chat model was tuned on.
- **Own-text calibration helps where it counts.** At 3.5 bpw it cuts KL 16% on chat and 10% on agent text, and costs
  some wikitext. Its agent calibration windows also lacked the start token; a rebuild with it may gain more.
- **Use an 8-bit KV cache, not 4-bit** (simulated on 3.5-nb, agent rows with BOS): 8-bit adds nothing (0.066 →
  0.066), 6-bit +5%, 5-bit +21%, 4-bit +70% (0.112), about the cost of dropping the weights from 3.5 to 3.0 bpw.
  `fit_3090.py` already assumes fp8 KV.

Reading: 3.5-nb is the candidate (59K fp8 context on a 3090 by `fit_3090.py`). 4.0 is clearly closer to bf16 but
leaves 41K. 3.0 roughly doubles the drift again. KL alone can't say whether a 1-point TB2 lead survives. Next: rebuild
the 3.5 custom quant with start-token calibration rows (maybe a per-tensor `sc_optimize` recipe), then a TB2 subset at
3.5-nb vs bf16. The 3.0, 3.5-nb and 4.0 quants are kept locally in `quant/models/` (gitignored).

## Quantization test 2: TB2 at 3.5 bpw (2026-10-27)

Pod nb-tb2q1, one H200, 8h34m, ~$46. Logs in `quant/results/night2/`. Comparison:
`06-agents/harness-evals/results/nanbeige-exl3-tb2-compare.md`.

**Rebuild with start-token calibration (`3.5-nb2`): no gain.** On agent rows its KL is 0.0663, against 0.0659 for
3.5-nb. On chat rows it's 0.0367 against 0.036. The missing start token in last night's calibration windows didn't
matter, so the pilot served 3.5-nb.

**Server.** vLLM can't load EXL3, so `quant/serve_exl3.py` serves it. It matches the BF16 pilot's vLLM setup except
for the weights: same template, T 1.0 / top_p 0.95, qwen3 reasoning and qwen3_coder tool parsing, and the same 400 on
image input. KV cache is 8-bit. Decode speed per stream on the H200 is 66 tok/s alone, 46 at 3 streams, 35 at 7 and
22 at 21. In the agent runs it averaged 14.0 output tokens per model-second against vLLM BF16's 28.3 (0.49×). To keep
that from costing timeouts, the task cap was 120 minutes instead of 60. No other H200 was free, so the three attempts
ran one after another on one pod at `-c 7`.

| Solved of 19 (benchmark lookups scored as failures) | a1 | a2 | a3 | Mean |
| :-- | --: | --: | --: | --: |
| BF16, vLLM, 60-min cap | 11 | 9 | 13 | **11.0** |
| EXL3 3.5-nb, 120-min cap | 10 | 12 | 12 | **11.3** |
| EXL3 3.5-nb, only runs done inside 60 min | 7 | 10 | 9 | 8.7 |
| EXL3 3.5-nb, 60 min at BF16 decode speed (model time × 0.49 + tool time) | | | | 11.0 |

- **The quant keeps the score.** On paired per-task solve rates it's +1.8 points a task vs BF16, better on 3 tasks
  and worse on 3, p = 1.0. That rules out a large loss but not a 1-point one: 3 attempts can't resolve that.
- **It talks about 20% more.** Solved runs used 1.21× BF16's output tokens (95% CI 1.04–1.43), and were longer on 10
  of 12 shared tasks. That's a real cost, on top of Nanbeige already using 2.71× R1s's tokens.
- **Strict 60 minutes on this server costs 2.6 solves.** The cause is the decode speed: with BF16's speed, the same runs
  score 11.0. One timeout (a1 git-multibranch) was a shell command that kept hanging (104 of 120 minutes in tools).
- **The image refusal hits both builds the same way.** financial-document-processor doesn't need vision: Ornith and
  Flint solve it 2/3 with OCR (tesseract) and never send an image. Nanbeige opens the scans first, gets the 400, and the
  run ends. That ended 6 BF16 runs (one had already solved its task) and 5 EXL3 runs: all three attempts at that
  task for both, plus build-pov-ray and pytorch-model-cli once or twice. Leaving those runs out, solve rates are 63%
  (BF16) and 65% (EXL3). A future pilot could reply with a text note instead of the 400, for every build at once.
- Lookups: mteb-retrieve was "solved" twice by EXL3 and once by BF16 after fetching the benchmark from GitHub or HF.
  Those runs count as failures.

Reading: 3.5-nb holds BF16's TB2 score within what 3 attempts can measure, at 1.2× the tokens. That clears the gate
for the TensorFold port. Its speed matters more than the weights now: exllamav3's 35 tok/s at 7 streams is the slow
path here, not the quant.

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
