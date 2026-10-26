# Nanbeige4.3-33B

Dense 33.1B Llama-architecture thinking model (Nanbeige / BOSS Zhipin, 2026-10-24, Apache-2.0). Reading-spine entry:
the top-level README. Terminal-Bench 2 pilot: `06-agents/harness-evals/README.md`.

**Status (2026-10-25): building a draft head.** On our 19-task TB2 pilot it averaged 11.0 of 19 (lookup run failed) vs
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
