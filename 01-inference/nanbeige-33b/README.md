# Nanbeige4.3-33B

Dense 33.1B Llama-architecture thinking model (Nanbeige / BOSS Zhipin, 2026-10-24, Apache-2.0). Reading-spine entry:
the top-level README. Terminal-Bench 2 pilot: `06-agents/harness-evals/README.md`.

**Status (2026-10-25): parked.** On our 19-task TB2 pilot it averaged 11.0 of 19 (lookup run failed) vs R1s's 10.0,
within noise (p ≈ 0.7), at 2.71× R1s's tokens per solve. That didn't meet the gate for a 3090 build or a draft head.

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
