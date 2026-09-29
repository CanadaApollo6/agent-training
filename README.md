# agent-training

Playground for learning LLM mechanics end to end on one RTX 3090: inference, CUDA kernels, pretraining,
post-training and RL, and compression. Where it leads is the pipeline behind a custom self-hosted agent:

**serve → make fast → teach (SFT/RL) → shrink (distill/quantize) → harness → deploy**

## Hardware and compute

| Where | What runs there |
|---|---|
| RTX 3090 24 GB (sm_86: bf16 yes; no FP8/FP4) | Policy and inference servers, kernels, small-model training |
| [Prime Sandboxes](https://www.primeintellect.ai/blog/sandboxes) | CPU microVM environments for agentic RL, reaching the local GPU through Prime Tunnels |
| Hugging Face Pro / rented GPUs | Bursts: RL above ~4B, ternary QAT, DMD, big eval sweeps |

Measured on this card (`uv run 00-setup/roofline.py`): 829 GB/s DRAM (89% of spec), 76.7 TFLOPS bf16 matmul.

## Modules

| # | Module | What gets built |
|---|---|---|
| 0 | [Setup](00-setup/) | uv env, CUDA toolkit, roofline benchmark |
| 1 | Inference | Qwen3.8 27B (4-bit) vs Bonsai 2 27B vs Kev-9B, measured against the bandwidth ceiling; KV cache, batching, speculative decoding |
| 2 | CUDA | Reduction → transpose → online softmax → matmul worklog → FlashAttention 1/2 → Triton, profiled in Nsight Compute; final boss: ternary GEMV for Bonsai |
| 3 | Transformer from scratch | ~100M-parameter pretrain; tokenizer, attention and sampling with no black boxes |
| 4 | Post-training | SFT → DPO → GRPO on Countdown with `verifiers`; MiMo-style groupwise grading; Ornith-style task proposer; agentic RL through Prime Tunnels |
| 5 | Compression | Distillation, ternary QAT on a small Qwen, Kev calibration under quantization, calibration RL for a System One model |
| 6 | Agents | Prime Agent + Cua Driver + a self-hosted model: the Smart Data prototype |

## Reading spine

- [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources): the kernel and inference track. Skip Hopper/Blackwell-only material (TMA, DeepGEMM, tcgen05, FlashAttention 3/4).
- [Kev](https://github.com/jaredpalmer/kev) and [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked): System One decision models.
  [Julia 1](https://huggingface.co/SupersonicLabs/Julia-1) makes the opposite bet to Kev: a 144M-parameter
  mmBERT-small encoder with a decision head, not a LoRA on a 4–9B LLM. It uses Jev's interface (state, question,
  2–20 options, choice/score/Boolean), runs on CPU, and is Apache 2.0. Its Jev comparisons are 100-example pilots
  against supplied reference numbers, not head-to-head runs. Planned: Jev vs Kev vs Julia on the Songbird gold set.
- [interference-search](https://github.com/Badtheorylabs/interference-search): beam search with merged states and a learned judge; a Countdown baseline for Module 4.
- [TensorFold](https://github.com/ashhart/TensorFold): exact speculative decoding with hand-written per-family
  kernels. Its [CUDA recipe book](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/cuda.md) is a
  worked example of Module 1 and 2 ideas. Its CUDA engine is only tested on DGX Spark; the 3090 is untried.
- [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss): Xiaomi's released agentic RL
  environments. 7.8K tasks: SWE with executable tests, cyber, enterprise knowledge work, webdev and music. Each
  knowledge-work task is a Docker environment with mock business systems served as MCP tools over SQLite, plus a
  rubric verifier: gates, then binary LLM-judge items, then a weighted mean. This is the template for Module 4
  environments, for Smart Data client environments, and for a Songbird environment.
- [Automating eval design and hillclimbing](https://claude.dev/blog/automating-eval-design-and-hillclimbing/)
  (Lance Martin, Anthropic, 2026-09-28): two Claude Code commands, shipped in the
  [claude-api skill](https://github.com/anthropics/skills/tree/main/skills/claude-api).
  - `build-eval` samples cases from production transcripts, bug reports and hand-written cases, grades them with
    code checks or an LLM judge, and checks that the grader is consistent.
  - `hillclimb` changes prompts, skills, tools or model settings, round by round, and keeps a train/test split. If
    train improves but test stays flat, it reverts the change as overfitting.
  - Its rules for a good eval: mirror real use, stronger models score higher, headroom below 100%, low
    run-to-run variance, and hard cases chosen by human judgement, not by "current models fail here".
  - It optimizes the harness, not the weights, so it is the cheap step before RL. The numbers are the author's own
    case studies (ticket cost 4.6¢ → 1¢ with accuracy 74.4% → 98.9%; the claude-api skill 66% → ~88% over 24
    rounds), not independent benchmarks.
  - For the Songbird gold set, Smart Data client evals and Module 4.
- [Strata](https://github.com/Niko1221/Strata): runs Qwen3.8-Flash-Next (125B MoE, 24,576 experts) at 2–3 bits on
  one 12–24 GB card. Hot experts are cached in VRAM, the CPU computes the rest from RAM in parallel, an n-gram
  table sits on SSD, and MTP drafts are exact. Measured only on an RTX 5070; its 3090 figures (100–140 tok/s) are
  estimates (±20%). It needs 64 GB of RAM, and this PC has 32 GB. A good worked example of MoE decode bytes.
- [Qwen3.8-27B one-click install for 16–32 GB NVIDIA GPUs](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install)
  (Mia's AI Lab, MIT): a serving kit around turboderp's [EXL3 quants](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
  in [ExLlamaV3](https://github.com/turboderp-org/exllamav3), plus DeepSeek Harness as the chat UI.
  - It plans context from the free VRAM.
  - The KV cache is int4 (the kit reports it within 0.001 KL of fp16), so each token of context costs a quarter of
    bf16.
  - MTP drafting is on by default.
  - Its measured 24 GB rows: **4.0 bpw at the full 262K context with images**, 5.0 bpw at 180–205K, 6.0 bpw at 84K.
    Compare our vLLM AWQ build: ~10K. Measured here: the 4.0 bpw build loads 262K on the 3090 (one sequence, 20.5
    GiB in use), decodes at 33 tok/s and 61.6 with MTP ([01-inference/qwen38-27b](01-inference/qwen38-27b/)).
  - turboderp's mean KL vs bf16 by bits per weight: 2.0 → 0.35, 2.5 → 0.30, 3.0 → 0.11, 4.0 → 0.05, 5.0 → 0.014,
    6.0 → 0.007.
  - EXL3 is rotation plus trellis post-training quantization, which makes it the strongest no-training baseline for
    the Ornith quantization work. Qwen3.8-27B exists as both EXL3 2.0 bpw and ternary Bonsai 2 (1.72 bpw, trained),
    so it's a clean rounding-vs-training comparison on one base model.
  - The figures are the kit author's measurements. The engine wheels come from turboderp's own releases.
- [AutoGym: Blueprint-First Generation of Verifiable Agent Gyms](https://arxiv.org/abs/2609.22592) (Noronha,
  Ravikumar, Lin; Amazon AGI; NeurIPS 2026 workshop): generating RL training tasks for tool-using agents,
  solvable by construction.
  - Write the blueprint first: the persona, the correct answer's entities and conditions, what the environment must
    contain, and how to verify. Only then build the environment (a per-task database behind tools) and the question.
    Every task is solvable because the answer existed before the data.
  - Explicit difficulty knobs: task topology (direct lookup → multi-hop → cross-source), capability axes (aggregation,
    temporal reasoning, negative checks…), how much the question withholds (0–1), and distractors.
  - Curriculum: per capability, below 30% success gets easier and above 80% gets harder. That keeps tasks in the band
    where GRPO gets a signal, since all-pass or all-fail groups have zero advantage.
  - Verifier: programmatic checks where the answer is exact, plus process, outcome and meta LLM judges.
  - Claimed $2–5 per generated task (Claude Sonnet 4.6 as generator) vs $500–1,000 hand-written.
  - Weak spots: one 8B RL run ("consistent improvement over 500 steps", no held-out transfer numbers), quality scored
    by its own rubric, partly LLM-judged rewards, no code release.
  - The pattern is the useful part: our own training pool, separate from Terminal-Bench, for harness-in-the-loop RL.
    It's also how to build a Smart Data client gym from a client's schema. It pairs with MiMo's mock-business MCP
    environments and Ornith's task proposer.
- [canada-quant](https://huggingface.co/canada-quant) (CQL.ca): a lab that quantizes giant open MoEs (GLM-5.3-Flash,
  DeepSeek V4, Hunyuan 3) and trains its own DFlash2 speculative drafters. Their
  [GLM-5.3-Flash W4A16](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) card is a worked example of MoE
  quantization done selectively:
  - Only the 36,288 routed-expert matrices go to INT4 (GPTQ, group 128, 256 calibration samples × 4K tokens).
    Attention, router, shared experts, embeddings, LM head and MTP head stay bf16. 599 → 178 GiB.
  - The experts hold most of the bytes and each token touches only 8 of 288, so rounding them costs little. The
    always-on parts are the sensitive ones. This is the recipe to try first on Ornith 35B-A3B.
  - Build gates before any eval: exact packed-tensor count, nothing quantized outside the experts, no collapsed expert
    scales.
  - On RTX PRO 6000, ≈63% of their AIME deficit was a thinking-budget wall (empty answers at the cap), not rounding.
    That's our "cap-limited" category in the reasoning probe.
  - NVIDIA's NVFP4 build loses to plain INT4 on H100 because Hopper has no FP4 math and emulates it. The number
    format has to match the card's native math: on the 3090, INT4/INT8 yes, FP8/FP4 no.
  - Drafters: MTP with 2 draft tokens is their sweet spot (5 collapses acceptance). A drafter with its own KV cache
    raises single-stream speed +46% but shrinks the KV pool ~6×, so aggregate throughput collapses under load.
  - 178 GiB never fits this PC; read it for method. The figures are the lab's own. They report losses too (NVIDIA's
    stack won all 28 cells on DGX Spark), which is a credibility signal.
- [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063) (Cowsik, Dolev, Li et al., 2026): a
  generator writes Brainf*ck programs, and a learner does next-byte prediction on their outputs. The generator is
  RL-trained on a learning-progress reward: how well the learner's gradient aligns with its recent parameter
  movement. There's no natural data, yet zero-shot loss on text, images, audio and code improves as a power law.
  Models are under 25M parameters, and absolute losses are far from useful (text only drops from ~8 to ~4.5
  bits/byte). [Code](https://github.com/nourya-aliz/self_play_pretraining) covers figures, evaluation, program
  logs and [learner checkpoints](https://huggingface.co/nourya-cohen/solomonoff-paper) (100K–24M, Llama-style),
  but not the self-play loop (generator, interpreter, reward). A toy reproduction fits Module 3 (and the 3090),
  with their checkpoints as the yardstick. The reward design fits Module 4, next to Ornith's task proposer.
- [Qwen-Image-2.1 viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo): DMD step distillation, LoRA merge precision loss.
