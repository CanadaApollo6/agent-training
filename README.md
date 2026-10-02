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

- [micrograd](https://github.com/karpathy/micrograd) (Karpathy): a scalar autograd engine and a tiny neural-net
  library in about 150 lines, with the video "The spelled-out intro to neural networks and backpropagation". The
  opening exercise for Module 3: write backprop by hand, then check the gradients against PyTorch. It works on
  single numbers, not tensors, so it teaches the chain rule but nothing about speed.
- [tinygrad](https://github.com/tinygrad/tinygrad) (tiny corp): the next rung, a full tensor framework small enough
  to read. Operations are lazy. It builds a graph, fuses it into as few kernels as it can, and generates code for
  CUDA, AMD, Metal or CPU. It connects Modules 2 and 3: `DEBUG=4` prints the kernels it writes for your model code,
  and `BEAM=2` searches kernel variants for speed (a small version of the SOL-ExecBench hill-climbing below). It
  runs LLaMA-class models and the driving model in comma.ai's openpilot.
- [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources): the kernel and inference track. Skip Hopper/Blackwell-only material (TMA, DeepGEMM, tcgen05, FlashAttention 3/4).
- [Kev](https://github.com/jaredpalmer/kev) and [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked): System One decision models.
  [Julia 1](https://huggingface.co/SupersonicLabs/Julia-1) makes the opposite bet to Kev: a 144M-parameter
  mmBERT-small encoder with a decision head, not a LoRA on a 4–9B LLM. It uses Jev's interface (state, question,
  2–20 options, choice/score/Boolean), runs on CPU, and is Apache 2.0. Its Jev comparisons are 100-example pilots
  against supplied reference numbers, not head-to-head runs. Planned: Jev vs Kev vs Julia on the Songbird gold set.
- [interference-search](https://github.com/Badtheorylabs/interference-search): beam search with merged states and a learned judge; a Countdown baseline for Module 4.
- [TensorFold](https://github.com/ashhart/TensorFold): exact speculative decoding with hand-written per-family
  kernels. Its [CUDA recipe book](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/cuda.md) is a
  worked example of Module 1 and 2 ideas. Its CUDA engine is tested only on DGX Spark (GB10).
  - Since 0.3.6.1 it refuses GPUs below compute capability 9.0, including the 3090 (8.6).
  - The blockers are narrow. The 4-bit matmul (`qmm.cu`) sums split-K partials across a thread-block cluster, but
    it already has a non-cluster path, used for more than 8 slices, that sums in the same order.
  - FP8 MMA appears only in the FP8/NVFP4 prefill kernels; 4-bit weights also have a bf16 prefill path.
  - No wgmma, TMA or bulk copies.
  - Its `qwen3_5_moe` family is Ornith 1.5 35B-A3B's architecture, and the `qwen3_5` dense family covers the 9B and
    Qwen3.8 27B. On one GB10 it decodes Qwen3.6-35B-A3B at 141–179 tok/s against vLLM's 101–122 (their numbers).
    That family runs one request at a time.
  - It reads only MLX 4-bit, group-64 weights on CUDA: a 35B-A3B is ~21 GB, tight on 24 GB.
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
- [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench) (NVIDIA;
  [paper](https://arxiv.org/abs/2603.19173), [code](https://github.com/nvidia/sol-execbench),
  [data](https://huggingface.co/datasets/nvidia/SOL-ExecBench)): 235 CUDA kernel problems taken from 124 real models,
  each scored against an analytic **speed-of-light** bound for a B200, not against a software baseline. It's our
  roofline method (measure against the hardware ceiling) turned into a benchmark.
  - Databricks reports #1 on all 4 tracks with Kernel Design Agents: frontier models (Claude writing, Codex reviewing)
    in a self-hillclimbing loop, about $70K in tokens. They say frontier models are still far ahead of open models
    at kernels. That's from their announcement, not verified here.
  - Why it works: the verifier is hard and cheap. The kernel must be correct and is timed against a fixed ceiling, so
    an agent can iterate unattended. It's the same `hillclimb` loop as in the eval-design entry.
  - Songbird angle: SQL has the same kind of verifier. A query's result can be checked against ground truth and its
    runtime measured, so text-to-SQL and query tuning can hillclimb the same way. Blackwell-only, so the problems
    themselves are out of reach on the 3090.
- [sudoingx Bonsai 2 PTQ1_0 + MTP](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF) (and
  [bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu)): the 5.95 GB ternary Bonsai 2 with Qwen3.8-27B's
  MTP head grafted back on, for 8–16 GB cards (measured on an RTX 3060 12 GB and a 5060 Ti 16 GB).
  - The graft works because Bonsai's residual stream stays in the original basis (its RMSNorm weights match Qwen's), so
    a head trained on fp16 activations still reads sensible ones. Draft acceptance is 0.45–0.95, lower than on stock
    Qwen.
  - The lesson is the "verification wall". Speculative decoding only pays if checking 3 tokens costs about the same
    as 1, which is true when decode is limited by memory reads. The ternary kernel was compute-bound on unpacking, so
    3 tokens cost 2.3–3× and MTP gained +1.6–8%. Their kernel
    ([PrismML-Eng/llama.cpp#218](https://github.com/PrismML-Eng/llama.cpp/pull/218)) cuts that to 1.55× and single
    decode goes 26 → 40 tok/s; with MTP, 50 tok/s on a 3060. It's the same compute-bound unpacking that holds EXL3 to
    53% of its ceiling here, and it's the Module 2 "ternary GEMV" final boss, already solved in the open.
  - Batch-invariant numerics: batched verification changes floating-point summation order, so greedy text can differ
    at near-ties. `GGML_CUDA_BATCH_INVARIANT=1` makes head-on and head-off byte-identical.
  - The Bonsai GGUF's chat template defaults to reasoning effort **xhigh**, which injects "think carefully…". At that
    setting their build tasks spent the whole budget thinking and answered nothing; `medium` fixed it. Our Bonsai
    reasoning-probe run got that xhigh line while the EXL3 runs didn't, so it's being rerun at `medium`.
  - Self-reported figures, with probe scripts and sweeps in the repo. There's a prebuilt sm86 tarball for this 3090.
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
- [Recursive Multi-Agent Systems](https://arxiv.org/abs/2604.25917) (Zou, Pan, Qiu, Lu, Diao, Jiang, Tong, Zhang,
  Buehler, He, J. Zou; April 2026, revised July 2026; [site](https://recursivemas.github.io/),
  [code](https://github.com/RecursiveMAS/RecursiveMAS) (MIT), [checkpoints and data](https://huggingface.co/RecursiveMAS)).
  - Frozen off-the-shelf models (1–9B: Qwen2.5-Math, Qwen3/3.5, Llama 3.2, Gemma 3) pass last-layer hidden states to
    each other instead of text, through a small trained two-layer residual "RecursiveLink" (~13M parameters).
    - The inner link feeds a model's own hidden state back in as its next input embedding, a latent "thought" in the
      style of Coconut.
    - The outer link maps one model's hidden size to the next one's. The last model loops back to the first, and only
      the final round decodes text.
  - Training runs in two loops. The inner loop regresses latent thoughts toward the embedding distribution. The outer
    loop unrolls all rounds and backpropagates the final answer's cross-entropy through the frozen models.
  - Claimed results, over 9 benchmarks (math, science, medicine, search, code) with gold answers:
    - +8.3% average accuracy over text-based multi-agent baselines
    - 35–76% fewer tokens
    - 1.2–2.4× faster
    - about $4 of training per system
  - **For us:** the baselines are small-model text teams, not one strong model, and the tasks are single-answer QA,
    not multi-turn tool use.
    - It isn't a quantization or a build. Latent rounds need an engine that passes hidden states, and there's no
      OpenAI-style text or tool-call API in the middle.
    - The inner link is the relevant piece. R2 reasons 40–50% longer than Q8, and a latent thought could shorten that
      thinking.
    - Trying it on Ornith would mean training the link against the bf16 model on a pod (a hybrid GDN/MoE, which the
      paper never uses). Serving would need TensorFold to prefill embeddings instead of token ids.
    - Worth trying only after the R1/R2 agent evals show whether reasoning length, not quality, is what holds a
      low-bit build back.
- [CUDA-L2: Surpassing cuBLAS for Matrix Multiplication through RL](https://arxiv.org/abs/2512.02551) (Su, Li, Wang,
  Wang, Li, Shum; Dec 2025, v4 Aug 2026; [kernels](https://github.com/ornith-ai/CUDA-L2), MIT, under the same
  `ornith-ai` GitHub org as the Ornith models).
  - An LLM writes HGEMM kernels, and RL uses measured speed as the reward, across 1,000 (M, N, K) shapes.
  - Only the kernels are released, not the RL pipeline. That's 3,741 files for A100, RTX 3090 and H100, fp16 inputs
    with fp16 or fp32 accumulate, and each kernel is tuned for its own GPU.
  - The 3090 set covers 736 shapes with M ≥ 64. Its own CSV claims 1.21× over `torch.matmul` on average (median
    1.21×) and 1.13× over cuBLASLt auto-tuning. The gains are biggest on small shapes.
  - **For us:** it doesn't drop into TensorFold.
    - Decode runs 1–16 rows over quantized weights and is bound by memory, not math.
    - Prefill uses our own dequantizing kernels, not fp16 HGEMM, and Ornith runs in bf16.
    - Worth reading: the 3090 kernels' tiling and pipelining as ideas for `qmm_prefill8`, since prefill (~4,100 tok/s)
      is what a returning or new long agent turn waits on. Also the method, speed as an RL reward, which is our
      hand-driven hill-climb automated. That would need their unreleased pipeline, plus our bit-exact gate as a hard
      constraint.
- [Context Language Models](https://arxiv.org/abs/2609.37725) (Shao et al., UW / Meta / MIT / Trillium; Sep 2026;
  [code](https://github.com/facebookresearch/context-language-models), CC BY-NC 4.0).
  - The agent's context is a file the model can edit with ordinary shell commands. It can delete, rewrite or
    summarize any part, and when it leaves the file alone, new tokens are appended as usual. Context management becomes
    the model's own behavior instead of a harness rule like "summarize at 80% full".
  - Zero-shot with existing models (Qwen3.6-27B, Qwen3.5-9B, frontier APIs), run as a Harbor agent (`clm-minimal`):
    - BrowseComp-Plus: +11.4% accuracy on 21.5% fewer FLOPs.
    - Terminal-Bench 2.1: matches the best baseline on 70% of its FLOPs.
    - TBLite: 73.7% vs 67.0%.
  - Small models don't do it well untrained. Qwen3.5-9B zero-shot is 6 points *below* the summarizing harness.
    - Stepwise GRPO fixes that: success first, then cheaper successful runs ranked higher. It takes the 9B from 28.8%
      to 42.5%, on 1.34 vs 2.19 PFLOPs a question.
  - Suffix Cache Reuse (an SGLang patch) keeps the cache for text that survives an edit, instead of prefilling
    everything after the edit point again:
    - Attention KV spans are moved with RoPE re-rotated, up to 6 spans per edit.
    - Linear-attention (GDN) layers keep a snapshot from before the edit, so they still carry the deleted text.
    - The result is 35% less server compute. It is approximate by design: stale state is kept.
  - **For us:**
    - It backs Prime's thesis. Context handling that the model learns in training beats harness logic bolted on,
      and small models need the training. That matches our pilot, where an untrained Ornith lost with
      prime_agent's extra harness features.
    - Ornith 35B-A3B is the paper's Qwen3.5/3.6 hybrid family with 3B active. Expect it to behave like their 9B:
      a likely loss zero-shot, so try it only as a trained behavior.
    - Serving cost on the 3090: TensorFold's cache is exact and prefix-only. Every mid-context edit re-prefills
      from the edit point, at ~4,100 tok/s, so an edit at 10K of a 50K context costs ~10 s. Suffix Cache Reuse
      would change the numbers, so it would have to pass the reasoning probe, not the bit-exact gate.
    - The RAM swap already snapshots GDN state per conversation, which is half the machinery.
    - It doesn't touch R1's current problem: its failures run out of turns, not context.
    - The code is non-commercial. Reimplementing the idea is fine, but copying their code into Smart Data isn't.
- [Raven](https://github.com/EverMind-AI/Raven) (EverMind; Apache 2.0; paper arXiv 2609.33439): "the harness of
  harnesses", a host agent that orchestrates its own agents (research, code, design, on-call) and 13 third-party ones
  over ACP, CLI or OpenAI-compatible APIs, with EverOS (a local-first Markdown memory store) across sessions.
  - The interesting part is the Curator, recursive self-improvement of the *harness*. The agent loop splits into four
    modules (Memory: what a turn sees; Planning; Capability: which tools a turn gets; Action), and the Curator rewrites
    them in rounds. A change is installed only after it passes a check, and the reference answer is stripped from the
    signals it sees.
  - Evidence is thin. The README doesn't name the models behind its numbers (DataAgentBench 0.8762 Pass@1; nanochat
    val_bpb −5.8% in the same 20-minute, one-GPU budget), and nothing is reported for small or open models.
  - **For us:** it's the opposite lever to Riel's thesis. Raven adapts the harness to the model; we want to train the
    model to the harness. Ornith 1.5's "scaffold construction" does both. A Curator-style search over prime_agent's
    prompt and tools for R1s is a no-training baseline that training should beat, but the Curator itself needs a
    strong model, so it would run on an API.
- [Qwen-Image-2.1 viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo): DMD step distillation, LoRA merge precision loss.
- [WorkflowEvals](https://evals.typesafe.ai/) (TypeSafe, the company behind Jev;
  [code](https://github.com/typesafe-ai/WorkflowEvals), Apache 2.0;
  [data](https://huggingface.co/collections/typesafe/workflowevals)): 705 synthetic cases in four business workflows:
  invoice approval (150), customer-service routing (204), agent-trace review (111) and security-alert triage (240).
  - Fixed Python code runs each workflow. At each gated step the model only answers Jev-style questions (yes/no,
    choice, or a 0–n score, given as probabilities), and thresholds in the code pick the actions. The score is an exact
    match of the action set against GPT-6 Astra and Claude Fable 5.1's averaged answers. There's no judge at run time,
    so scoring is free and deterministic. It measures agreement with two frontier models, not correctness.
  - It isn't agentic: no tool calls, turns or planning. It tests the decision points of a workflow the code already
    owns, which is the Smart Data pattern of an LLM as a classifier inside code.
  - [EvalSafe O*NET](https://huggingface.co/datasets/typesafe/evalsafe-onet) is the same idea for reading documents:
    150 synthetic documents tagged with O*NET occupations (not O*NET data) and 7,500 questions, scored against the
    same consensus. Apache 2.0.
  - Leaderboard (site, 2026-09-28): Sol 74.1%, Opus 5 73.1%, Terra 67.9%, Jev 67.8% at $0.0004 and 0.4 s a case,
    DeepSeek V4 Pro and Flash 65.5% and 64.4%. There's no Qwen or small open model.
  - Weak evidence: it's a vendor benchmark in its own model's question format. The labels are model-made on synthetic
    cases, the snapshot is a day old, three of the four datasets have no license, and the site's "prompt" column
    can't be run from the repo.
  - Use: a gate for the quantized Ornith builds that's closer to Smart Data work than the reasoning probe. A local
    server needs a small patch. `--base-url` works for TypeSafe only, `OPENAI_BASE_URL` sends Responses API calls,
    and answers must be strict JSON (llama.cpp can enforce that, TensorFold's server can't). The full set is ~6–7M
    input tokens and 0.6M (thinking off) to 3M (thinking on) output tokens, so it's a pod job. The 240 security cases
    are the local version. It's also a ready harness for the Jev vs Kev vs Julia plan.
- [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) (Hugging Face and
  Liquid AI, 2026-10-01; [code](https://github.com/adithya-s-k/FineEnvs)). The same weights score very differently
  across agent harnesses: GLM-5.2 gets 23% vs 52% on SWE-bench Pro. Training in one harness mostly helps that harness.
  - What they built: GRPO through unmodified Claude Code, Codex, OpenCode and Mini-SWE-Agent. A capture proxy records
    exact tokens and logprobs, Harbor supplies tasks and sandboxes, and TRL trains.
  - Setup: LFM2.5-2.6B on 1,000 SmolDataEnvs data-analysis tasks, 2×H100, 32-46 h, scored on 250 held-out tasks.
  - Results:
    - Multi-harness RL went 42 → 54% pass@1, gaining under all four harnesses.
    - OpenCode-only RL gained mostly in OpenCode (34 → 58%).
    - SFT on Qwen3.8-27B's successful rollouts helped less: 47.5% for OpenCode-only, 43.1% for multi-harness. The
      multi-harness SFT model fell from 62 to 45% under Mini-SWE-Agent.
  - Reward: correctness plus a bonus of at most 0.1 for fewer tool calls on correct answers. In 18-23% of groups this
    bonus was the only contrast, because all 8 rollouts were correct. Tool calls fell 31%.
  - Without the bonus, the Qwen3.5-2B runs drifted from 13 to 41 calls, and 35-58% of steps taught nothing (whole
    groups all right or all wrong).
  - Claude Code's history rewrites turned each rollout into about 8 training rows.
  - Weak evidence: one seed per setup, one task family, unequal data between runs. RL vs SFT differs in data source and
    compute as well as method.
  - For us:
    - It matches our flat self-distill round 1 (SFT on a three-harness mix). It also says a prime_agent-trained model
      should be checked under pi for lock-in.
    - Our round-2 data, 8 rollouts per task, has GRPO's group shape. Tasks solved 3-5 of 8 are the ones RL learns from.
      Picking the shortest solve is a crude version of their efficiency bonus.
    - TB2's 69 training tasks are too few for RL. SmolDataEnvs is a 1,000+ task pool of data-analysis work, which is
      close to Smart Data. Its SFT data ([SmolDataEnvs-multiharness-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft))
      is public, with no license shown.
- [Prime Inference](https://www.primeintellect.ai/blog/prime-inference) (Prime Intellect, 2026-10): Prime's serving
  platform, built to close the train → serve → collect traces → retrain loop. It has serverless and reserved endpoints,
  is OpenAI-compatible at `api.pinference.ai/api/v1`, and runs on Blackwell. The blog gives no prices or terms; the
  `/models` endpoint does.
  - Checked 2026-10-02: GLM-5.3 costs $1.40 in, $4.40 out and $0.26 cached-read per 1M tokens. It has 1M context and
    reasoning is always on (low/high/max, default max). It returns `logprobs` with `top_logprobs` on reasoning tokens
    too. The catalog also has GLM-5.3-Flash, Kimi K3, DeepSeek V4 Pro and V4.1 Flash, Qwen3.8-max and MiMo V2.5.
  - The GLM-5.3 license is MIT-style. Its only extra condition is a security review for API businesses earning over
    $10B a year, so training on its outputs is allowed.
  - For us: this is the teacher endpoint for SFT on teacher traces. GLM's tokenizer isn't Ornith's (Qwen3.5), so its
    logprobs can't drive token-level or on-policy distillation directly. The teacher contributes whole trajectories.
- [Kled AI datasets](https://www.kled.ai/datasets): a marketplace selling exclusive-rights data licenses at quoted
  prices. It has film and animation video, 50M de-identified US medical records, 12M radiology reports, 200M
  claims/billing entries and 8M papers. Sourcing and consent details aren't published.
  - For us: not useful now. There are no agent trajectories, spreadsheet or ETL tasks, or open licenses. It's only
    worth revisiting if a Smart Data client is in healthcare and wants a gym built from claims-like tables.
