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
- [Qwen-Image-2.1 viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo): DMD step distillation, LoRA merge precision loss.
