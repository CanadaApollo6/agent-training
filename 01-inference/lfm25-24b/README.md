# LFM2.5-24B-A2B on the 3090

Liquid AI's `LiquidAI/LFM2.5-24B-A2B` (released 2026-10-12, LFM Open License v1.0). Goal: a TensorFold Zig-engine
build that runs it as fast as the 3090 allows, as a possible M3TiS release.

## The model in numbers

- 24.0B weights, about 2.45B used per token. 40 layers: 30 short-convolution layers and 10 attention layers.
- 64 experts per MoE layer, 4 picked per token (sigmoid router with a bias), plus 2 dense MLP layers at the start.
- Only 10 attention layers with 8 KV heads of 64 dims, so the KV cache is about 20 KB per token at bf16:
  128K tokens is about 2.6 GB. Context is capped at 131,072.
- No MTP head. Liquid lists a "DSpark" drafter for it as "coming soon".
- Official builds: GGUF (incl. a quantization-aware `QAD-Q4_0`), MLX 4/5/6/8-bit (4-bit is affine group 64 with a
  6-bit embedding, our R1s kernel format), FP8, ONNX.
- License: free for commercial use only below $10M annual revenue (section 5). Derivatives may be redistributed
  with the license and notices.

## Results

- [llama.cpp baseline](results/llamacpp-baseline.md): 270 tok/s decode on QAD-Q4_0, 48% of the 566 tok/s ceiling.

Local files (not in git): `~/models/lfm25-24b/{bf16,gguf,mlx4}`; llama.cpp built in `01-inference/tools/llama.cpp`.
