# The Zig engine port (2026-10-14)

TensorFold fork branch `lfm2` (CanadaApollo6/TensorFold), commit 619c0acd: `families/lfm2` plus `tf-lfm2-cuda`.
It reuses R1s's 4-bit matmuls and grouped expert kernels. New parts: the short-convolution and router kernel
(`lfm2.cu`), an expert runner without a shared expert, and attention for 64-wide heads through the existing
window-attention kernel.

## Weights

The MLX 4-bit build, with one change: the tied embedding (which is also the output head) is re-quantized from bf16
to 4 bits, because the engine's kernels read 4-bit tables only. `make_embed4.py` writes `embed4.safetensors` next to
the MLX shards.

KL against the bf16 model, on 443 positions of a code answer (`ref.py`, teacher-forced):

| weights | KL mean | KL median | top-1 agree |
|---|---|---|---|
| MLX 4-bit (6-bit embedding) | 0.135 | 0.053 | 85.6% |
| MLX 4-bit, 4-bit embedding | 0.152 | 0.058 | 86.0% |
| Zig engine on the 4-bit-embedding weights | 0.137 | 0.056 | 87.4% |

The 4-bit embedding costs about 0.017 KL. The engine itself adds nothing: its fp32 sums land slightly closer to
bf16 than the reference's bf16 maths on the same weights.

## Correctness

The engine against `ref.py` on the same weights: KL 0.024 mean, 96.2% top-1 agreement over 443 positions.
Decoding 4 rows at a time gives bit-identical logits to 1 row at a time. Prompt chunks (the prefill matmuls) differ
from the decode path by KL 0.011.

## Speed, first cut

Greedy, 256 tokens, 3090, nothing tuned:

| prompt | prompt tokens | decode tok/s |
|---|---|---|
| code | 53 | 197 |
| chat | 29 | 198 |
| tools | 64 | 198 |

llama.cpp does 270 on QAD-Q4_0; the bytes ceiling is 566.

Reproduce: `compare_logits.py ref.npy zig.bin`, with `tf-lfm2-cuda logits ~/models/lfm25-24b/mlx4 --prompts ids.json
--out zig.bin` and `ref.py --weights mlx4 --embed4 --ids ids.json --out ref.npy`; speed with `tf-lfm2-cuda run
~/models/lfm25-24b/mlx4 --prompts prompts.json --max-tokens 256`.
