# Tuning one-token decode (2026-10-14)

TensorFold fork branch `lfm2`, commits 97385e7b through the split-K commit. Each step was checked against the
reference before keeping it: KL against bf16 stayed at 0.141–0.145 (the same-weights reference itself is 0.152),
and every greedy run stayed coherent.

| step | short prompts tok/s | 14.7K-token prompt tok/s |
|---|---|---|
| first cut | 198 | — |
| router: a block per expert instead of a block per row | 263 | — |
| CUDA graph of the whole token step, next token picked on the GPU | 287 | 166 |
| own attention kernel that only reads the filled part of the cache | 300 | 264 |
| router also writes the expert plan (one launch fewer per layer) | 314 | 273 |
| expert matmuls: warps split the work and add up once at the end | **330** | **286** |

llama.cpp on QAD-Q4_0 does 270 on short prompts. The bytes ceiling is 566.

At long context the attention kernel matters most. The old kernel launched work for every slot of the cache, empty
or not, and handled each value one at a time. The new one walks only the filled part in 32-key tiles shared by the
four query heads of each key head. At 14.7K tokens it took prefill from 952 to 2,106 tok/s and decode from 166 to
264.

## Where a token's 3.0 ms goes now

From an nsys profile of 512 code tokens:

| part | per token | share |
|---|---|---|
| expert matmuls (gate/up, then down and the weighted sum) | ~1.45 ms | 48% |
| dense matmuls (conv in/out, attention q/k/v/o, two dense layers, head) | ~1.0 ms | 33% |
| norms (81 a token at ~2.4 us) | 0.19 ms | 6% |
| router | 0.18 ms | 6% |
| attention, conv, small glue | 0.2 ms | 7% |

The matmuls move their bytes at 55–65% of the card's bandwidth. They are small, about 2 to 14 MB each, so every
launch spends a microsecond or two starting up and draining. Changing how they split the work did not help:
2, 4 or 8 warps per tile, 2 to 4 loads in flight, and the dense kernels' K splits. The defaults or near them were
best each time. Two things are left for the one-token path. One is overlapping each kernel's start with the end
of the one before (programmatic dependent launch). The other is fusing the norms into their neighbours. Both are
worth maybe 10–15%.

The bigger lever now is drafting. Checking several guessed tokens in one pass reads the weights once for all of
them.

Reproduce: `tf-lfm2-cuda run ~/models/lfm25-24b/mlx4 --prompts prompts.json --max-tokens 256 --context 4096` and
`--prompts long_prompt.json --context 16384`.
