Per CONTRIBUTING, an arithmetic change needs an issue first, so we can agree on its gate. This proposes an **opt-in 4-bit attention cache** for the qwen27 CUDA engine (`TF_QWEN27_KV_BITS=4`, off by default). With it, Qwen3.8-27B serves its full 262K window on one 24 GB RTX 3090. With the bf16 cache it fits about 32K (more with machinegenieorg/TensorFold#5).

## What it is

- **Storage:** each (position, KV head) keeps a 144-byte row: Hadamard-rotated groups of 32 values, as 4-bit codes plus an fp16 scale. That is 18 KiB a position over the 16 attention layers, against 64 KiB in bf16.
- **Decode rounds:** pack the window's new keys and values the same way, and rotate the queries once. Then attend over the 4-bit cache in rotated space (shared 512-key chunks, each row's tail, an in-order merge), and rotate the output back once.
- **Prompt chunks:** pack into the cache, and attend with `prefill_attention`'s kernel over decoded slots.
- **Contracts:** drafted output still equals `"draft": false` output, and concurrent replies equal solo runs, because both sides use the same cache. It is not exact against the bf16 cache, which is why it is opt-in.

## Measurements so far

The kernels are ports of the CUDA code I wrote and measured first on 0.3.6's Python engine. The fidelity numbers below come from there, so the Zig port's own run is part of the gate below.

**Teacher-forced vs the bf16 cache** (one long real document, frozen token file, 3090):

| Depth | KL mean | KL, worst token | Same top token | Perplexity, bf16 / 4-bit |
|---|---|---|---|---|
| 1,024 | 0.0034 | 0.030 | 96.9% | 8.91 / 8.92 |
| 16,384 | 0.0055 | 0.045 | 96.9% | 5.71 / 5.74 |
| 49,152 | 0.0018 | 0.029 | 99.2% | 2.12 / 2.12 |

The top-token differences were near-ties. The check stops at 49K because the bf16 cache fits no more on 24 GB.

**Zig engine, 3090:**
- Drafted equals serial with the cache on (4/4).
- Served at 262K, it recalled a code planted in the middle of a 192K-token prompt. Peak use was 19.6 GB.
- Decode is ~14% faster than bf16 at 28K, because there are fewer bytes to read.

**Known weak spots, reported as found:**
- An earlier version of the 4-bit cache missed one exact lookup among thousands of near-identical records in a 159K-token prompt. No bf16 window that long fits 24 GB, so I can't tell whether storage or length caused it.
- A 20-problem math probe scored 15/20 at a 262K window with the 4-bit cache. 4-bit EXL3 with a bf16 cache scored 18/20, but that's a different engine and weights, so it isn't a clean comparison.

## Proposed gate

1. **Off by default.** It runs only under `TF_QWEN27_KV_BITS=4` (or a `serve` flag, if you prefer). With it unset, nothing changes, byte for byte.
2. **Exactness that must still hold with it on:** drafted equals plain, concurrent equals solo, and resumed equals fresh. Checked on the 3090 with `tf-qwen27-cuda draft/multi`, and on served requests.
3. **Fidelity on the Zig engine itself,** teacher-forced against the bf16 cache on public text, at 1K, 8K, 32K and the deepest bf16 window that fits. Record the KL mean and worst token, and every top-token difference with both margins. Proposed bar: mean KL ≤ 0.01 and worst ≤ 0.1 at each depth.
4. **Long-context recall** at 64K, 128K and ~250K with the cache on, with each prompt published.
5. **Memory and speed receipts** per CONTRIBUTING: the startup gate's numbers, peak device memory, and decode at 32K and 262K.

Questions:
- Is env-only fine, or do you want a `serve` flag?
- Should it stay CUDA-only and qwen27-only for now? On GB10 or the RTX PRO 6000 the bf16 cache already fits long windows, so I'd keep the default there.
- A related change: on pre-sm_90 cards, the rule that runs shared-input decode weights unsplit once they fill the card gives ~2-3%, but changes K splits. Should it go through the same kind of gate, in a separate issue?

Once we agree on the gate, I'll open the PR against #608's branch (after machinegenieorg/TensorFold#3 and #4 there) with those receipts.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
