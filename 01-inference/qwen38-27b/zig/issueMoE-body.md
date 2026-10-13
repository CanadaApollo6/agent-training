A coordination question before I build it, per CONTRIBUTING's new-family rule. It overlaps @jkuepker's plan on #491 and #785, so I'd like to agree on the split before either of us gets far.

## What I want to run

[Ornith 1.5 35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT), a `qwen3_5_moe` checkpoint with Qwen3.6-35B-A3B's shapes, on one RTX 3090 (sm_86, 24 GB). It is stored as MLX affine in groups of 64: routed experts at 3 bits, everything else at 4, routers and the shared-expert gate at 8, plus its MTP layer as `mtp-4bit.safetensors`. That is 15 GB of weights, which leaves the 3090 room for a long window.

It serves today on a fork of the Python 0.6 line with added 3-bit expert kernels. On the 3090 it decodes 416-535 tok/s drafted. KL against bf16 is 0.229, where bf16's kernel-noise floor is about 0.04. On 19 Terminal-Bench 2 tasks, 3 attempts each, it solves 10.0 a run against 10.3 for the Q8 GGUF. #286 froze that line, so I'd like it on the native engine.

## What exists on native

- **Layers:** ashhart/TensorFold#608's qwen27 CUDA engine already runs the Gated DeltaNet and gated attention layers this model shares with the 27B, with the 0.6.6 bytes. machinegenieorg/TensorFold#3 and #4 add sm_86 to it.
- **Routed experts:** `cuda/experts.zig`'s `Grouped` has `Affine4Experts`; there is no 3-bit or general-bit affine form.
- **MTP drafts:** Nemotron's `cuda_mtp.zig` and the lane core's drafted path.
- **Loading:** #626's loader reads `qwen3_5_moe` geometry, the MTP layer and MLX affine at 2 to 8 bits.

## Proposal and questions

1. **Where the engine lives.** One option is to grow #608's CUDA engine to the MoE shapes: routed experts in place of the dense MLP, and the MTP layer as the drafter in place of DFlash2. The other is a separate CUDA backend for the qwen3_5 family on #626's loader, as @jkuepker proposed on #491. I'm happy to do either, or to take a slice of @jkuepker's backend: the routed experts, the MTP drafter, or sm_86. Which do you want?
2. **Affine experts at other widths.** I'd add 3-bit (and if wanted 2- to 8-bit) affine experts beside `Affine4Experts` in `Grouped`. Each would have:
   - its own kernel and the 4-bit form's accumulation order;
   - an FP64 reference per projection class, with worst-column and aggregate error;
   - decode and prompt rows checked separately.

   Is that the gate, or does a new expert width count as an arithmetic change with more to agree?
3. **The exactness reference.** 0.6.6 reads only 4-bit group-64 checkpoints for this family. My proposal:
   - **4-bit:** the engine is byte-equal to 0.6.6 on every logit row, using `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` or Ornith's MLX 4-bit conversion. Drafted equals serial, concurrent equals solo, and resumed equals fresh.
   - **3-bit:** the same contracts, plus the FP64 references above. Add a teacher-forced fidelity run against bf16 at 1K, 8K and 32K, recording all top-token differences and margins.

   Does that work, or would you rather the 3-bit form wait until the 4-bit engine has landed?

Hardware I can test on: an RTX 3090 locally, and rented A6000, A100 and H100 cards. I have no DGX Spark or RTX PRO 6000.
