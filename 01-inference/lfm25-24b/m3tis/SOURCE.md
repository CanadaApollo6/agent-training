# Source

`bin/tensorfold-native` is built from TensorFold PR #608's Zig engine plus the M3TiS changes and the LFM2 port:

- Source: https://github.com/CanadaApollo6/TensorFold/tree/lfm2 (commit 9115658a)
- Base: https://github.com/machinegenieorg/TensorFold/pull/608

Build (Zig 0.17.0, CUDA 13.4 nvcc), then `strip --strip-all`:

```bash
zig build native -Dtarget=x86_64-linux-gnu.2.31 -Dcpu=x86_64_v3 -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86
```

All CUDA kernels are compiled into the binary; no captured kernel set is used.

The weights are [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit)'s files,
unchanged, plus `embed4.safetensors`: the tied embedding and output head re-quantized from
[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)'s bf16 `model.embed_tokens` to 4-bit
affine, group size 64, by
[`make_embed4.py`](https://github.com/CanadaApollo6/agent-training/blob/main/01-inference/lfm25-24b/make_embed4.py)
(MLX's affine rounding: each group's larger-magnitude edge lands exactly on a code).
