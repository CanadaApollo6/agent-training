# Source

`bin/tensorfold-native` is built from TensorFold PR #608's Zig engine plus the M3TiS changes and the R1s MoE port:

- Source: https://github.com/CanadaApollo6/TensorFold/tree/r1s-moe (commit edfb771d)
- Base: https://github.com/machinegenieorg/TensorFold/pull/608

Build (Zig 0.17.0, CUDA 13.4 nvcc), then `strip --strip-all`:

```bash
zig build native -Dtarget=x86_64-linux-gnu.2.31 -Dcpu=x86_64_v3 -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86
```

All CUDA kernels are compiled into the binary; no captured kernel set is used.

The weights were made from the bf16 [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
by an importance-weighted affine search (MLX affine format, group size 64), using the importance matrix
`Ornith-1.5-35B-A3B-imatrix.gguf` from [bartowski/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF).
`quantize-report.json` lists each expert tensor's weighted error next to plain round-to-nearest.
