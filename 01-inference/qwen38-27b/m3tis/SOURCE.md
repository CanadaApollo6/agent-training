# Source

`bin/tensorfold-native` is built from TensorFold PR #608's Zig engine plus the M3TiS changes:

- Source: https://github.com/CanadaApollo6/TensorFold/tree/m3tis-qwen27-3090 (commit c7c3f7c8)
- Base: https://github.com/machinegenieorg/TensorFold/pull/608

Build (Zig 0.17.0, CUDA 13.4 nvcc):

```bash
zig build native -Dtarget=x86_64-linux-gnu.2.31 -Dcpu=x86_64_v3 -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86
```

The kernels in `share/tensorfold/cuda/sm86/` are TensorFold 0.6.6's Triton glue kernels, captured on an RTX 3090
with Triton 3.8 by the capture scripts in `zig/tests/cuda/qwen27/`.
