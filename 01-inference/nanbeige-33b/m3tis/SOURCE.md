# Source

`bin/tensorfold-native` is built from TensorFold's Zig engine plus the Nanbeige (Llama family) port, its EXL3 kernels,
the DFlash2 draft head support and the 4-bit context store:

- Source: https://github.com/CanadaApollo6/TensorFold/tree/nanbeige (commit aacb6300)
- Base: https://github.com/ashhart/TensorFold main at 31ce9eb3 (v1.1.8 plus 14 commits)

Build (Zig 0.17.0, CUDA 13.4 nvcc), then `strip --strip-all`:

```bash
zig build native -Dtarget=x86_64-linux-gnu.2.31 -Dcpu=x86_64_v3 -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86
```

All CUDA kernels are compiled into the binary for sm_86; no captured kernel set is used.

The weights are [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) converted by
CanadaApollo6 with exllamav3 1.6.5 to EXL3 at 3.5 bits per weight (6-bit output head), calibrated on Nanbeige's own
chat and agent text (`quantization_config.json` holds the settings). `draft/` is a DFlash2 draft head trained by
CanadaApollo6 on 36K Nanbeige4.3-33B responses with SpecForge. Recipes, data notes and every measurement are in
[agent-training/01-inference/nanbeige-33b](https://github.com/CanadaApollo6/agent-training/tree/main/01-inference/nanbeige-33b).
