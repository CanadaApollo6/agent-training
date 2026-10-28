# TensorFold llama P4a

Local commit `5bd3231f56d8baa4fb2b0409dd077aede44a97fd` on `nanbeige`, based on `9a5732e0`. RTX 3090, sm_86, CUDA 13.4.92, Zig 0.17.0.
KV4 remains opt-in and is not qualified for pilot quality until the full-model comparison runs.
Set `TF_LLAMA_KV_BITS=4` for `serve`. Unset or `16` keeps the original bf16 path. Other values refuse.

| Check | Result | Receipt |
|---|---|---|
| Build | Native and bounded runner built for sm_86 | `build-release.log`, final rebuild in `build-commit.log` |
| CPU quantization reference | Both head widths pack byte-identically and dequantize bf16-identically; guarded rows | `synthetic.log` |
| Synthetic attention | 56/8 heads, dim 128, 1 through 115,001 keys; max RMSE 0.005082, max absolute 0.020264; five identical repetitions | `synthetic.log` |
| Independent attention reference | FP64 over CPU-dequantized KV at 1, 2, 127, 128, 511, 512, 513, 2049 keys; max absolute 0.001882 | `synthetic.log` |
| CUDA contract | Codec and attention pass nonfinite, five-repeat/alias guards and shape checks; both controls red; six sm_86 launches match | `contract-restored.log` |
| Real layers | Two real layers, cache slots 50/51 of 52, three 2048-token pilot prefixes; 6144 tokens; worst attention relative L2 0.409626 | `real-late.log` |
| Full-model quality | NOT RUN; both GPU admission checks refused before launching | `full-long.log`, `full-short.log` |
| Full-model speed | NOT RUN; decode and prefill are recorded by the same blocked full-model tool | Same commands below |
| Server host | Fresh 1015/1015, unchanged P2 gate | `host-fresh.log` |
| Qwen CPU | 58/58 after correcting undefined callback fields in its recording fixture; runtime code unchanged | `host-fresh.log` |
| Shared runtime / llama | Fresh 260/260 and 7/7 | `host-fresh.log` |
| Native KV4 server | Two-layer checkpoint, JSON repeats, JSON/SSE equality, refusal and recovery pass | `wire-kv4/summary.json` |
| Broad CUDA CPU suite | One inherited EXL3 allocation-fault failure; reproduced on untouched 9a5732e0 | `cuda-host-final.log`, `baseline-cuda-host.log` |
| Lean | 426 inherited, zero new; changed modules at most 600 lines | `lean-new.log` |

## Reuse and arithmetic

The named Qwen files at this checkout have no CUDA KV4 path. The working implementation is
`TensorFold-zig`, commit `dd06e939`, `zig/kernels/cuda/kv4.cu`. Its 256-wide attn4 MMA layout cannot serve
128-wide llama heads unchanged. This change generalizes its H32 midpoint codec into shared CUDA modules.
A head has 4-bit midpoint codes and one fp16 scale per 32 values. Llama's row is 72 bytes, with no GQA,
RoPE or output-gate assumption. RoPE stays in the existing QK kernel, before quantization.
The attention path uses FP32 query rotation, online softmax, chunk partials and inverse output rotation.
It retains BOS and the current token in bf16. It does not add a second scheduler or change Qwen execution.

The local #799 experiment is different from the running attn4 path. Its Qwen gate missed mean KL 0.01
and worst KL 0.1. At 32K its mean/worst were 0.104/17.9; at 64K, 0.0803/11.8. A chunk/arithmetic-order
outlier persisted with raw KV; removing that outlier still left mean KL about 0.034. This change avoids
extra bf16 query/output rotations and runs both formats through the same serial token schedule, but
that does not establish model quality. The full-model gate reports mean, p99, worst, every top-token
flip/margin, and 128-token greedy agreement. It refuses nonfinite logits and missing references.
The bounded accumulation fixtures do not separate fp16 from FP32 sums. Their explicit zero-margin
settings report that limitation; direct FP32 codec equality and FP64 attention checks remain enforced.
Historical Qwen numbers are in `qwen-reference.json` and the source receipt it names.

## Capacity

`llama-capacity` uses the checkpoint tensor map and the production allocator budget, including sampler,
attention partials, exact BOS storage and transient EXL3 workspace. Weights occupy 14,356,819,968 device
bytes; the bf16 embedding stays in pinned host memory. KV costs 212,992 bytes/token in bf16 and 59,904
in q4, a 3.56-fold reduction, plus 212,992 fixed BOS bytes for q4.

| Budget | bf16 maximum | q4 maximum | Largest-prompt pilot fit |
|---|---:|---:|---|
| 24 GiB, 2 GiB free reserve, allocator ceiling | 43,219 | 153,351 | 31/57 vs 57/57 |
| Same, with an additional 512 MiB driver allowance | 40,700 | 144,424 | 31/57 vs 57/57 |

These are computed ceilings, not measured full-model physical fits. Existing desktop and Vesuvius
allocations reduce the available budget further. The recorded maximum is 115,488 tokens, and all 57
runs also fit with 128 reply tokens under either projected q4 ceiling. No lower-bit V or workspace
reduction is required by this arithmetic. The GPU guard adds the 512 MiB allowance to the allocator
estimate and requires 2048 MiB free beyond it. It refused even the short full-model runs today.

## Rerun commands

From the engine root, build with the supplied Zig toolchain and CUDA compiler:

```bash
~/.local/share/zig-x86_64-linux-0.17.0/zig build native tf-llama-test -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j4 --summary all
```

Full quality and speed are NOT RUN. These exact commands prepare the recorded review-7 token IDs,
then check current free GPU memory before either format is launched. Each invocation selects one
history. Use indices 0, 1 and 2 in separate bounded runs. Short uses 128 prompt tokens, long 20,000.

```bash
bash /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p4a/rerun-full.sh short 0
bash /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p4a/rerun-full.sh long 0
```

The full-model tool holds only one format at a time, scores the final 128 teacher-forced positions,
and generates 128 greedy tokens. It reports 127 timed decode forwards and total prompt processing
including those diagnostic head/readback calls. These are single-run rates, not a five-pair speed
comparison. P1's historical plain baseline is 28.12 tok/s, not a new measurement.
Large logits and temporary checkpoint data stay under `/tmp/nanbeige-p4a`, outside these receipts.
No full-model pass, GPU server quality, long-context physical fit, or new full-model throughput is claimed.

The first attention nonfinite plant poisoned every head and correctly failed the proof gate. The final
plant leaves 128 nonfinite and 768 finite outputs, all matching the reference. `contract.log` retains
the earlier fixture failure.

The original GPU server harness is stale against P2's accepted wire format because it requires a
removed `tensorfold.token_ids` field. Its failure is retained in `server-kv4.log`; the direct wire
check uses the current JSON/SSE fields. Setup and disk-quota failures are retained, not counted as
product failures. Only task-owned temporary files were cleaned. No Vesuvius process was signaled
or waited for, and no commit or push was made in agent-training.
