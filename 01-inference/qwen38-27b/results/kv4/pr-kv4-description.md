DRAFT for Riel. Not sent. The gate fails as agreed; see "Gate results" and "Open question" before deciding to open it.

## What this changes

`--kv-cache q4` for `qwen3_5` on the CUDA engine, as asked on #799. bf16 stays the default, and the flag stays off unless asked for. Base: `d1018b31` (TensorFold 1.1.1 tree). Branch `qwen35-kv-cache-4bit`, four commits (`31ba5241`..`54b5e9be`):

- `qwen3_5: --kv-cache q4, H32-turned 4-bit attention caches with fp16 group scales`
  - `KvFormat` (`bf16`, `fp8`, `q4`) replaces the `kv8` bool in the engine, geometry, cost key, kv pool text and reuse layout.
  - A q4 row is 144 bytes for a 256-wide head: 128 bytes of 4-bit codes, element d's in byte d / 2 (low nibble first), then 8 fp16 group scales.
  - Each group of 32 is turned by a 32-wide Hadamard before rounding. Codes are clamp(floor(8 y / s) + 8, 0, 15) with s the group's fp16 absmax; a code reads back as (c - 7.5) s / 8. A non-finite group stores a NaN scale.
  - New `zig/kernels/cuda/qwen35_kv4.cu`: `tf_q35_rope4` (rope_rows with a 4-bit store), `tf_q35_attn_partial4`, `tf_q35k_compact4`.
    - The partial kernel turns the query once, reads keys in fours, and turns the output back once. The cache stays in turned space. The merge is the bf16 one.
  - Every launch goes through the existing dispatch (`KvKernels` per format). Lane rounds, drafts, windows, trees, keep/compact and the prompt cache take the same path as bf16.
  - Refusals: Metal, XPU and HIP refuse `--kv-cache q4`. So does any CUDA family whose `Options` has no `kv4`, and `--slide` with q4.
  - Kernel contract: pending 13.0 records for the three launches, like kv8's.
- `qwen3_5 kv-gate: the worst token's position and per-depth progress; kv pool test takes the format text`
  - `tf-qwen35-cuda <model> kv-gate <fixture.json> <out.json> <bf16|fp8|q4>` (`zig/tests/qwen35_kv_gate.zig`).
    - It runs the fixture through bf16 caches, then through the named format's, one engine at a time, teacher-forced.
    - It writes the mean, p99 and worst KL(bf16 || other) per depth, the worst token's position and top-1 agreement.
    - It writes every top-token difference with both margins, perplexity on each side, and recall of a planted fact.
    - It exits 1 if any depth misses the bar.
  - `tools/kv_gate_fixture.py` builds the named fixture (public text, below).
- `contract: nvcc 13.2 usage for the qwen35_kv4 launches`: measured on the 13.2 fatbins, sm_80/86/89/120/121.
- `qwen3_5_moe: refuse --kv-cache q4 until the routed model has a gate run of its own`
  - `qwen3_5_moe` shares the engine and would take the flag through the shared `Options`. It now refuses it, keeping this to the qwen27-only first slice from the thread.
  - I have no gate or receipts for the MoE.

## Gate results

**The bar is not met.** q4 passes mean KL at 1K, 8K and 258K. It misses mean KL at 32K, 64K and 128K, and worst-token KL at every depth past 1K. The fp8 cache already on main misses the same bar at 32K and 64K on this fixture.

Fixture `kv-gate tensorfold-md@d1018b31 seed 799 rows 256`:
- Text: every `*.md` in this repository at `d1018b31`, concatenated in `git ls-tree` order, each under a `=== path ===` header. That gives 258,305 Qwen3.8 tokens.
- Recall: at each depth d, a passphrase line is planted at d/2. The question sits at d, and the answer (9 tokens) is scored among the 256 rows from d.
- Built with: `uv run --with tokenizers python3 tools/kv_gate_fixture.py --repo . --commit d1018b31 --tokenizer <checkpoint>/tokenizer.json --out fixture.json`
- sha256 of the full fixture: `6fd31915815cd761046d03f7b3797a96d18302753014572e0716d5768c4a10d1`. The 8K and 64K fixtures are its prefixes.

Command: `tf-qwen35-cuda <checkpoint> kv-gate fixture.json gate-q4.json q4`.
- Checkpoint: `TensorFold/Qwen3.8-27B-MLX-4bit` at `22d8d538`.
- Depths 1K to 128K plus 258,048 ("the deepest bf16 window that fits": 258,305 tokens is the fixture's end, run on a 48 GB card so bf16 fits).
- The 1K to 64K numbers are identical on the A10 and the A40, digit for digit.

| depth | q4 mean KL | q4 p99 | q4 worst (position) | q4 top-1 | ppl bf16 / q4 | recall | fp8 mean / worst (same fixture) |
|---|---|---|---|---|---|---|---|
| 1,024 | **0.0065** | 0.046 | **0.085** (1176) | 96.1% | 11.01 / 11.00 | yes | 0.0020 / 0.071 |
| 8,192 | **0.0093** | 0.089 | 0.171 (8380) | 93.8% | 11.36 / 11.70 | yes | 0.0022 / 0.090 |
| 32,768 | 0.1040 | 0.895 | 17.9 (32944) | 93.4% | 29.11 / 29.43 | yes | 0.0480 / 11.3 (32944) |
| 65,536 | 0.0803 | 0.703 | 11.8 (65755) | 93.8% | 26.03 / 31.63 | yes | 0.0379 / 7.59 (65683) |
| 131,072 | 0.0459 | 0.854 | 2.29 | 95.3% | 94.68 / 98.38 | yes | not run |
| 258,048 | **0.0083** | 0.065 | 0.192 | 95.7% | 4.51 / 4.56 | yes | not run |

- bf16 against bf16 through the same tool: KL 0 at every depth, top-1 100%. The tool and the engine are deterministic.
- Every flip of more than 2 logits on either side sits 100 or more rows after the planted question. That is where the document resumes mid-sentence after the answer.
- Position 32944 is not a cache effect alone. Three checks show it:
  - A diagnostic build stored raw bf16 rows beside the 4-bit ones and read only the raw rows through the q4 kernel (the turn in fp32, nothing rounded). It still gives KL 24.5 there; the other depths fall to 0.0008 to 0.0016 mean.
  - Served bf16 (`/v1/completions`, greedy, the same 32,945 ids) predicts ` Tensor` at that row, while the gate's bf16 rows put `Each` 20 logits above it.
  - fp8 flips the same row (11.3).
  So a change in prompt chunking or arithmetic order alone moves that row by 20 logits. No format can hold a worst-token bar of 0.1 at 32K on this fixture.
- Without that row, q4's 32K mean is 0.034 and fp8's 0.0038. At 64K, without the worst row, q4's mean is 0.034 and fp8's 0.0083. That is the real cost of 4 bits here: about 4 to 9 times fp8's drift. At 64K it raises perplexity by 21%.
- Raw keys alone or raw values alone (the diagnostic build, 8K fixture) each still miss the 0.1 worst bar: K raw 0.0087 / 0.31, V raw 0.0053 / 0.14. Both halves contribute. [64K split: pending]
- Earlier, a Python engine on a different document read 0.0034 / 0.030 (1K), 0.0055 / 0.045 (16K) and 0.0018 / 0.029 (49K). It used the same scheme. This fixture is harder: perplexity is 4.5 to 95 here, against 2 to 9 there.

## Receipt

- Environment:
  - 24 GB card: NVIDIA A10 (sm_86, 23,028 MiB, driver 570.148.08 with the CUDA 13 compat libraries), Linux, CUDA 13.4 `nvcc`, Zig 0.17.0.
  - Deep reference: NVIDIA A40 48 GB (sm_86, driver 595.91.07).
  - **Not an RTX 3090.** The A10 is the same architecture and memory size class, but it has 600 GB/s against the 3090's 936. Its decode and prompt speeds are lower than a 3090's, and they are only comparable with each other.
- Memory (startup line, one stream):

  | window | bf16 | q4 |
  |---|---|---|
  | 8,192, `--parallel 2` | 1.21 GiB a stream, 18,537 MiB on the device | 0.85 GiB a stream, 17,801 MiB (peak 17,831) |
  | 40,960 | 3.21 GiB, 19,341 MiB | 1.41 GiB, 17,501 MiB |
  | 262,144 | does not fit (by the startup's arithmetic, bf16's largest window here is about 76K) | 5.21 GiB of 5.4 GiB free, 21,389 MiB idle, peak [258K prompt: pending] MiB |

- Long context with q4 at a 262,144 window (`longctx.py`: the fixture up to the planted answer, greedy, 128 tokens, drafted then `"draft": false`):

  | prompt | card | recall | drafted == plain | time to first token | decode, plain / drafted (tok/s) |
  |---|---|---|---|---|---|
  | 32,790 | A10 | yes | yes | 160.2 s (bf16 at 40,960: 133.8 s, recall yes) | 22.3 / 30.2 (bf16: 23.6 / 36.7) |
  | 65,558 | A40 | yes | yes | 346.7 s | 24.7 / 32.5 |
  | 131,094 | A40 | yes | yes | 1,237.3 s | 19.9 / 21.0 |
  | 258,070 | A10 | [pending] | not run (plain only) | [pending] | [pending] |

  - Prompts and codes are in `long-*.json`. At 131K the reply after the code repeats `The`. That happens on both drafted and plain runs, and there is no bf16 window there to compare against.
  - Prompt reading is slower with q4: 205 tok/s against 245 at 32K on the A10, and 556 s against about 430 s for the gate's 64K prompt. Like bf16's, the kernel reads every key again for each row of a 128-row pass, and q4 also decodes it each time. fp8's gate prompt took 1,168 s for the same 64K.
- Short-prompt decode (`bench_openai.py --tokens 128 --reps 3`, median tok/s, greedy, code / chat):
  - bf16 at 40,960: 103.0 / 77.4.
  - q4 at 262,144: 99.0 / 75.0.
  - q4 at 40,960: [pending].
  - bf16 on this branch equals bf16 on `d1018b31`: every reply hash is equal, greedy and seeded, in `bench_openai` at `--parallel 2 --context 8192` (98.9 / 77.8 against 99.0 / 77.5).
- Exactness with `--kv-cache q4` (A10):
  - `tf-qwen35-cuda` over 160 ids after a 1,500-token prefill:
    - `window`, `tree`, a resumed stream and a stream beside another 700-token one all equal `step` byte for byte;
    - a resumed prompt equals the fresh prompt.
  - `bench_concurrent.py --tokens 128 --reps 2 --levels 1,2 --alone --serial --temperatures 0,1.0` at `--parallel 2 --context 8192`: 24 of 24 alone and 12 of 12 serial equal, 0 failed.
  - `bench_cache.py --require-cold`: qualified, with first tokens and first outputs equal.
  - Drafted equals plain at 32K, 64K and 128K (above).
- Tests:
  - On the A40, at the branch head:
    - `zig build test-cuda-host`: all pass.
    - `zig build test`: all pass except three `tool_parse_test` timing checks ("... read in linear time", a 20 ms or 24x bound). They ran beside a GPU job on a shared host. They don't touch this change, and they passed on the A10.
  - On the A10, at the first two commits: `cuda_cost_identity: stable reads reuse a drafter, while a changed payload at the same path invalidates it` failed. It fails the same way on `d1018b31` on that machine, and it passes on the A40.
  - New tests: the q4 row bytes, the q4 refusal, and the routed-model refusal.
  - `tools/zig/lean_check.py`: 444 problems before and after. `zig fmt --check`: clean.
- Not run:
  - An RTX 3090 (see above). Windows. Metal, which refuses the flag.
  - `qwen3_5_moe`, which now refuses the flag.
  - fp8 past 64K.
  - nvcc 13.0 records (pending, as for kv8).

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo, all with the flag on.
- [ ] No precision traded for speed: the flag trades precision for context by design. It is off by default, and the gate above is not met.
- [ ] Prompt processing is no slower than the last release: bf16 is unchanged, but q4's prompt reading is 17% slower than bf16's at 32K.
- [x] The description says which platforms I ran (CUDA sm_86 on an A10 and an A40) and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.

## Open question

Your bar, mean ≤ 0.01 and worst ≤ 0.1 at every depth, is not met by q4 on a public fixture past 8K. It is not met by the fp8 cache already on main past 8K either, and at 32K one row moves 20 logits under a lossless change in arithmetic order. Which would you rather have?
1. A bar relative to fp8 on the same fixture (for example, mean within 3x of fp8's), with the worst row reported, not gated.
2. A fixture whose scored rows are unbroken text, with recall checked separately. All the confident flips here sit after the planted question.
3. A wider format (5-bit codes, or 8-bit keys with 4-bit values) that I gate the same way before you look again.
