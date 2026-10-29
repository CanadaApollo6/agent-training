# Nanbeige GPU gate input staging

The complete input set is uploaded to a private HF bucket. The CPU DRY job finished with HF status `ERROR`, exit 1, and durable `status.txt` containing `FAIL`.
The reviewed engine's `llama-hf.lock` omits click. Its installed HF CLI fails on its first sync with `ModuleNotFoundError: No module named 'click'`.
Repair that engine dependency and rerun the CPU rehearsal before any GPU launch. This staging task leaves the engine and both locks unchanged.
No GPU job was launched. No local GPU was used. Engine source files and commits are unchanged.

The reviewed engine is `a5454ed3931caa5dee27ef45c468f44f8ea76b82` on `nanbeige`.

| Item | Value |
| --- | --- |
| Private inputs mounted at `/inputs` | `hf://buckets/CanadaApollo6/nanbeige-gate-inputs/fix3-a5454ed3-20261029T183916Z` |
| Private DRY output mounted at `/out` | `hf://buckets/CanadaApollo6/nanbeige-gate-outputs/dry-fix3-a5454ed3-20261029T183916Z` |
| Reserved future GPU output | `hf://buckets/CanadaApollo6/nanbeige-gate-outputs/gpu-fix3-a5454ed3-20261029T183916Z` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7` |
| Input size, including SHA256SUMS | 20,998,793,879 bytes across 55 files |
| SHA256SUMS digest | `91af114b5406f7eaea0e55b2a04591cd8565ca01d471cbb2f527c5f3c39c116f` |
| CPU DRY job | [6ae393825ddcdc0830264492](https://huggingface.co/jobs/CanadaApollo6/6ae393825ddcdc0830264492) |

## Input verification

`SHA256SUMS` pins 54 files. The manifest itself is the 55th file.
`input-manifest.json` records every pinned file's size and hash. The remote prefix has exactly the expected file set and sizes.
The uploaded checksum manifest was downloaded and compared byte-for-byte before submission.

The model was uploaded directly from `01-inference/nanbeige-33b/quant/models/3.5-nb`, including its tokenizer files.
The drafter was uploaded directly from `01-inference/nanbeige-33b/drafter/data/draft-cool-step570`.
Its only two files are `config.json` and `model.safetensors`. The DFlash2 loader reads the former and opens the latter through `core.Checkpoint.openModel`.
No model or drafter copy was made locally.

The entrypoint and both lock files match `git show a5454ed3:tools/zig/<file>` byte-for-byte.
The bundle was created from a temporary bare copy of the engine repository. The shallow boundary was copied into that temporary repository before fetching upstream ancestors.
A fresh clone of the final bundle and detached checkout of the reviewed commit succeeded. See `bundle-check.log`.

`bootstrap.tar.xz` holds static musl uv 0.13.6 and the installed python-build-standalone CPython 3.12.14, build 20260814, under `python/`.
The uv source is `https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-unknown-linux-musl.tar.gz`.
Python comes from `~/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu` without bytecode caches.
The archive pins both tools and extracts to `uv` and `python/bin/python3.12`.

The Zig archive is the existing `~/.local/share/zig-dl/zig-x86_64-linux-0.17.0.tar.xz`.
It extracts with `--strip-components=1` to a binary reporting `0.17.0`.

## Real histories

Each history is one actual pre-generation request from a Nanbeige EXL3 TB2 pilot, reconstructed along the sampled node's parent chain.
No text was fabricated, padded, repeated, concatenated across runs, or truncated to achieve the required length.

| Task | Pilot | Trace | Call node | Tokens |
| --- | --- | --- | ---: | ---: |
| custom-memory-heap-crash | EXL3 a1, 1a1fed7e | cc10e4c943f54a25a6610e8f17a48620 | 108 | 115,488 |
| protein-assembly | EXL3 a1, 1a1fed7e | c84c1f42996f4a98b6de32e61d18a583 | 102 | 114,476 |
| build-pov-ray | EXL3 a2, 66d99a40 | da63f021f8b84078840ce5b900f45e3c | 113 | 113,322 |

The source files are `06-agents/harness-evals/outputs/primeintellect/terminal-bench-2--nanbeige-exl3-a1--pi--1a1fed7e/traces.jsonl` and the corresponding a2 directory `terminal-bench-2--nanbeige-exl3-a2--pi--66d99a40`.
`histories-provenance.json` records their SHA256 hashes and the exact parent chains. It also lists a fourth qualifying request, 108,205 tokens, which was not selected.

`prepare_histories.py` extracts the actual `prepare` and `text_of` functions from `serve_exl3.py` without importing inference code.
It converts the trace's flat tool schemas and tool calls to OpenAI shape, preserving schema `strict` fields, then renders with Transformers 4.57.1 and the model's own tokenizer and chat template.
It uses `tools=...`, `add_generation_prompt=True`, `tokenize=False`, then tokenizes with `add_special_tokens=False`, as the pod does.
All selected lengths exactly match the original request usage counts.

Every history begins with token 166100 and contains only integer IDs in `[0, 166912)`.
The three prefixes are distinct at 20K, 60K, and 100K. All BOS-plus-final-127-token suffixes are distinct.
The gate's own `--plan-only` execution passed without querying a GPU and prepared its complete 26-job plan. See `gate-preflight.json`.
No token dump is included in this receipt directory.

## Container verification

All Docker checks used `--runtime=runc`, without GPU flags or device mounts.
The image has CUDA nvcc 12.8.93. The local `/opt/cuda` nvcc is 13.4.92, used only for `--version`.
The engine's compiler flags require C++20. Seven selected engine kernels compiled for sm_89 with CUDA 12.8.93.
`nvcc-sm89.log` records successful fatbin compilation of `exl3_linear`, `sample`, `qk_rows`, `kv4`, `tree_attention`, `block_attention`, and `dense_rows`.
The entire native engine build in this image is NOT RUN.

APT resolved 29 missing or upgraded packages inside the exact pinned image using `--download-only --no-install-recommends`.
The requested set was git, build-essential, xz-utils, util-linux, mount, ca-certificates, libssl3, libffi8, libbz2-1.0, liblzma5, zlib1g, libreadline8, libsqlite3-0, and libgomp1.
`debs-resolve.log` records resolution. Each downloaded .deb has its own checksum entry.

A fresh container of the same image with `--network none` passed `dpkg --install /inputs/debs/*.deb` and `dpkg --audit`.
It also passed tool lookup, static uv and Python execution, Python SSL/SQLite/bz2/lzma/ctypes/readline/venv imports, uv venv creation, Zig extraction, and version checks.
`offline-setup.log` ends with `OFFLINE_SETUP_PASS`.

## CPU rehearsal and launch deviations

The job uses `--flavor cpu-basic --timeout 30m -e DRY=1 -e JOB_SECONDS=1800`, the reviewed commit, the exact pinned image, and the separate DRY output prefix.
Its command is `bash /inputs/llama_hf_job.sh`, with the input prefix mounted read-only and output prefix mounted writable.

The host CLI is `uvx --python 3.12 --from 'huggingface_hub==1.10.1' --with click --with typer hf`.
This pinned version has no `--name` option. Submission uses `--label name=nanbeige-full-gate-fix3-dry` instead.
It also interprets `--secrets-file -` as a literal file and initially failed before submitting a job with `FileNotFoundError: '-'`.
A temporary working directory containing the symlink `- -> /dev/stdin` preserves the requested stdin-only secret command.
The token was never included in arguments or printed. Only one job was submitted.

The job checked all 54 SHA256SUMS entries successfully, installed the .deb set, and synced both locked Python environments.
The inference environment installed 64 packages. The HF environment installed 21 packages, with no click package.
`sync_out` then failed on both its work and receipt uploads. The exit trap repeated those failing uploads and preserved mounted logs and FAIL status.
`CPU_DRY_PASS` was never reached. The HF job is terminal, so no running job remains.

Key lines in the downloaded `container.log` are:

```text
CONTAINER_START 1793299369 DEADLINE 1793301169
histories.json: OK
model/model-00001-of-00002.safetensors: OK
model/model-00002-of-00002.safetensors: OK
uv 0.13.6 (x86_64-unknown-linux-musl)
Python 3.12.14
Installed 21 packages in 281ms
ModuleNotFoundError: No module named 'click'
CONTAINER_END exit=1
```

`container.log`, `status.txt`, `dry-job.json`, and `dry-result.json` preserve the remote outcome.
The mounted output files were downloaded from the DRY prefix after the job became terminal.
This is an engine lock defect, so the staging was not altered and no redundant DRY job was submitted.

The fully filled GPU command, including the host CLI corrections, is in `gpu-launch.md`. It has not been executed and currently reproduces this dependency failure.

## Disk and spend

Staging file sizes peaked at 493,343,545 bytes. The pulled Docker image reported 14,518,513,191 logical bytes.
The image, temporary bundle repositories, static uv download, and large staged archives were deleted after upload and verification.
Observed free space under `/home` went from about 25 GiB to 13 GiB during setup and returned to about 24 GiB.
No model or drafter copy was made, and `/tmp` was not used for large artifacts.

HF currently lists cpu-basic at USD 0.01/hour. The 30-minute DRY job has a USD 0.005 runtime ceiling at that rate.
The container ran for approximately 169.369 seconds, from its logged start to the output object's last write, for estimated runtime spend USD 0.000470.
The entire creation-to-final-write interval was 207.665 seconds, corresponding to USD 0.000577 at that rate.
Actual billing is not available from the job receipt.
Spend is logged in `/tmp/prime-spend.log`. GPU spend is zero.
