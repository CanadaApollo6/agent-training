"""Write the local P2 receipt index and bind it to the final source and binaries."""
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama')
OUT = Path(__file__).resolve().parent
QUANT = OUT.parent.parent
source = json.loads((OUT / 'source-manifest.json').read_text())
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == source['head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
for row in source['files']:
    assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def link(name):
    return f'[{name}]({OUT / name})'


slice_result = json.loads((OUT / 'slice-serve/summary.json').read_text())
server_vram = json.loads((OUT / 'slice-serve/server-vram.json').read_text())
oracle_vram = json.loads((OUT / 'slice-serve/oracle-vram.json').read_text())
blocked = json.loads((OUT / 'full-serve/summary.json').read_text())
prompts = json.loads((ROOT / 'zig/src/server/fixtures/llama/prompts.json').read_text())
outputs = json.loads((ROOT / 'zig/src/server/fixtures/llama/outputs.json').read_text())
host = (OUT / 'host-final.log').read_text()
summary = re.search(r'Build Summary: (.+)', host).group(1)
assert 'NondeterministicMemoryUsage' in host
assert 'NondeterministicMemoryUsage' in (OUT / 'exl3-base.log').read_text()
assert '19/19 tests passed' in (OUT / 'parity.log').read_text()
assert '121/121 steps succeeded' in (OUT / 'build-final.log').read_text()
assert '24/24 steps succeeded' in (OUT / 'build-plain.log').read_text()
assert '0 new' in (OUT / 'lean-new.txt').read_text()
assert slice_result['state'] == 'PASS' and slice_result['greedy_equal'] == 128
assert blocked['state'] == 'BLOCKED'
commits = subprocess.check_output(['git', 'log', '--reverse', '--format=%H %s', 'fd409155..HEAD'], cwd=ROOT, text=True).splitlines()
large = [(p.relative_to(OUT).as_posix(), p.stat().st_size, p.is_symlink())
         for p in sorted(OUT.rglob('*')) if p.is_file() and p.stat().st_size > 1_000_000]
artifacts = {name: dict(bytes=size, sha256=digest(OUT / name), symlink=symlink) for name, size, symlink in large}
for name in ['parity.log', 'host-final.log', 'build-final.log', 'build-plain.log', 'lean-new.txt',
             'exl3-base.log', 'unicode-before.log', 'slice-serve/summary.json', 'full-serve/summary.json']:
    artifacts[name] = dict(sha256=digest(OUT / name), bytes=(OUT / name).stat().st_size)
extra = [ROOT / 'tools/zig/llama_gpu_run.py', ROOT / 'tools/zig/llama_reference.py',
         QUANT / 'serve_exl3.py', QUANT / 'smoke_serve.py', QUANT / 'models/3.5-nb/config.json']
manifest = dict(head=head, created_utc=datetime.now(timezone.utc).isoformat(), source=source,
                artifacts=artifacts, extra_sources={str(p): digest(p) for p in extra},
                acceptance=dict(prompts=len(prompts), prompt_ids=sum(len(c['ids']) for c in prompts),
                                raw_outputs=len(outputs), slice=slice_result, full=blocked))
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
file_table = '\n'.join(f'| [{r["path"]}]({ROOT / r["path"]}) | {r["lines"]} |' for r in source['files'])
large_table = '\n'.join(f'| `{name}` | {size:,} | {"Read-only source symlink" if sym else "Generated here"} |'
                        for name, size, sym in large)
commit_text = '\n'.join(f'- `{line}`.' for line in commits)
readme = f'''# Nanbeige TensorFold P2

The greedy OpenAI-compatible serving path is implemented and verified on the CPU and a two-layer model slice.
Full 52-layer tool use, server throughput, and 128-token equality remain **BLOCKED** by shared GPU memory.
A TB2 harness that sends positive temperature must use temperature 0 for this implementation.
Positive-temperature sampling is explicitly rejected before SSE opens.

The source is branch `nanbeige`, final commit `{head}`.
P1 is `{source['p1']}`. The lean baseline is `31ce9eb3`.
These are local commits. No push, issue, pull request, or external message was created.
The original checkpoint stayed read-only. Only task-owned processes were signaled.
No CUDA operator, weight arithmetic, precision mode, or second scheduler was added.

## Acceptance results

| Check | Result | Evidence |
| --- | --- | --- |
| 1. Prompt tokens | PASS. {len(prompts)} conversations, {sum(len(c['ids']) for c in prompts)} IDs, zero differences from HF. | {link('parity.log')} |
| 2. Output parsing | PASS. {len(outputs)} raw texts through the actual pod parser, TF parser, and both JSON and SSE handlers. | {link('parity.log')} |
| 3. Full-model bash turn and decode speed | BLOCKED. No full-model tool or tok/s claim. | {link('full-serve/summary.json')} |
| 3. Slice HTTP plumbing | PASS. Solo/concurrent 16-token replies equal, streamed IDs equal, cancellation leaves the next request equal. | {link('slice-serve/summary.json')} |
| 4. Full-model 128-token equality | BLOCKED. | {link('full-serve/oracle.log')} |
| 4. Slice equality | PASS. All 128 tokens match serial generation through both chat and completions, using the exact 195-token P1 chat prompt. | {link('slice-serve/greedy-chat.json')}, {link('slice-serve/greedy-server.json')} |
| 5. Request settings | PASS for the supported greedy path and explicit refusals below. | {link('parity.log')}, {link('slice-serve/rejections.json')} |

The prompt corpus includes system/user, the model's default system, prior assistant reasoning and its alias,
reasoning embedded in content, JSON-string tool arguments, malformed JSON arguments, consecutive tool results,
tools present and absent, mixed string/text parts, and the exact P1 prompt.
It exercises `preserve_thinking=false`, `enable_thinking=false`, and `tool_call_format=json`.
The existing Jinja engine renders the checkpoint template correctly for every case. Its implementation is unchanged.
Chat encoding adds no tokenizer post-processor prefix. Completion text adds a missing BOS and preserves an existing one.
Message-boundary `<|im_start|>` tokens remain part of the model's template.
The primary EOS is 166101. The pod's alternate stop 166102 is also honored.

The reference script extracts `prepare`, `typed`, `split_output`, and their regex constants from `serve_exl3.py` via AST.
It executes those actual definitions without importing Torch or starting the pod server.
It compares reasoning/content exactly and tool names plus decoded JSON arguments, excluding generated tool IDs.
The raw corpus covers string/number/integer/boolean/object/nullable parameters, Python object spelling,
multiple calls, missing thinking/call/parameter closers, unknown functions, malformed envelopes, trailing prose,
and Chinese/emoji output.
The focused build passes 19/19 tests, including the five P2 tests and imported module tests.
The stream boundary test checks all 35 valid split positions across its ASCII and Chinese/emoji examples.

{link('prompt-before.log')} reproduces the original reasoning-alias token mismatch.
{link('output-mutation.log')} records the intended assertion failures when generic thinking replaced the required-close policy.
That control predates the final UTF-8 fix. {link('parity-restored.log')} records its restored green run.
{link('unicode-before.log')} freshly reproduces a failing UTF-8 validity assertion from the new stream path.
The final {link('parity.log')} is green after holding back seven complete characters.

## GPU scope and admission

The actual native binary serves two original checkpoint layers with the original embeddings, head, and tokenizer.
The slice is a new safetensors file under this directory. The source weights were never edited.
The server uses context 1024, one lane, greedy decoding, no drafts, no prompt cache, and no keep-warm requests.
The serial oracle uses context 4096 and the same `generate` function as `llama-greedy`.
This verifies request/stream/state behavior and token equality on the slice. Its generated text is not model-quality evidence.

The final server run lasts {server_vram['seconds']:.2f} seconds, peaks at {server_vram['peak_process_vram_mib']:,} MiB,
and leaves at least {server_vram['minimum_free_mib']:,} MiB free.
The serial oracle lasts {oracle_vram['seconds']:.2f} seconds, peaks at {oracle_vram['peak_process_vram_mib']:,} MiB,
and leaves at least {oracle_vram['minimum_free_mib']:,} MiB free.
Both use an honest 3,000 MiB estimate and preserve the 2,048 MiB desktop reserve.
The concurrency receipt and server log show that the second request queues and receives its own solo token sequence.
The slice also verifies three pre-stream HTTP 400 refusals and reset after a disconnected stream.

The full-model attempt uses context 4096 to reduce KV memory from the default 16384-token setting.
Its guard requires 15,500 MiB plus the 2,048 MiB reserve, or 17,548 MiB free.
The final admission says `{blocked['admission']}`.
The command exits 2 for BLOCKED. Neither the full oracle nor the full server starts.
Full-model `smoke_serve.py`, its bash call, and throughput are **NOT RUN**.
The prepared full run measures five 128-token single-stream replies and retains server decode rates separately
from the smoke client's latency including prefill. No slice speed is substituted for full-model throughput.

## Sampling and request settings

| Field | Behavior |
| --- | --- |
| `temperature` | 0 or omitted uses greedy. Positive values return HTTP 400 before SSE. Negative values are refused. |
| `top_p`, `top_k`, `min_p` | Accepted under temperature 0; filters do not change argmax. Positive-temperature combinations are refused. `top_p` and `min_p` must be in [0,1]. |
| `presence_penalty`, `frequency_penalty` | Zero or omitted accepted. Nonzero returns HTTP 400 before SSE. |
| `repetition_penalty` | Numeric 1 or omitted accepted. Other values return HTTP 400 before SSE. |
| `max_tokens`, `max_completion_tokens` | Enforced across reasoning and answer. `max_completion_tokens` wins when both are supplied. Requests exceeding the context get an explicit context error. |
| `stop` | A nonempty string or list of nonempty strings is honored by the existing stop handler. |
| `chat_template_kwargs` | Passed into the existing template renderer. Non-object values are refused. |
| `tool_choice` | Auto/none supported. Required/named calls are explicitly refused because this adapter has no token grammar. |
| Concurrent requests | One serial lane owns model state. HTTP requests queue through the existing lane host. |

## Reproduce

These commands run from the assigned worktree. All GPU commands inside the checker go through `llama_gpu_run.py`.
The checker records exact expanded commands and VRAM receipts. It never signals a process it did not launch.

```sh
TF_ROOT={ROOT}
TF_QUANT={QUANT}
TF_OUT="$TF_QUANT/results/tensorfold-p2"
TF_ZIG=/home/riels/.local/share/zig-x86_64-linux-0.17.0/zig
cd "$TF_ROOT"
export ZIG_GLOBAL_CACHE_DIR="$TF_OUT/global-cache"
"$TF_ZIG" build native tf-llama-test -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -Doptimize=ReleaseFast -j4 --prefix "$TF_OUT/install" --cache-dir "$TF_OUT/host-cache" --summary all
/home/riels/exl3-local/bin/python tools/zig/llama_serve_reference.py "$TF_QUANT/models/3.5-nb" "$TF_QUANT/serve_exl3.py" --zig "$TF_ZIG" --results "$TF_OUT"
python3 tools/zig/llama_serve_check.py slice "$TF_QUANT/models/3.5-nb" "$TF_OUT/slice" --layers 2
python3 tools/zig/llama_serve_check.py serve "$TF_OUT/install/native/bin/tensorfold-native" "$TF_OUT/slice" "$TF_OUT/slice-serve" --need-mib 3000 --context 1024 --oracle "$TF_OUT/install/bin/tf-llama-test" --refs "$TF_QUANT/results/tensorfold-p1"
```

When the GPU has enough free memory, run this exact full-model command.
It first runs the serial 128-token oracle, then starts the native server, checks queue/cancellation behavior,
runs the supplied smoke test with concurrency 1, measures five single-stream replies,
and compares all 128 P1 continuation IDs through both endpoints.

```sh
python3 tools/zig/llama_serve_check.py serve "$TF_OUT/install/native/bin/tensorfold-native" "$TF_QUANT/models/3.5-nb" "$TF_OUT/full-serve" --need-mib 15500 --context 4096 --timeout 600 --full --smoke "$TF_QUANT/smoke_serve.py" --oracle "$TF_OUT/install/bin/tf-llama-test" --refs "$TF_QUANT/results/tensorfold-p1"
```

The shared host invocation is below. Its inherited allocation-test failure returns exit 1.

```sh
TF_LLAMA_MODEL="$TF_QUANT/models/3.5-nb" "$TF_ZIG" build test test-kernel-contract test-golden fuzz-cortheon -j4 -Dfatbins="$TF_ROOT/zig-out/fatbin" --prefix "$TF_OUT/install" --cache-dir "$TF_OUT/host-cache" --summary all
python3 tools/zig/lean_new.py "$TF_OUT/lean-base.txt" "$TF_OUT/lean-head.txt"
git diff --check
```

## Shared-code review and verification limits

Review `hf_text.zig` closely for arbitrary kwargs, BOS deduplication, and the profile's alternate EOS.
Review `model_text.zig`, `errors.zig`, `prompt.zig`, and `chat.zig` for request-local kwargs propagation, including probe renders.
Review `tool_parse.zig`, `tool_params.zig`, and `tool_qwen.zig` for the explicit XML mode and schema conversions.
The generic parser retains its old strict behavior. Nanbeige uses the scored pod's tolerant qwen3 behavior.
Review `server.zig`, `serve.zig`, `nanbeige.zig`, and `openai.zig` for profile selection, required-close reasoning,
UTF-8 streaming, final complete tool-call events, and pre-stream refusals.
The profile requires config model_type `llama` and `<|im_start|>` ID 166100.
`openai_client.zig` is an unchanged socket-output adapter extracted to keep `openai.zig` below 600 lines.
Review `families/llama/native.zig` and `native/cuda.zig` for rows=1 and greedy capability reporting.

The final native/oracle build passes 121/121 steps. The plain build passes 24/24 steps.
The final shared run reports `{summary}`. Other results are cached.
Its sole failure is `exl3_load.test.a scan that fails an allocation frees every group prefix it held`,
with `NondeterministicMemoryUsage` at `zig/src/core/exl3_load.zig:319`.
That file is byte-identical to `31ce9eb3`, and {link('exl3-base.log')} freshly reproduces the same assertion on that base.
`test-kernel-contract`, `test-golden`, and `fuzz-cortheon` succeed in the same final invocation.
The golden corpora report 322/335 and 55/55 equal, with zero unexpected differences and 13 inherited known differences.
The lean result is `426 problems, 426 on the base, 0 new`. {link('lean-new.txt')} is the comparison receipt.
`git diff --check` passes. All 28 changed files are at most 600 lines; the largest is 597.
No full-model, Metal, HIP, Windows, multi-GPU, or sampled-decoding qualification is claimed.
Independent review is NOT RUN in this implementation sub-agent. The parent can review the named shared-code changes.

Model the Domain kept typed request options and the existing single lane owner.
Prove It Works drove the source-extracted HF/pod comparisons and actual native HTTP checks.
Sequence Work into Verifiable Units kept the protocol, admission, P1 route equality, and UTF-8 correction in ordered commits.

## Changed files and commits

Line counts are for the final commit. Full source hashes are in {link('source-manifest.json')}.

| File | Lines |
| --- | ---: |
{file_table}

{commit_text}

## Files over 1 MB

The threshold is 1,000,000 bytes. Every retained file above it is listed here.
The tokenizer symlinks point to the original read-only assets. Build caches and the baseline source archive were removed
after verification. Installed binaries, the slice, token dumps, logs, source hashes, and receipts remain here.
{link('manifest.json')} records their identity. The full-model blocked attempts have no process-VRAM receipt because no process starts.

| File | Bytes | Ownership |
| --- | ---: | --- |
{large_table}
'''
(OUT / 'README.md').write_text(readme)
assert len(readme.splitlines()) <= 600
print(f'README lines={len(readme.splitlines())}; large files={len(large)}; source files={len(source["files"])}')
