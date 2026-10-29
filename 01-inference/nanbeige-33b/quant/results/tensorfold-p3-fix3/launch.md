# Run or resume the repaired gate

No command in this document was submitted to HF or Prime during implementation.
Stage the reviewed engine bundle, wrapper and both lock files together. Use a fresh output prefix for this repair.
The entrypoint now requires these additional checksum-pinned setup inputs.

- `image.txt` contains an immutable CUDA development image reference ending in `@sha256:<digest>`.
- `bootstrap.tar.xz` extracts to `uv` and `python/bin/python3.12`. Its archive checksum pins both tools.
- `debs/` contains the complete offline dependency closure for git, build-essential, mountpoint and required runtime libraries for that image. Each .deb has its own SHA256SUMS entry. `dpkg` performs no network resolution.
- `llama-gate.lock` and `llama-hf.lock` are copied verbatim from the reviewed engine.
- Existing model/drafter, histories, Zig archive, engine bundle, pod and smoke script inputs retain their SHA256SUMS entries.

The image, bootstrap and .deb set have not been staged or installed here. A CPU `DRY=1` job must validate that setup before GPU qualification.
The output bucket must be mounted writable at `/out`. Logging starts there before setup, so the first minute is durable.

After setting `IN_BUCKET`, `OUT_BUCKET`, and `CONTAINER_IMAGE` to the staged bucket prefixes and the exact value of `image.txt`, run the following commands.
Keep credentials in the local HF token file. The command passes the token through stdin.

```bash
export ENGINE_COMMIT=a5454ed3931caa5dee27ef45c468f44f8ea76b82
printf 'HF_TOKEN=%s\n' "$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")" |
  uvx --from 'huggingface_hub==1.10.1' hf jobs run --detach --secrets-file - \
    --flavor l40sx1 --timeout 210m --name nanbeige-full-gate-fix3 \
    -e ENGINE_COMMIT="$ENGINE_COMMIT" -e CONTAINER_IMAGE="$CONTAINER_IMAGE" \
    -e OUT_BUCKET="$OUT_BUCKET" -e SM=89 -e RESUME=0 -e JOB_SECONDS=12600 \
    -v "$IN_BUCKET:/inputs:ro" -v "$OUT_BUCKET:/out" \
    "$CONTAINER_IMAGE" bash /inputs/llama_hf_job.sh
```

For resume, use the same command and same inputs with `-e RESUME=1`.
The absolute deadline starts again for the new container. Completed jobs are accepted only after content and artifact validation.
For a CPU setup rehearsal, use `--flavor cpu-basic`, `--timeout 30m`, `-e DRY=1`, and `-e JOB_SECONDS=1800` with a separate output prefix.
For an A6000, use SM=86 in the operator's Prime container launcher with the same entrypoint and writable `/out` bucket mount. No Prime API command is inferred here.

Run the following CPU commands locally from the engine root.

```bash
python3 tools/zig/llama_full_gate.py --dry-run --out /tmp/nanbeige-fix3-restart
python3 -m unittest discover -s tools/zig -p 'test_llama*.py'
~/.local/share/zig-x86_64-linux-0.17.0/zig build tf-llama-replay -Doptimize=ReleaseFast -j4 --summary all
python3 tools/zig/llama_sample_gate.py --power-trials 128 --out /tmp/nanbeige-fix3-power.json
DRY=1 LOCAL_DRY=1 W=/tmp/nanbeige-job-dry OUT=/tmp/nanbeige-job-dry/out bash tools/zig/llama_hf_job.sh
```

A direct gate invocation inside an already prepared GPU container uses this command. The absolute deadline must be set at container start, before setup.

```bash
: "${GATE_DEADLINE:?set at container start, before setup}"
/workspace/exl3/bin/python tools/zig/llama_full_gate.py \
  --model /inputs/model --draft /inputs/draft --histories /inputs/histories.json \
  --pod /inputs/serve_exl3.py --smoke /inputs/smoke_serve.py \
  --baseline-runner zig-out/bin/tf-llama-test --reference-python /workspace/exl3/bin/python \
  --sm 89 --work /workspace/work --out /out/receipts --budget-minutes 210 \
  --deadline "$GATE_DEADLINE" --timeout 2400
```

An incomplete selection produces PARTIAL. A deadline, failed check, failed copy or failed upload produces nonzero exit and FAIL or BUDGET receipts.
P1 decode speed is a separate nonblocking verdict. A slow result is reported as `p1_speed=FAIL` and does not convert correct model outputs into a correctness failure.
