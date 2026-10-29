# Parent-only GPU launch

This command has not been executed. Check the CPU DRY result in README.md before launching.

The pinned CLI requires click and typer, uses labels instead of `--name`, and reads `--secrets-file -` as a literal file.
The temporary symlink makes that file read stdin. The token never appears in arguments or output.

```bash
(
  set -euo pipefail
  export ENGINE_COMMIT=a5454ed3931caa5dee27ef45c468f44f8ea76b82
  export IN_BUCKET=hf://buckets/CanadaApollo6/nanbeige-gate-inputs/fix3-a5454ed3-20261029T183916Z
  export OUT_BUCKET=hf://buckets/CanadaApollo6/nanbeige-gate-outputs/gpu-fix3-a5454ed3-20261029T183916Z
  export CONTAINER_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
  submit_dir=$(mktemp -d)
  trap 'rm -rf "$submit_dir"' EXIT
  ln -s /dev/stdin "$submit_dir/-"
  cd "$submit_dir"
  printf 'HF_TOKEN=%s\n' "$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")" |
    uvx --python 3.12 --from 'huggingface_hub==1.10.1' --with click --with typer hf jobs run --detach --secrets-file - \
      --flavor l40sx1 --timeout 210m --label name=nanbeige-full-gate-fix3 \
      -e ENGINE_COMMIT="$ENGINE_COMMIT" -e CONTAINER_IMAGE="$CONTAINER_IMAGE" \
      -e OUT_BUCKET="$OUT_BUCKET" -e SM=89 -e RESUME=0 -e JOB_SECONDS=12600 \
      -v "$IN_BUCKET:/inputs:ro" -v "$OUT_BUCKET:/out" \
      "$CONTAINER_IMAGE" bash /inputs/llama_hf_job.sh
)
```

For resume, use the same inputs and GPU output prefix with `-e RESUME=1`.
The CPU rehearsal used the separate `dry-fix3-a5454ed3-20261029T183916Z` output prefix, `--flavor cpu-basic`, `--timeout 30m`, `-e DRY=1`, and `-e JOB_SECONDS=1800`.
