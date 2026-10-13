#!/usr/bin/env bash
# M3TiS: Qwen3.8-27B on one RTX 3090, OpenAI-compatible server at http://127.0.0.1:${PORT:-8080}/v1
# Downloads the 4-bit weights (~15 GB) and the DFlash2 draft model (~3.6 GB) on first run.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v hf >/dev/null || { echo "needs the Hugging Face CLI: pip install -U huggingface_hub" >&2; exit 1; }
model="${M3TIS_MODEL:-$(hf download Vontra/Qwen3.8-27B-MLX-4bit --quiet)}"
drafter="${M3TIS_DRAFTER:-$(hf download z-lab/Qwen3.8-27B-DFlash2 --quiet)}"
# 4-bit context store: the full 262K window fits in 24 GB
export TF_QWEN27_KV_BITS="${TF_QWEN27_KV_BITS:-4}"
exec "$here/bin/tensorfold-native" serve "$model" --drafter "$drafter" \
  --context "${CONTEXT:-262144}" --parallel 1 --port "${PORT:-8080}" \
  --backend cuda --no-update-check "$@"
