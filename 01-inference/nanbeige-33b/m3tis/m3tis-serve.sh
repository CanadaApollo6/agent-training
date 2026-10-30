#!/usr/bin/env bash
# M3TiS-33B: Nanbeige4.3-33B with a DFlash2 draft head, OpenAI-compatible server at http://127.0.0.1:${PORT:-8080}/v1
# The weights and the draft head are in this folder. Sampling defaults to Nanbeige's (temperature 1.0, top_p 0.95);
# add --temperature 0 for greedy replies. Both draft.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 4-bit context store; TF_LLAMA_KV_BITS=16 keeps it in bf16 with a shorter window
export TF_LLAMA_KV_BITS="${TF_LLAMA_KV_BITS:-4}"
exec "$here/bin/tensorfold-native" serve "$here" --drafter "$here/draft" \
  --context "${CONTEXT:-65536}" --parallel 1 --port "${PORT:-8080}" \
  --temperature 1.0 --top-p 0.95 --backend cuda --no-update-check "$@"
