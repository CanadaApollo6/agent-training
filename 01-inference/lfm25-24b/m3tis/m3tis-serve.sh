#!/usr/bin/env bash
# M3TiS: LFM2.5-24B-A2B (Liquid AI) on one RTX 3090, OpenAI-compatible server at http://127.0.0.1:${PORT:-8080}/v1
# The weights are in this folder. Sampling defaults to Liquid's (generation_config.json); add --temperature 0 for
# greedy replies, which also turns on drafting from the text itself.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/bin/tensorfold-native" serve "$here" \
  --context "${CONTEXT:-131072}" --parallel 1 --port "${PORT:-8080}" \
  --backend cuda --no-update-check "$@"
