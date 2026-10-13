#!/usr/bin/env bash
# M3TiS: Ornith-1.5-35B-A3B (R1s, 3-bit experts) on one RTX 3090, OpenAI-compatible server at http://127.0.0.1:${PORT:-8080}/v1
# The weights are in this folder; the built-in draft head (MTP) is on by default.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/bin/tensorfold-native" serve "$here" \
  --context "${CONTEXT:-262144}" --parallel 1 --port "${PORT:-8080}" \
  --backend cuda --no-update-check "$@"
