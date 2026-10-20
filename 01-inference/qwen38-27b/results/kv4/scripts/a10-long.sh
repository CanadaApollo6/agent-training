#!/bin/bash
# a10-long.sh: after the diagnostics, the 24 GB card's ~250K recall and decode with q4 caches at a 262144 window.
set -u
until grep -q "DIAG3 DONE" ~/diag3.log 2>/dev/null; do sleep 20; done
LONG_PLAIN=1 bash ~/serve.sh q4-262k-deep ~/out/serve long:258048 --kv-cache q4 --parallel 1 --context 262144
echo "A10LONG DONE $(date +%T)"
