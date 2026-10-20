#!/bin/bash
# a10-after.sh: short-prompt decode with q4 caches at the bf16 run's 40960 window, for a matched comparison.
set -u
until grep -q "A10LONG DONE" ~/a10-long.log 2>/dev/null; do sleep 20; done
bash ~/serve.sh q4-40k ~/out/serve openai --kv-cache q4 --parallel 1 --context 40960
echo "A10AFTER DONE $(date +%T)"
