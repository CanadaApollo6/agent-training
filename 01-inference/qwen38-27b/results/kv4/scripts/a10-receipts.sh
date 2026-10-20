#!/bin/bash
# The 24 GB card's served receipts, one server at a time (serve.sh); results under ~/out/serve.
set -u
O=~/out/serve; mkdir -p $O
BASEBIN=~/tf-base/zig-out/native/bin/tensorfold-native
echo "== exactness, q4, parallel 2"; LEVELS=1,2 bash ~/serve.sh q4-p2 $O conc,cache --kv-cache q4 --parallel 2 --context 8192
echo "== bf16 default: this branch"; bash ~/serve.sh bf16-p2 $O openai --parallel 2 --context 8192
echo "== bf16 default: base"; BIN=$BASEBIN bash ~/serve.sh base-bf16-p2 $O openai --parallel 2 --context 8192
echo "== 32K bf16"; bash ~/serve.sh bf16-32k $O openai,long:32768 --parallel 1 --context 40960
echo "== 32K q4"; bash ~/serve.sh q4-32k $O openai,long:32768 --parallel 1 --context 40960
echo "== 262K q4"; bash ~/serve.sh q4-262k $O openai,long:32768+65536+131072 --kv-cache q4 --parallel 1 --context 262144
echo RECEIPTS_DONE
