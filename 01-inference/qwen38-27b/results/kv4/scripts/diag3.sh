#!/bin/bash
# diag3.sh: the served bf16 and q4 greedy next token at the 32K rows that flip in the gate.
set -u
until grep -q "DIAG2 DONE" ~/diag2.log; do sleep 20; done
P=probe:32833+32943+32944+32945
bash ~/serve.sh probe-bf16 ~/out/probe $P --parallel 1 --context 40960
bash ~/serve.sh probe-q4 ~/out/probe $P --kv-cache q4 --parallel 1 --context 40960
echo "DIAG3 DONE $(date +%T)"
