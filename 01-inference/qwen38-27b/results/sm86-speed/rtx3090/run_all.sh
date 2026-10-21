#!/usr/bin/env bash
# waits for >=14 GB free RAM and an idle GPU, then runs the 3090 alternation
while :; do a=$(free -g | awk '/Mem/{print $7}'); g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$a" -ge 14 ] && [ "$g" -lt 3000 ] && break; sleep 30; done
echo "start $(date +%T) avail ${a}G gpu ${g}MiB"
./alt.sh r1s 3 base qmm && ./alt.sh r1s 1 exp both && ./alt.sh q27 1 base qmm
echo "ALL_DONE $(date +%T)"
