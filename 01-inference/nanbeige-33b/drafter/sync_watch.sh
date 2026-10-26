#!/bin/bash
# Copies the pod's generated responses home every 10 minutes and exits loudly when the pod stops answering
# (preemption, termination) or generation finishes. Lesson of 2026-10-25: a spot pod died with 45 min of
# unsynced output and nobody noticed for two hours.
#
#   bash sync_watch.sh <user@ip> <pod id>
H=$1 POD=$2 DIR=$(dirname "$0")/data/live; mkdir -p "$DIR"
O="-i $HOME/.ssh/prime_ed25519 -o ConnectTimeout=20"
fails=0
while true; do
  if rsync -qz -e "ssh $O" "$H:data/gen-s*.jsonl" "$DIR/" 2>/dev/null; then
    fails=0; echo "$(date +%T) synced $(cat "$DIR"/*.jsonl | wc -l) rows"
    ssh $O "$H" 'grep -q GEN_DONE /tmp/gen-0.log 2>/dev/null' 2>/dev/null && { echo "GEN_DONE"; exit 0; }
  else
    fails=$((fails + 1)); echo "$(date +%T) sync failed ($fails)"
    if [ $fails -ge 3 ]; then
      echo "POD DOWN: $(cd /tmp && timeout 60 uvx -q prime pods status $POD --plain 2>&1 | grep Status)"; exit 1
    fi
  fi
  sleep 600
done
