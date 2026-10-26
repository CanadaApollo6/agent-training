#!/bin/bash
# Training-pod watcher (see sync_watch.sh for why): every 10 minutes copies the train log, TensorBoard events and the
# draft weights of each new checkpoint (training_state.pt, ~4 GB; the 24 GB optimizer file stays on the pod) home,
# and exits loudly when the pod stops answering or the run writes TRAIN_EXIT.
#
#   bash train_watch.sh <user@ip> <pod id> <run name, e.g. full>
H=$1 POD=$2 RUN=$3 DIR=$(dirname "$0")/data/live/train-$3; mkdir -p "$DIR/ckpt"
O="-i $HOME/.ssh/prime_ed25519 -o ConnectTimeout=20"
fails=0
while true; do
  if ssh $O "$H" true 2>/dev/null; then
    fails=0
    # Exit 23 (a source not there yet, e.g. before the first TensorBoard write) is fine; the pod answered.
    rsync -qrz -e "ssh $O" "$H:/tmp/train-$RUN-1gpu.log" "$H:out/$RUN/runs" "$DIR/" 2>/dev/null
    # SpecForge repoints *-latest after a save completes, so only checkpoints up to that step are whole.
    last=$(ssh $O "$H" "readlink out/$RUN/*-latest" 2>/dev/null | grep -oE '[0-9]+$')
    for c in $(ssh $O "$H" "ls -d out/$RUN/*-step[0-9]* 2>/dev/null" 2>/dev/null); do
      n=$(basename "$c"); [ -s "$DIR/ckpt/$n.pt" ] && continue
      [ -n "$last" ] && [ "${n##*step}" -le "$last" ] || continue
      rsync -q --partial -e "ssh $O" "$H:$c/training_state.pt" "$DIR/ckpt/$n.pt.part" && mv "$DIR/ckpt/$n.pt.part" "$DIR/ckpt/$n.pt"
    done
    echo "$(date +%T) $(grep -E '^step ' "$DIR/train-$RUN-1gpu.log" | tail -1 | cut -c1-12) ckpts=$(ls "$DIR/ckpt" | grep -c '\.pt$')"
    grep -q TRAIN_EXIT "$DIR/train-$RUN-1gpu.log" && { echo "$(grep TRAIN_EXIT "$DIR/train-$RUN-1gpu.log")"; exit 0; }
  else
    fails=$((fails + 1)); echo "$(date +%T) sync failed ($fails)"
    if [ $fails -ge 3 ]; then
      echo "POD DOWN: $(cd /tmp && timeout 60 uvx -q prime pods status $POD --plain 2>&1 | grep Status)"; exit 1
    fi
  fi
  sleep 600
done
