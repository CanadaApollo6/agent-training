#!/bin/bash
# race.sh JOB_A JOB_B: two submissions of the same run on different hardware; as soon as one leaves SCHEDULING, cancel
# the other so only one ever bills. Prints the winner and exits.
A=$1 B=$2; HF="uvx -q --from huggingface_hub hf"
stage() { $HF jobs inspect $1 2>/dev/null | grep -oE "'stage': '[A-Z_]+'" | cut -d"'" -f4; }
while true; do
  for x in "$A $B" "$B $A"; do set -- $x
    s=$(stage $1)
    if [ -n "$s" ] && [ "$s" != SCHEDULING ]; then
      $HF jobs cancel $2 > /dev/null 2>&1
      echo "$(date -Iseconds) TERMINATE hfjob $2 (rl smoke: other hardware started first, never ran)" >> /tmp/prime-spend.log
      echo "winner $1 (stage $s); cancelled $2 (stage $(stage $2))"; exit 0
    fi
  done
  sleep 30
done
