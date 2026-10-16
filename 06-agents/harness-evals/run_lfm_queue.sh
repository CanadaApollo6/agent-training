#!/bin/bash
# Runs lanes from a queue across servers: one worker per local port, each taking the next lane when its last one ends,
# so 6 half-attempts finish on 5 servers without a server idling. A lane is "<label> <first task index> <count>";
# each runs through run_lfm_lane.sh at one conversation at a time. Progress: logs/<queue name>.log.
#
#   setsid nohup ./run_lfm_queue.sh <queue name> "<port> <port> ..." "<lane>" "<lane>" ... > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
NAME=$1; PORTS=$2; shift 2
Q=logs/$NAME.queue; LOG=logs/$NAME.log
printf '%s\n' "$@" > "$Q"
take() {  # pops the first lane off the queue, under a lock
  flock "$Q.lock" sh -c "head -1 '$Q'; sed -i 1d '$Q'"
}
for port in $PORTS; do
  (
    while lane=$(take) && [ -n "$lane" ]; do
      read -r label first count <<< "$lane"
      echo "$(date +%T) start $label on $port" >> "$LOG"
      ./run_lfm_lane.sh "$label" "$port" "$first" "$count" > "logs/$label.summary" 2>&1
      rc=$?
      echo "$(date +%T) end $label on $port rc=$rc solved=$(awk '$2>=1' "logs/$label.summary" | wc -l)/$count" >> "$LOG"
    done
  ) &
done
wait
echo "$(date +%T) queue done" >> "$LOG"
