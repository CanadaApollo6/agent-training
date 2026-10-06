#!/bin/bash
# tl_watchdog.sh [MAX_S]: the docker runtime doesn't enforce the rollout timeout on a hung sandbox (a pilot container
# ran 5 hours), so every 5 minutes remove vf-* containers older than MAX_S (default 3000: 30-minute rollouts plus the
# 5-minute verifier and slack). Logs to logs/tl-watchdog.log.
MAX=${1:-3000}
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
while true; do
  now=$(date +%s)
  docker ps --filter name=vf- --format '{{.Names}}' | while read -r c; do
    s=$(date -d "$(docker inspect -f '{{.State.StartedAt}}' "$c")" +%s 2>/dev/null) || continue
    if [ $((now - s)) -gt "$MAX" ]; then
      img=$(docker inspect -f "{{.Config.Image}}" "$c"); docker rm -f "$c" > /dev/null && echo "$(date +%T) removed $c ($img, up $((now - s))s)" >> logs/tl-watchdog.log
    fi
  done
  sleep 300
done
