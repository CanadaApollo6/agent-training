#!/bin/bash
# Keep SSH tunnels to the pod servers alive: every 20 s, any local port that isn't listening gets its tunnel restarted
# (detached, so it outlives the shell that started it). Each argument is <local port>:<pod ip>:<remote port>.
#
#   setsid nohup ./tunnels.sh 8101:216.81.200.120:8081 8102:204.12.169.194:8082 > /dev/null 2>&1 &
cd "$(dirname "$0")"
while true; do
  for spec in "$@"; do
    IFS=: read -r lport ip rport <<< "$spec"
    ss -ltn "sport = :$lport" | grep -q LISTEN && continue
    echo "$(date +%T) tunnel $lport -> $ip:$rport down; restarting" >> logs/tunnels.log
    setsid ssh -f -N -i ~/.ssh/prime_ed25519 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -L "$lport:127.0.0.1:$rport" "ubuntu@$ip" < /dev/null
  done
  sleep 20
done
