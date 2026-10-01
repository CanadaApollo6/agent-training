#!/bin/bash
# Terminate a Prime pod once a process exits (its eval is done), and log it in /tmp/prime-spend.log.
#   setsid nohup ./terminate_after.sh <pid> <pod id> <note> &
PID=$1; POD=$2; NOTE=$3
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} ($NOTE) exit=$?" >> /tmp/prime-spend.log
