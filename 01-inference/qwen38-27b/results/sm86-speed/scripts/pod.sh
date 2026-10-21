#!/bin/bash
# pod.sh CMD...: run on the sm86-speed pod
exec ssh -i $HOME/.ssh/prime_ed25519 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20 ubuntu@157.151.232.64 "$@"
