#!/bin/bash
# fixes_big.sh: the bigger harness-fix test. The 87 TB2 tasks that get a turn (tb2_noqemu_tasks.txt), 2 attempts each,
# fixes off vs fixes on (cutoff,check,image), both arms through fix_proxy.py. Each arm is split over two R1s-SD HF
# servers (tb2_noqemu_a.txt: 44 tasks, tb2_noqemu_b.txt: 43), so four servers and four fixes_run.sh drivers. Each
# driver cancels its job when done. Labels ornith35b-r1s-sd-big{off,on}{a,b}.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
P=8111
for arm in off on; do
    F=none; [ $arm = on ] && F=cutoff,check,image
    for half in a b; do
        N=big$arm$half
        J=$(TIMEOUT=10h ./hf_serve.sh submit sd-out/r1s-sd ornith35b-r1s-sd-$N | tail -1)
        echo "$(date -Iseconds) START hfjob $J a10g-large fixes big test $N, 10h cap, auto-cancel" >> /tmp/prime-spend.log
        nohup setsid ./fixes_run.sh $N $J $F tb2_noqemu_$half.txt 2 $P > logs/fixes_run-$N.out 2>&1 &
        P=$((P + 1))
    done
done
