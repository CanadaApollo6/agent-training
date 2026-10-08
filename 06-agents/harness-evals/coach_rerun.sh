#!/bin/bash
# coach_rerun.sh JOB: the coaching test's runs lost to Prime killing the sandbox, rerun once each on R1s-SD server JOB.
# The coached arm lost 11 (compile-compcert 3; mteb-retrieve, pytorch-model-cli, rstan-to-pystan 2; hf-model-inference,
# portfolio-optimization 1): coach_lost_r1..r3.txt. The plain 32K arm lost 1 (compile-compcert), rerun as c32-r1.
# qemu-startup (setup 404s, both arms) and image requests (model behaviour, counted as fails) are not rerun.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
J=$1
export HF_TOKEN=$(cat ~/.cache/huggingface/token); U=https://$J--8080.hf.jobs/v1
log() { echo "$(date +%T) $*" >> logs/hf-series.log; }
for i in $(seq 180); do curl -sf -m 10 -H "Authorization: Bearer $HF_TOKEN" $U/models > /dev/null && break; sleep 20; done
for r in 1 2 3; do
  log "ornith35b-r1s-sd-coach-r$r prime_agent start (job $J)"
  TASKS_FILE=coach_lost_r$r.txt BASE_URL=$U KEY_VAR=HF_TOKEN TIMEOUT=3600 \
    EXTRA="--sampling.max-tokens 32768 --env.taskset.system-prompt $PWD/coaching/prime_agent_coach.md" \
    ./run_eval.sh tb2 prime_agent ornith35b-r1s-sd-coach-r$r 6 > /dev/null 2>&1
  log "ornith35b-r1s-sd-coach-r$r prime_agent done: $(grep -c 'reward=1' logs/tb2-ornith35b-r1s-sd-coach-r$r-prime_agent.log) solved"
done
log "ornith35b-r1s-sd-c32-r1 prime_agent start (job $J)"
TASKS_FILE=coach_lost_r3.txt BASE_URL=$U KEY_VAR=HF_TOKEN TIMEOUT=3600 EXTRA="--sampling.max-tokens 32768" \
  ./run_eval.sh tb2 prime_agent ornith35b-r1s-sd-c32-r1 1 > /dev/null 2>&1
log "ornith35b-r1s-sd-c32-r1 prime_agent done: $(grep -c 'reward=1' logs/tb2-ornith35b-r1s-sd-c32-r1-prime_agent.log) solved"
uvx -q --from huggingface_hub hf jobs cancel $J > /dev/null 2>&1; log "coach reruns: cancelled job $J"
echo "$(date -Iseconds) TERMINATE hfjob $J (coaching reruns done)" >> /tmp/prime-spend.log
