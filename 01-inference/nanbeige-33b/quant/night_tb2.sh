#!/bin/bash
# On a quant pod after setup.sh, with build_rows.py's rows (every row starting with <|im_start|>) in ~/rows and, if
# uploaded, last night's ~/q/3.5-nb. Rebuilds the own-text 3.5 bpw quant with the fixed calibration rows (3.5-nb2),
# scores it against bf16 on the same chat and agent rows as before, picks the better of 3.5-nb2 and 3.5-nb (lower
# agent KL), and serves it for the TB2 pilot on 127.0.0.1:8000 as nanbeige-exl3-a1..a3.
# Status lines: CONVERTED, CONVERT_FAIL, KL_DONE, KL_FAIL, CHOSEN, SERVE_DONE, SERVE_FAIL. Logs in ~/results.
set -uo pipefail
. ~/exl3/bin/activate
. ~/compat.sh
M=~/nanbeige R=~/results
mkdir -p $R ~/q ~/w

resume=(); [ -d ~/w/3.5-nb2 ] && resume=(-r)
if [ -f ~/q/3.5-nb2/config.json ] || python ~/qkit/convert.py -i $M -o ~/q/3.5-nb2 -w ~/w/3.5-nb2 -b 3.5 -hb 6 \
     -cd ~/rows/cal-nb.safetensors "${resume[@]}" > $R/convert-3.5-nb2.log 2>&1; then
  echo "CONVERTED 3.5-nb2 $(du -sh ~/q/3.5-nb2 | cut -f1)"
else
  echo "CONVERT_FAIL 3.5-nb2"; tail -5 $R/convert-3.5-nb2.log; exit 1
fi

kl() {  # kl <quant> <set> <rows>
  local log=$R/kl-$1-$2.log
  if KL_ROWS=$HOME/rows/kl-$2.safetensors python ~/qkit/model_diff.py -ma ~/q/$1 -mb $M -r $3 -l 2048 > $log 2>&1; then
    echo "KL_DONE $1-$2 $(grep -o 'KL divergence (A, B): *[0-9.]*' $log | awk '{print $NF}')"
  else
    echo "KL_FAIL $1-$2"; tail -5 $log
  fi
}
kl 3.5-nb2 gen 100 & kl 3.5-nb2 agent 40 & wait

agent_kl() { grep -o 'KL divergence (A, B): *[0-9.]*' $R/kl-$1-agent.log 2>/dev/null | awk '{print $NF}'; }
pick=3.5-nb2
# Last night's 3.5-nb scored 0.06585500 on these exact agent rows (kl-3.5-nb-agentbos.log)
if [ -f ~/q/3.5-nb/config.json ] && python -c "import sys; sys.exit(not float('$(agent_kl 3.5-nb2)' or 9) > 0.06585500)"; then
  pick=3.5-nb
fi
echo "CHOSEN $pick (3.5-nb2 agent KL $(agent_kl 3.5-nb2), 3.5-nb 0.06586)"

nohup python ~/qsrc/serve_exl3.py --model ~/q/$pick --name nanbeige-exl3-a1 --name nanbeige-exl3-a2 \
  --name nanbeige-exl3-a3 --port 8000 > $R/serve.log 2>&1 &
for i in $(seq 120); do
  curl -sf -m 5 127.0.0.1:8000/health > /dev/null && { echo "SERVE_DONE $pick"; exit 0; }
  kill -0 $! 2> /dev/null || break
  sleep 10
done
echo SERVE_FAIL; tail -20 $R/serve.log; exit 1
