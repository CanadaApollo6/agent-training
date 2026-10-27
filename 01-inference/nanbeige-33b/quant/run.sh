#!/bin/bash
# On the quant pod after setup.sh, with our token rows in ~/rows (build_rows.py). Four EXL3 conversions of
# Nanbeige4.3-33B run side by side on one GPU, then each quant is scored against bf16 (model_diff: KL divergence,
# perplexity, top-K agreement) on wikitext-2, Nanbeige's own chat answers, and its agent trajectories.
#   3.0 / 3.5 / 4.0   stock calibration, head at 6 bits
#   3.5-nb            3.5 bits calibrated on Nanbeige's own text (cal-nb.safetensors) instead of the bundled mix
# Status lines go to stdout (CONVERTED, CONVERT_FAIL, KL_DONE, KL_FAIL, RUN_DONE); logs and summary.tsv go to ~/results.
set -uo pipefail
. ~/exl3/bin/activate
. ~/compat.sh
M=~/nanbeige R=~/results
mkdir -p $R ~/q ~/w

conv() {  # conv <name> <bits> [convert.py args...]
  local resume=(); [ -d ~/w/$1 ] && resume=(-r)
  if python ~/qkit/convert.py -i $M -o ~/q/$1 -w ~/w/$1 -b $2 -hb 6 "${resume[@]}" "${@:3}" > $R/convert-$1.log 2>&1 \
     && [ -f ~/q/$1/config.json ]; then
    echo "CONVERTED $1 $(du -sh ~/q/$1 | cut -f1)"
  else
    echo "CONVERT_FAIL $1"; tail -5 $R/convert-$1.log
  fi
}

kl() {  # kl <quant> <set: wiki2|gen|agent> <rows> [model_diff args...]
  local tag=$1-$2${4:+-${*:4}}; tag=${tag// /}
  [ -s $R/kl-$tag.log ] && grep -q "KL divergence (A, B):" $R/kl-$tag.log && return
  local rows=(); [ $2 != wiki2 ] && rows=(KL_ROWS=$HOME/rows/kl-$2.safetensors)
  if env "${rows[@]}" python ~/qkit/model_diff.py -ma ~/q/$1 -mb $M -r $3 -l 2048 "${@:4}" > $R/kl-$tag.log 2>&1; then
    echo "KL_DONE $tag $(grep -o 'KL divergence (A, B): *[0-9.]*' $R/kl-$tag.log | awk '{print $NF}')"
  else
    echo "KL_FAIL $tag"; tail -5 $R/kl-$tag.log
  fi
}

score() {  # every set for one quant, one after another
  kl $1 wiki2 100; kl $1 gen 100; kl $1 agent 40
}

conv 3.0 3.0 & conv 3.5 3.5 & conv 4.0 4.0 & conv 3.5-nb 3.5 -cd ~/rows/cal-nb.safetensors &
wait

for q in 3.0 3.5 4.0 3.5-nb; do [ -f ~/q/$q/config.json ] && score $q & done
wait
# Quantized KV cache, simulated on the agent rows (long context is where it is used)
for q in 3.0 3.5-nb; do [ -f ~/q/$q/config.json ] && { kl $q agent 40 -cq 8; kl $q agent 40 -cq 4; } & done
wait

python - "$R" <<'EOF'
import glob, os, re, sys
R = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"{R}/kl-*.log")):
    t = open(p).read()
    g = lambda pat: (re.search(pat, t) or [None, ""])[1]
    rows.append([os.path.basename(p)[3:-4], g(r"KL divergence \(A, B\): *([0-9.]+)"), g(r"A perplexity: *([0-9.]+)"),
                 g(r"B perplexity: *([0-9.]+)"), g(r"Top-K agreement, A vs B:\s+K = 1: *([0-9.]+)"),
                 g(r"per-token: median ([0-9.]+)"), g(r"p90 ([0-9.]+)")])
with open(f"{R}/summary.tsv", "w") as f:
    f.write("run\tkl_ab\tppl_quant\tppl_bf16\ttop1_agree\tkl_median\tkl_p90\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(open(f"{R}/summary.tsv").read())
EOF
du -sh ~/q/* >> $R/sizes.txt 2>/dev/null
echo RUN_DONE
