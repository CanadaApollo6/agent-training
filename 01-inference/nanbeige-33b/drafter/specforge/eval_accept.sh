#!/bin/bash
# Held-out test of a Nanbeige DFlash2 checkpoint on a pod set up by train_setup.sh: exports it to an HF draft dir,
# serves Nanbeige (FP8, as in training) with SGLang's DFLASH decoder, and runs `specforge benchmark` on its five
# held-out sets (none were in the training prompts). Acceptance length = completion tokens / verify steps, the number
# z-lab reports (Qwen3.8-27B-DFlash2: 4.1-5.5). Then single-user speed on two sets, with and without the draft.
#
#   bash eval_accept.sh <checkpoint dir, e.g. ~/out/full/nanbeige-dflash2-full-step2278> [prompts per set, default 64]
#
# Writes ~/eval/<checkpoint name>/{accept-*.json,speed-*.json,serve-*.log}; prints EVAL_DONE when finished.
set -uo pipefail
ck=$(realpath "$1"); n=${2:-64}; name=$(basename "$ck")
out=~/eval/$name; mkdir -p "$out"
if [ -d /usr/local/cuda-13.0/compat ] && [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
fi
[ -d /usr/local/cuda-13.0/bin ] && export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export PATH=$HOME/sf/bin:$PATH
cd ~/SpecForge

draft=~/draft-$name
[ -s "$draft/model.safetensors" ] || CUDA_VISIBLE_DEVICES="" specforge export --to hf --checkpoint "$ck" \
  --draft-config configs/nanbeige4.3-33b-dflash2.json --output-dir "$draft" --embedding-source ~/nanbeige

# SpecForge pins SGLang 0.5.18, which has no DFlash2DraftModel; serve from SGLang main in its own venv:
#   uv venv ~/sgl --python 3.12 && SGLANG_BUILD_RUST_EXTS=none VIRTUAL_ENV=~/sgl uv pip install \
#     "sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
SGL_PY=${SGL_PY:-$HOME/sgl/bin/python}
serve() {  # serve <tag> [extra args...]: start SGLang on :30000 and wait until it answers
  local tag=$1; shift
  (cd /tmp && exec "$SGL_PY" -m sglang.launch_server --model-path ~/nanbeige --quantization fp8 --attention-backend fa3 \
    --context-length 16384 --mem-fraction-static 0.85 --port 30000 "$@") > "$out/serve-$tag.log" 2>&1 &
  srv=$!
  for _ in $(seq 120); do curl -sf localhost:30000/health >/dev/null && return 0; kill -0 $srv 2>/dev/null || break; sleep 10; done
  echo "SERVE_FAIL $tag"; tail -20 "$out/serve-$tag.log"; return 1
}
stop() { kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 5; }
bench() {  # bench <file tag> <dataset> <prompts> <concurrency>
  specforge benchmark --model ~/nanbeige --dataset "$2" --num-prompts "$3" --concurrency "$4" --enable-thinking \
    --max-new-tokens 2048 --output-json "$out/$1-$2.json" > "$out/$1-$2.log" 2>&1
  echo "$1 $2: $(tail -3 "$out/$1-$2.log" | tr '\n' ' ' | cut -c1-300)"
}

serve dflash --speculative-algorithm DFLASH --speculative-draft-model-path "$draft" --speculative-num-draft-tokens 8 \
  || exit 1
for ds in gsm8k math500 humaneval mbpp mt-bench; do bench accept $ds "$n" 16; done
for ds in humaneval mt-bench; do bench speed $ds 10 1; done
stop
serve plain || exit 1
for ds in humaneval mt-bench; do bench base $ds 10 1; done
stop
echo EVAL_DONE
