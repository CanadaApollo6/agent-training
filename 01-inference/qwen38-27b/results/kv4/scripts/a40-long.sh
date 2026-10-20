#!/bin/bash
# a40-long.sh: after the q4 deep gate, stop the fp8 gate and serve q4 at 262144 for recall at 64K and 128K.
set -u
until grep -q "gate q4 EXIT" ~/out/gate-q4.log 2>/dev/null; do sleep 20; done
for p in $(pgrep -f "^bash /root/deep.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "gate-fp8.json fp8"); do kill $p; done
export HF_HUB_DISABLE_PROGRESS_BARS=1
~/.local/bin/uvx -q --from huggingface_hub hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 > ~/drafter.log 2>&1
sleep 5
bash ~/serve.sh q4-262k-mid ~/out/serve long:65536+131072 --kv-cache q4 --parallel 1 --context 262144
echo "A40LONG DONE $(date +%T)"
