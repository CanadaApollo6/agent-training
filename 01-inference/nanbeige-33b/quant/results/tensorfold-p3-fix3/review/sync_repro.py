import subprocess
from pathlib import Path
source=Path('tools/zig/llama_hf_job.sh').read_text()
body=source[source.index('sync_out() {'):source.index('(while sleep 600;')]
root=Path('/tmp/nanbeige-review-a3c96094/sync');root.mkdir(exist_ok=True)
script='''set -euo pipefail
W=/tmp/nanbeige-review-a3c96094/sync
OUT=$W/receipts
WORK=$W/work
OUT_BUCKET=stub://bucket
mkdir -p "$OUT" "$WORK"
rm() { :; }
mock_hf() { if [[ "$3" == "$WORK" ]]; then echo WORK_FAILED; return 17; fi; echo RECEIPTS_UPLOADED; }
HF=(mock_hf)
'''+body+'''
rc=0
sync_out || rc=1
echo final_rc=$rc
exit "$rc"
'''
(root/'repro.sh').write_text(script)
p=subprocess.run(['bash',str(root/'repro.sh')],capture_output=True,text=True)
(root/'result.txt').write_text(p.stdout+p.stderr);print(p.stdout,p.stderr)
assert p.returncode==0 and 'WORK_FAILED' in p.stdout and 'RECEIPTS_UPLOADED' in p.stdout
