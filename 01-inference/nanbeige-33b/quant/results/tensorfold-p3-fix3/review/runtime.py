import sys,json
from pathlib import Path
sys.path.insert(0,'tools/zig')
from llama_gate_budget import estimate,calibrate
labels=['p4-100000-0']+[f'p3-{f}-kv{k}' for f in ['bf16','affine4'] for k in ['16','4']]+['p1-reference','p1-logits','p1-greedy','p1-speed','p2','p4-100000-1','p4-100000-2']+[f'p3-{f}-kv{k}-wire' for f in ['bf16','affine4'] for k in ['16','4']]
rows={l:estimate(dict(label=l)) for l in labels}
r=dict(seconds=sum(rows.values()),minutes=sum(rows.values())/60,setup_headroom_minutes=148-sum(rows.values())/60,by_job=rows,p3_draws_per_job=3*3*4*8192*2,draw_milliseconds_allowed=240/(3*3*4*8192*2)*1000,p4_logits_bytes=9*2*512*166912*4)
Path('/tmp/nanbeige-review-a3c96094/runtime.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
