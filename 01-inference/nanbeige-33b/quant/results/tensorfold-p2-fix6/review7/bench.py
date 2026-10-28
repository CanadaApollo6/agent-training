import os,subprocess,re,json,time,statistics
from pathlib import Path
p=Path('/tmp/nanbeige-review-9a5732e0')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_LLAMA_MODEL='/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb',REVIEW_INPUT=str(p/'bench-input.jsonl'))
results=[]
for repeat in range(5):
 for side in ['base','head']:
  binary=p/'base/routes' if side=='base' else p/'routes'
  path=p/f'bench-{repeat}-{side}.log'
  with path.open('w') as f: r=subprocess.run([str(binary)],env=env,stdout=f,stderr=f)
  assert r.returncode==0,(side,r.returncode)
  rows=[]
  for line in path.open():
   if line.startswith('TIMING '):
    fields=line.strip().split();rows.append(dict(name=fields[1],**{k:int(v) for k,v in [field.split('=') for field in fields[2:]]}))
  results.append(dict(repeat=repeat,side=side,rows=rows))
  print(repeat,side,len(rows),round(sum(r['json_ns']+r['sse_ns'] for r in rows)/1e6,2),flush=True)
(p/'bench-results.json').write_text(json.dumps(results,indent=2))
