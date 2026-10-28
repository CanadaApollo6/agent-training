import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
p=Path(__file__).parent
root=p/'extended'
handles=[(root/f'input-{i}.jsonl').open('w') for i in range(8)]
for i,line in enumerate((root/'input.jsonl').open()):handles[i%8].write(line)
for f in handles:f.close()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_LLAMA_MODEL='/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb')
def run(i):
    start=time.monotonic()
    with (root/f'routes-{i}.log').open('w') as log:
        done=subprocess.run([str(p/'routes')],env=dict(env,REVIEW_INPUT=str(root/f'input-{i}.jsonl')),stdout=log,stderr=log)
    row={'shard':i,'exit':done.returncode,'elapsed':time.monotonic()-start}
    print(json.dumps(row),flush=True)
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(run,range(8)))
(root/'shards.json').write_text(json.dumps(results,indent=2))
assert all(r['exit']==0 for r in results)
with (root/'routes.log').open('wb') as out:
    for i in range(8):
        for line in (root/f'routes-{i}.log').open('rb'):out.write(line)
import independent as m
m.ROOT=root
m.compare()
