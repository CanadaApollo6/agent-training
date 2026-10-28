import json
from pathlib import Path
from transformers import AutoTokenizer
import independent as m
from extended import case
p=Path(__file__).parent
hf=AutoTokenizer.from_pretrained(m.MODEL,local_files_only=True)
with (p/'minimal-input.jsonl').open('w') as f:
    for i,value in enumerate(["b'a'", "'\\ud800'"]):
        c=case('minimal-'+str(i),'minimal',[('payload',value)])
        c['ids']=hf(c['raw'],add_special_tokens=False)['input_ids']
        f.write(json.dumps(c)+'\n')
