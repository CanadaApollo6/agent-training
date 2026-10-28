import json,re,sys
from pathlib import Path
P=Path('/tmp/nanbeige-review-9a5732e0'); sys.path.insert(0,str(P/'source/tools/zig'))
from llama_serve_failure_transport import decode

def main():
    expected={r['name']:r for r in map(json.loads,(P/'expected.jsonl').open())}
    bad=[]; count=0; prompts=0; errors=[]
    for line in (P/'routes.log').open():
        if not line.startswith('RESULT '): continue
        got=json.loads(line[7:]); e=expected.pop(got['name']); count+=1
        for k in ('json','sse'):
            d=decode(got[k].encode('utf8')); ref=e[k]
            differences=[key for key in ('status','body','complete') if d[key]!=ref[key]]
            if d['headers'].get('content-type') != ref['headers'].get('content-type'): differences.append('content-type')
            if differences: bad.append(dict(name=e['name'],mode=k,fields=differences,expected=ref,actual=d))
        if e['prompt']:
            prompts+=1
            if got['prompt'] != e['prompt']: errors.append(e['name'])
    (P/'mismatches.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=True)+'\n' for x in bad))
    summary=dict(cases=count,pairs=count*2,mismatches=len(bad),prompt_comparisons=prompts,prompt_mismatches=errors,missing=len(expected))
    (P/'comparison-summary.json').write_text(json.dumps(summary,indent=2)); print(summary)
if __name__=='__main__': main()
