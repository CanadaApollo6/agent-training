import json,statistics,sys
from pathlib import Path
p=Path('/tmp/nanbeige-review-9a5732e0');sys.path.insert(0,str(p/'source/tools/zig'))
from llama_serve_failure_transport import decode
rows=json.loads((p/'bench-results.json').read_text());expected={x['name']:x for x in map(json.loads,(p/'expected.jsonl').open())}
summary={}; mismatches=[]; outputs=0
for r in rows:
 path=p/f"bench-{r['repeat']}-{r['side']}.log"
 for line in path.open():
  if not line.startswith('RESULT '):continue
  v=json.loads(line[7:]);outputs+=2
  for kind in ['json','sse']:
   got=decode(v[kind].encode());want=expected[v['name']][kind]
   if any(got[k]!=want[k] for k in ['status','body','complete']):mismatches.append([r['repeat'],r['side'],v['name'],kind])
for side in ['base','head']:
 rr=[r for r in rows if r['side']==side];totals=[sum(x['json_ns']+x['sse_ns'] for x in r['rows'])/1e6 for r in rr]
 summary[side]=dict(repeats=len(rr),cases=len(rr[0]['rows']),median_total_ms=statistics.median(totals),range_ms=[min(totals),max(totals)],totals_ms=totals)
a=next(r['rows'] for r in rows if r['side']=='base');b=next(r['rows'] for r in rows if r['side']=='head')
for field in ['json_capacity','total_capacity']:
 differences=[y[field]-x[field] for x,y in zip(a,b)]
 summary[field]=dict(median_extra_bytes=statistics.median(differences),max_extra_bytes=max(differences),median_relative=statistics.median(y[field]/x[field] for x,y in zip(a,b)),max_relative=max(y[field]/x[field] for x,y in zip(a,b)))
summary.update(output_comparisons=outputs,mismatches=mismatches)
(p/'bench-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
