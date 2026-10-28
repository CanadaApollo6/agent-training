import json,collections,math
from pathlib import Path
p=Path(__file__).parent
rows=json.loads((p/'semantic-mismatches.json').read_text())
groups=collections.Counter(); samples={}
def add(kind,row,ev,gv):
    groups[kind]+=1
    s=dict(raw=row['case']['raw'],pod=ev,engine=gv)
    if kind not in samples or len(s['raw'])<len(samples[kind]['raw']): samples[kind]=s
for row in rows:
    e=row['pod'];g=row['engine']
    if e['reasoning']!=g['reasoning']: add('reasoning',row,e['reasoning'],g['reasoning'])
    if e['content']!=g['content']: add('content',row,e['content'],g['content'])
    if len(e['calls'])!=len(g['calls']): add('call count',row,e['calls'],g['calls']);continue
    for ec,gc in zip(e['calls'],g['calls']):
        ef=ec['function'];gf=gc['function']
        if ef['name']!=gf['name']: add('function name',row,ef['name'],gf['name'])
        ea=json.loads(ef['arguments']);ga=json.loads(gf['arguments'])
        if ea.keys()!=ga.keys(): add('argument keys',row,ea,ga);continue
        for k in ea:
            ev=ea[k];gv=ga[k]
            if ev==gv:continue
            if isinstance(ev,(float,int)) and isinstance(gv,str) and any(ord(c)>128 for c in gv): kind='unicode number'
            elif isinstance(ev,float) and not math.isfinite(ev):kind='nonfinite'
            elif isinstance(ev,str) and isinstance(gv,str):kind='string changed'
            elif isinstance(ev,str):kind='overparsed '+type(gv).__name__
            elif isinstance(gv,str):kind='underparsed '+type(ev).__name__
            else:kind='other '+type(ev).__name__+' '+type(gv).__name__
            add(kind,row,ev,gv)
print(json.dumps(dict(counts=groups,samples=samples),indent=2,ensure_ascii=False))
