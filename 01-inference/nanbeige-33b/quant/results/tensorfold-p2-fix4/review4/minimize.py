import differential as d, json
from pathlib import Path
p=Path(__file__).parent
cases=[]
for key,values in {'i':['١','１'], 'n':['١','1e2','nan','inf','1e309','1.0','-0'], 'o':["{'a': 1}\n ","{1:'x'}\n ","{'x':r'\n'}",'None','TRUE','False','NULL'], 'a':['(1,)\n ','[0x1]\n ','[1,]\n ','[0x1]','[1,]']}.items():
    for v in values: cases.append(dict(raw='</think><tool_call><function=run><parameter='+key+'>'+v,tools=d.TOOLS,label=key+'='+repr(v)))
out=d.engine(cases)
result=[]
for c,g in zip(cases,out):
    e=d.expected(c)
    result.append(dict(case=c, pod=d.normalized(e,True), engine=d.normalized(g,True), semantic_equal=d.normalized(e)==d.normalized(g)))
(p/'minimal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for r in result: print(r['case']['label'], 'semantic_equal='+str(r['semantic_equal']), 'pod='+r['pod']['calls'][0]['arguments'], 'engine='+r['engine']['calls'][0]['arguments'])
