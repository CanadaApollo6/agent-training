import ast, collections, json, os, random, re, subprocess, uuid
from pathlib import Path
HERE = Path(__file__).parent
POD = Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py')
module = ast.parse(POD.read_text())
names = {'typed', 'split_output'}
constants = {'THINK_END', 'TOOL_CALL', 'FUNCTION', 'PARAMETER'}
nodes = [n for n in module.body if (isinstance(n, ast.FunctionDef) and n.name in names) or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
scope = dict(ast=ast, json=json, uuid=uuid, re=re)
exec(compile(ast.Module(nodes, type_ignores=[]), str(POD), 'exec'), scope)
SCHEMAS = dict(s={'type': 'string'}, i={'type': 'integer'}, n={'type': 'number'}, b={'type': 'boolean'}, o={'type': 'object'}, a={'type': 'array'}, e={'type': 'string', 'enum': ['auto', 'never']}, u={'type': ['null', 'integer']})
TOOLS = [{'type':'function', 'function':{'name':'run', 'parameters':{'type':'object', 'properties':SCHEMAS}}}]
VALUES = {
's':['echo hi', '  echo hi  ', '中文💡', '\nhi\n\n', '', 'null', '<parameter=s>inner', '<parameter=>bad', '</function>', '</tool_call>', '<parameter=s\nx>bad', 'x\x1c'],
'i':['0','-1','+7','0007','1_000',' 42 ','１２','١٢','123456789012345678901234567890', '3.0','1e2','null','NULL','nan','bad'],
'n':['0','1','1.0','1e2','1e-8','-0','-0.0','2.5','1_000',' 3.5 ','１２','١٢','nan','inf','-Infinity','1e309','null','bad'],
'b':['true','TRUE',' True ','false','False','1','0','null','NULL','bad','\ntrue\n'],
'o':['{}','{"a": 1, "b": [false, null]}',"{'a': True, 'b': None}","{'a': (1, 2)}",'[]','true','null','bad',"{1: 'x'}", "{'x': '\\x41'}", "{'x': 0x10}","{'x': -1.5}","{'x': r'\\n'}"],
'a':['[]','[1, 2.5, false, null]',"[True, None, {'x': 2}]",'(1, 2)','[1,]','null','bad','[0x10, 0b11]','[1e2, -0.0]'],
'e':['auto','never','invalid','  auto  ','null','中文'],
'u':['null','3','-2','3.5','NULL','１２']}

def expected(case):
    r,c,calls = scope['split_output'](case['raw'], case['tools'])
    return dict(reasoning=r, content=c, calls=calls)

def engine(cases):
    path=HERE/'input.jsonl'
    path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    p=subprocess.run([str(HERE/'parser')],env=dict(os.environ,REVIEW_INPUT=str(path),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    (HERE/'parser-last.log').write_text(p.stderr)
    if p.returncode: raise RuntimeError(p.stderr[-5000:])
    out=[json.loads(line.split('RESULT ',1)[1]) for line in p.stderr.splitlines() if 'RESULT ' in line]
    assert len(out)==len(cases), (len(out), len(cases))
    return out

def normalized(v, exact=False):
    return dict(reasoning=v['reasoning'], content=v['content'], calls=[dict(name=c['function']['name'], arguments=c['function']['arguments'] if exact else json.loads(c['function']['arguments'])) for c in v['calls']])

def generate():
    rng=random.Random(0x8fd873cb)
    cases=[]
    for label,raw in [
        ('header-close','</think><tool_call><function=</function></tool_call>'),
        ('param-newline','</think><tool_call><function=run><parameter=s>hi\n\n</tool_call>'),
        ('function-newline','</think><tool_call><function=run><parameter=s>hi\n\n</function></tool_call>'),
        ('tool-newline','</think><tool_call><function=run><parameter=s>hi\n\n\n'),
        ('trailing-spaces','</think><tool_call><function=run><parameter=s>hi  ')]:
        cases.append(dict(label=label, raw=raw, tools=TOOLS))
    for idx in range(24000):
        keys=rng.sample(list(SCHEMAS), rng.randint(1,5))
        fields=[]
        for key in keys:
            header=rng.choice([key]*12+[' '+key+' ', '\u2003'+key+'\u2003', key.upper(), '',key+'\nx','<parameter='+key,'</parameter'])
            value=rng.choice(VALUES[key])
            fields.append('<parameter='+header+'>'+rng.choice(['','\n','\n\n'])+value+rng.choice(['','\n','\n\n'])+rng.choice(['</parameter>']*8+['']))
        fname=rng.choice(['run']*20+[' run ','\u2003run\u2003','RUN','other','','</function','<function=run','ru\nn'])
        call='<tool_call>'+rng.choice(['','\n','<function=>','<function=bad\n>'])+'<function='+fname+'>'+''.join(fields)+rng.choice(['</function>']*5+[''])+rng.choice(['</tool_call>']*5+[''])
        if idx%7==0: call += '\n'+call
        if idx%11==0: call='<tool_call>broken</tool_call>'+call
        if idx%4==0: call=call[:rng.randrange(len(call)+1)]
        raw=rng.choice(['Plan.','<think>\nPlan.\n','\n  plan  \n','', '计划💡。'])+rng.choice(['</think>']*19+[''])+rng.choice(['','\n  Checking.  \n',' \t','\u2003'])+call+rng.choice(['','\n','\n\n','  ','\n  ','\nafter'])
        cases.append(dict(label='grammar-'+str(idx), raw=raw, tools=[] if idx%19==0 else TOOLS))
    return cases

def main():
    cases=generate()
    (HERE/'corpus.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    got=engine(cases)
    semantic=[]; serialization=[]; schemas=collections.Counter(); calls=0
    for idx,(case,g) in enumerate(zip(cases,got)):
        e=expected(case); calls += bool(e['calls'])
        if normalized(e)!=normalized(g):
            semantic.append(dict(index=idx,case=case,pod=e,engine=g))
            for c in e['calls']:
                for key in json.loads(c['function']['arguments']): schemas[key]+=1
        elif normalized(e,True)!=normalized(g,True): serialization.append(dict(index=idx,case=case,pod=e,engine=g))
    (HERE/'semantic-mismatches.json').write_text(json.dumps(semantic,ensure_ascii=False,indent=2))
    (HERE/'serialization-samples.json').write_text(json.dumps(serialization[:30],ensure_ascii=False,indent=2))
    result=dict(cases=len(cases), with_calls=calls,semantic_mismatches=len(semantic), serialization_only_mismatches=len(serialization), mismatched_keys=dict(schemas), directed=[dict(label=c['label'], equal=normalized(expected(c))==normalized(g)) for c,g in zip(cases[:5],got[:5])])
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    groups={}
    for x in semantic:
        ec=normalized(x['pod']); gc=normalized(x['engine'])
        sig=json.dumps([ec['calls'],gc['calls']],ensure_ascii=False)
        if sig not in groups or len(x['case']['raw']) < len(groups[sig]['case']['raw']): groups[sig]=x
    for x in sorted(groups.values(),key=lambda x:len(x['case']['raw']))[:12]:
        print(json.dumps(dict(raw=x['case']['raw'],pod=normalized(x['pod']),engine=normalized(x['engine'])), ensure_ascii=False))
if __name__=='__main__': main()
