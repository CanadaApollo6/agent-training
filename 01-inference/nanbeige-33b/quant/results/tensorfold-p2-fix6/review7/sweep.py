import asyncio, copy, json, random, sys, time
from pathlib import Path
P=Path('/tmp/nanbeige-review-9a5732e0')
REPO=P/'source'
sys.path.insert(0,str(REPO/'tools/zig'))
from llama_serve_failures import AppOracle
from transformers import AutoTokenizer
MODEL='/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb'
POD=Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py')
TRACE=Path('/home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals/outputs/primeintellect/terminal-bench-2--nanbeige-exl3-a1--pi--1a1fed7e/traces.jsonl')
R=random.Random(0x9a5732e0)
def histories():
    result=[]
    for i,line in enumerate(TRACE.open()):
        trace=json.loads(line)['traces'][0]
        tools=[dict(type='function',function={k:v for k,v in t.items() if k!='type'}) for t in trace['tools']]
        for j,call in enumerate(trace['calls']):
            if j not in {0,len(trace['calls'])//2,len(trace['calls'])-1} or 'node' not in call: continue
            idx=trace['nodes'][call['node']].get('parent'); messages=[]
            while idx is not None:
                node=trace['nodes'][idx]; msg=copy.deepcopy(node['message'])
                if msg.get('tool_calls'):
                    msg['tool_calls']=[c if 'function' in c else dict(id=c['id'],type='function',function=dict(name=c['name'],arguments=c['arguments'])) for c in msg['tool_calls']]
                messages.append(msg); idx=node.get('parent')
            messages.reverse()
            result.append(dict(messages=messages,tools=tools,max_tokens=65000))
    assert len(result)==55,len(result)
    return result

def xml(name,params):
    return '<tool_call>\n<function='+name+'>\n'+''.join('<parameter='+k+'>'+v+'</parameter>\n' for k,v in params)+'</function>\n</tool_call>'

def continuation(i):
    tag=['雪','café','🧪','Δx','路径','naïve','e\u0301'][i%7]
    data={'task':i,'label':tag,'command':"printf '%s\\n' \"$HOME\"",'pattern':r'\b(foo|bar)\s+','nested':{'json':json.dumps({'q':'"quoted"','path':r'C:\tmp\a'})}}
    source='import json\n\ndef test_roundtrip():\n\tvalue = '+repr(data)+'\n\tassert json.loads(json.dumps(value)) == value\n'
    if i%37==0: source += '\n'.join('item_'+str(n)+' = '+repr(data) for n in range(180))+'\n'
    if i%11==0: source=source.replace('\n','\r\n')
    commands=["python - <<'PY'\n"+source+'PY\n', "cat > 'tests/test_"+tag+".py' <<'PY'\n"+source+"PY\npytest -q\n", "printf '\033[31mFAIL\033[0m\\r\\n'; rg -n 'foo\\s+bar' src/\n", "python -c "+repr('import json; print(json.loads('+repr(json.dumps(data,ensure_ascii=False))+'))'), "git diff --check && python -m pytest -q tests/test_api.py", "python - <<'PY'\r\nprint('"+tag+"')\r\nPY\r\n"]
    cmd=commands[i%len(commands)]
    edits=[{'oldText':'return None\n','newText':source},{'oldText':'VALUE = {}','newText':'VALUE = '+repr(data)}]
    variants=[json.dumps(edits,ensure_ascii=False),repr(edits),json.dumps(edits,ensure_ascii=True), "[{'oldText': 'x', 'newText': r'\\N{SNOWMAN}\\n'}]", "[{'oldText':'x','newText':'\\N{SNOWMAN}'}]", "[{'oldText':'x','newText':'\\x+1'}]", "[{'oldText':'x','newText':'\x00'}]"]
    calls=[xml('bash',[('command',cmd),('timeout',R.choice(['30','1e2','0.5','120']))]),xml('write',[('path','/app/tests/test_'+tag+'.py'),('content',source)]),xml('edit',[('path','/app/src/main.py'),('edits',variants[i%len(variants)])]),xml('read',[('path','/app/src/main.py'),('offset',str(i%50)),('limit','100')])]
    k=i%10
    answer=calls[k%4]
    if k in (4,5,6): answer='\n'.join(calls[:2+k%3])
    if k==7: answer='Tests pass. The fixture now covers '+tag+'.\n'
    if k==8: answer=calls[2].replace('</parameter>','',1)
    if k==9: answer=calls[0].removesuffix('</tool_call>')
    thought=['Inspect the failing test.\n','<think>\n检查 '+tag+' and the fixture.\n','Run the narrow test before editing.\t',''][i%4]
    return thought+'</think>'+['','\n','I will apply the change.\r\n'][i%3]+answer

class CachedHF:
    def __init__(self,hf): self.hf=hf; self.cache={}; self.tokens={}; self.last=[]
    def apply_chat_template(self,messages,**kw):
        key=json.dumps([messages,kw],ensure_ascii=True)
        if key not in self.cache:
            s=self.hf.apply_chat_template(messages,**kw)
            self.cache[key]=s; self.tokens[s]=self.hf(s,add_special_tokens=False)['input_ids']
        s=self.cache[key]; self.last=self.tokens[s]; return s
    def __call__(self,text,**kw): return dict(input_ids=self.tokens[text])

async def main():
    hs=histories(); (P/'histories.json').write_text(json.dumps(hs,ensure_ascii=True))
    hf=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    oracle=AppOracle(POD,hf); cached=CachedHF(hf); oracle.env['S']['hf']=cached
    class Job:
        def __init__(self,*a,**kw): self.kw=kw
        def __aiter__(self): return self.run()
        async def run(self):
            for t in oracle.state['chunks']: yield dict(text=t)
            yield dict(eos=True,new_tokens=oracle.state['count'],eos_reason=oracle.state.get('reason','stop'))
        async def cancel(self): pass
    oracle.env['AsyncJob']=Job
    counts={}; longest=0; start=time.monotonic()
    n=int(sys.argv[1]) if len(sys.argv)>1 else 10010
    with (P/'input.jsonl').open('w') as f,(P/'expected.jsonl').open('w') as e:
        for i in range(n):
            h=i%55; raw=continuation(i); ids=hf(raw,add_special_tokens=False)['input_ids']; longest=max(longest,len(ids))
            overrides=dict(stream_options=dict(include_usage=bool(i%2)))
            if i%31==0:
                ids=ids[:max(1,len(ids)*2//3)]; overrides['max_tokens']=len(ids)
            oracle.chunks(ids); oracle.state['reason']='max_new_tokens' if i%31==0 else 'stop'
            req=dict(hs[h],**overrides)
            row=dict(name=f'pilot-{h:02}-{i:05}',history=h,raw=raw,ids=ids,request=overrides)
            j=await oracle.request(dict(req,stream=False)); s=await oracle.request(dict(req,stream=True))
            exp=dict(name=row['name'],json=j,sse=s,prompt=cached.last if i<55 else [])
            f.write(json.dumps(row,ensure_ascii=True)+'\n'); e.write(json.dumps(exp,ensure_ascii=True)+'\n')
            for kind,result in [('json',j),('sse',s)]:
                key=kind+'_'+str(result['status'])+'_'+str(result['complete']); counts[key]=counts.get(key,0)+1
            if i%250==249: print(i+1,round(time.monotonic()-start,2),counts,flush=True); f.flush(); e.flush()
    (P/'generation-summary.json').write_text(json.dumps(dict(seed='0x9a5732e0',histories=55,cases=n,counts=counts,max_output_tokens=longest,max_prompt_tokens=max(map(len,cached.tokens.values())),seconds=time.monotonic()-start),indent=2))
    print('DONE',n,flush=True)
if __name__=='__main__': asyncio.run(main())
