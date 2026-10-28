import differential as d
import asyncio, ast, json, os, re, subprocess, time, uuid
from pathlib import Path
HERE=Path(__file__).parent
cases=d.generate()[:5]
cases += [dict(label=label,raw=raw,tools=tools,extra=extra) for label,raw,tools,extra in [
    ('empty','',[],{}), ('reasoning-only','Plan.',[],{}), ('empty-answer','Plan.</think> \n',[],{}),
    ('answer-only','</think> answer ',[],{}),('space-reasoning','<think>\n  plan  \n</think> answer ',[],{}),
    ('empty-tools','</think>',d.TOOLS,{}),('unused-tools','Plan.</think> answer ',d.TOOLS,{}),
    ('malformed-call','Plan.</think> <tool_call>broken ',d.TOOLS,{}),
    ('integer','</think><tool_call><function=run><parameter=i>1</parameter></function></tool_call>',d.TOOLS,{}),
    ('float','</think><tool_call><function=run><parameter=n>1e2',d.TOOLS,{}),
    ('unicode-integer','</think><tool_call><function=run><parameter=i>١',d.TOOLS,{}),
    ('overparsed','</think><tool_call><function=run><parameter=a>[1,]\n ',d.TOOLS,{}),
    ('multiple','Plan.</think><tool_call><function=run></tool_call><tool_call><function=run></tool_call>',d.TOOLS,{}),
    ('limit-calls','Plan.</think><tool_call><function=run></tool_call><tool_call><function=run></tool_call>',d.TOOLS,{'parallel_tool_calls':False}),
    ('length','thinking continues at length',[],{'max_tokens':1}),
    ('lfm-prose','</think><|tool_call_start|>[run()]<|tool_call_end|>',d.TOOLS,{}),
    ('lfm-parameter','</think><tool_call><function=run><parameter=s><|tool_call_start|>[run()]<|tool_call_end|></parameter></function></tool_call>',d.TOOLS,{}),
    ('usage','Plan.</think>answer',[],{'stream_options':{'include_usage':True}}),
]]
path=HERE/'routes-input.jsonl'
path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
p=subprocess.run([str(HERE/'routes')],env=dict(os.environ,REVIEW_INPUT=str(path),TF_LLAMA_MODEL='/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb',CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,cwd=str(HERE/'source'))
(HERE/'routes.log').write_text(p.stderr)
if p.returncode: raise RuntimeError(p.stderr[-6000:])
got=[json.loads(line.split('RESULT ',1)[1]) for line in p.stderr.splitlines() if 'RESULT ' in line]
assert len(got)==len(cases)
class FakeTorch:
    long=None
    @staticmethod
    def tensor(x,**kw):return x
class FakeHF:
    count=1
    def apply_chat_template(self,*a,**kw):return ''
    def __call__(self,*a,**kw):return {'input_ids':[1]*self.count}
class Stream:
    def __init__(self,events,**kw): self.events=events
class Job:
    def __init__(self,*a,**kw):pass
    def __aiter__(self):return self.run()
    async def run(self):
        for chunk in CHUNKS:yield {'text':chunk}
        yield {'eos':True,'new_tokens':N,'eos_reason':EOS}
    async def cancel(self):pass
class Request:
    def __init__(self,body):self.body=body
    async def json(self):return self.body
    async def is_disconnected(self):return False
hf=FakeHF()
module=ast.parse(d.POD.read_text())
node=next(n for n in module.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='chat')
node.decorator_list=[]
scope=dict(d.scope,asyncio=asyncio,time=time,Request=Request,S={'names':['nanbeige'],'hf':hf,'gen':None},torch=FakeTorch,AsyncJob=Job,ComboSampler=lambda **kw:None,CONTEXT=262144,STOP_IDS=[166101,166102],StreamingResponse=Stream)
exec(compile(ast.Module([n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in {'text_of','prepare'}]+[node],type_ignores=[]),str(d.POD),'exec'),scope)
def delta_aggregate(events):
    out={'reasoning':'','reasoning_content':'','content':'','calls':[],'finish':None}
    for evt in events:
        for choice in evt.get('choices',[]):
            delta=choice.get('delta',{})
            for key in ['reasoning','reasoning_content','content']:out[key]+=delta.get(key) or ''
            for call in delta.get('tool_calls',[]):
                fn=call['function'];out['calls'].append({'name':fn['name'],'arguments':json.loads(fn['arguments'])})
            if choice.get('finish_reason') is not None:out['finish']=choice['finish_reason']
    return out
async def run():
    global CHUNKS,N,EOS
    results=[]; aggregate_mismatches=0
    for case,g in zip(cases,got):
        hf.count=g['whole']['usage']['prompt_tokens']
        N=g['whole']['usage']['completion_tokens']
        raw=case['raw'] if case['label']!='length' else 'thinking'
        CHUNKS=[raw[i:i+3] for i in range(0,len(raw),3)]
        EOS='max_new_tokens' if case['label']=='length' else 'stop'
        body=dict(messages=[{'role':'user','content':'x'}],tools=case['tools'],max_tokens=512,**case.get('extra',{})) if case['label']!='length' else dict(messages=[{'role':'user','content':'x'}],tools=[],max_tokens=1)
        whole=await scope['chat'](Request(body))
        body['stream']=True
        streamed=await scope['chat'](Request(body))
        events=[]
        async for data in streamed.events:
            if data.startswith('data: ') and data.strip()!='data: [DONE]':events.append(json.loads(data[6:]))
        pa=delta_aggregate(events);ga=delta_aggregate(g['events'])
        equal=pa==ga
        aggregate_mismatches+=not equal
        results.append(dict(case=case,pod=dict(whole=whole,events=events),engine=g,aggregate_equal=equal,pod_aggregate=pa,engine_aggregate=ga))
        print(case['label'], 'aggregate_equal='+str(equal),'pod_message='+json.dumps(whole['choices'][0]['message'],ensure_ascii=False),'engine_message='+json.dumps(g['whole']['choices'][0]['message'],ensure_ascii=False))
    (HERE/'full-response.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print('ROUTE_CASES',len(results),'AGGREGATE_MISMATCHES',aggregate_mismatches)
asyncio.run(run())
