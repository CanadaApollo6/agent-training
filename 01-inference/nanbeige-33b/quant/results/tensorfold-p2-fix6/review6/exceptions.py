import ast
import asyncio
import json
from pathlib import Path
import sys
import independent as m
from extended import case
from fastapi import FastAPI, Request
from transformers import AutoTokenizer
p=Path(__file__).parent
hf=AutoTokenizer.from_pretrained(m.MODEL,local_files_only=True)
env=m.scope(hf)
env['chat'].__annotations__['request']=Request
app=FastAPI()
app.add_api_route('/v1/chat/completions',env['chat'],methods=['POST'])
async def request(body):
    data=json.dumps(body).encode()
    events=[]
    used=False
    async def receive():
        nonlocal used
        if not used:
            used=True
            return {'type':'http.request','body':data,'more_body':False}
        await asyncio.Event().wait()
    async def send(event):events.append(event)
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1',
        'scheme':'http','method':'POST','path':'/v1/chat/completions','raw_path':b'/v1/chat/completions',
        'root_path':'','query_string':b'', 'server':('test',80),'client':('local',1),
        'headers':[(b'content-type',b'application/json')]}
    error=None
    try: await app(scope,receive,send)
    except Exception as e:error=type(e).__name__+': '+str(e)
    status=next(e['status'] for e in events if e['type']=='http.response.start')
    body=b''.join(e.get('body',b'') for e in events if e['type']=='http.response.body')
    return {'status':status,'body':m.normalized(body),'exception':error,
        'complete':any(e['type']=='http.response.body' and not e.get('more_body',False) for e in events)}
async def main():
    out=[]
    for value in ["b'a'", "'\\ud800'", "{'x':b'a'}", "{'x':'\\ud800'}"]:
        obj=case('exception','exception',[('payload',value)])
        ids=hf(obj['raw'],add_special_tokens=False)['input_ids']
        for stream in [False,True]:
            m.STATE.update(chunks=[obj['raw']],n=len(ids),prompt=[])
            r=await request(dict(obj['request'],stream=stream))
            out.append(dict(literal=value,stream=stream,**r))
    (p/'exceptions-asgi.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
asyncio.run(main())
