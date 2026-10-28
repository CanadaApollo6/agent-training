import asyncio
import collections
import json
import sys
from pathlib import Path

ENGINE=Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama')
sys.path.insert(0,str(ENGINE/'tools/zig'))
from llama_serve_failures import AppOracle
from llama_serve_envelopes import normalize


def wire(text):
    head,_,raw=text.partition('\r\n\r\n')
    headers={k.lower():v.strip() for line in head.split('\r\n')[1:] if ': ' in line for k,v in [line.split(':',1)]}
    complete=True
    if headers.get('transfer-encoding')=='chunked':
        data=raw.encode('utf8',errors='surrogatepass');at=0;parts=[];complete=False
        while at<len(data):
            end=data.index(b'\r\n',at);n=int(data[at:end],16);at=end+2
            if n==0:
                assert data[at:]==b'\r\n';complete=True;break
            parts.append(data[at:at+n]);assert data[at+n:at+n+2]==b'\r\n';at+=n+2
        raw=b''.join(parts).decode('utf8',errors='surrogatepass')
    return dict(status=int(head.split(' ')[1]),headers=headers,body=normalize(raw),complete=complete)


async def compare_async(m):
    expected={c['name']:c for c in map(json.loads,(m.ROOT/'expected.jsonl').open())}
    exceptional={k:v for k,v in expected.items() if any(key+'_error' in v for key in ['json','sse'])}
    labelled={}
    if exceptional:
        from transformers import AutoTokenizer
        hf=AutoTokenizer.from_pretrained(m.MODEL,local_files_only=True)
        oracle=AppOracle(m.POD,hf)
        for line in (m.ROOT/'input.jsonl').open():
            c=json.loads(line)
            if c['name'] not in exceptional:continue
            oracle.chunks(c['ids'])
            labelled[c['name']]={key:await oracle.request(dict(c['request'],stream=stream)) for key,stream in [('json',False),('sse',True)]}
        (m.ROOT/'failures-asgi.json').write_text(json.dumps(labelled,ensure_ascii=True,indent=1))
    counts=collections.Counter();examples=[]
    with (m.ROOT/'routes.log').open(errors='surrogatepass') as f, (m.ROOT/'mismatches.jsonl').open('w') as out:
        for line in f:
            if 'RESULT ' not in line:continue
            got=json.loads(line.split('RESULT ',1)[1]);want=expected.pop(got['name']);counts['cases']+=1
            failures=[]
            for key in ['json','sse']:
                actual=wire(got[key])
                if got['name'] in labelled:
                    ref=labelled[got['name']][key]
                    matched=(actual['status']==ref['status'] and actual['headers'].get('content-type')==ref['headers']['content-type'] and actual['body']==ref['body'] and actual['complete']==ref['complete'])
                    counts[key+'_reference_failures_compared']+=1
                    if matched:counts[key+'_reference_failures_matched']+=1
                else:
                    matched=actual['body']==want[key]
                try:actual['body'].encode('utf8',errors='surrogatepass').decode('utf8')
                except UnicodeDecodeError:matched=False;counts[key+'_invalid_utf8']+=1
                if not matched:counts[key+'_mismatches']+=1;failures.append(key)
            if want['prompt'] and got['prompt']!=want['prompt']:counts['prompt_mismatches']+=1;failures.append('prompt')
            if failures:
                out.write(json.dumps(dict(name=got['name'],failures=failures),ensure_ascii=True)+'\n')
                if len(examples)<30:examples.append(got['name'])
    counts['missing_engine_results']=len(expected)
    summary=dict(counts=dict(counts),first_mismatches=examples)
    (m.ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    return summary


def compare(m):return asyncio.run(compare_async(m))
