import asyncio,json,os,queue,re,socket,subprocess,sys,threading,time
from pathlib import Path
P=Path('/tmp/nanbeige-review-9a5732e0'); sys.path.insert(0,str(P/'source/tools/zig'))
from llama_serve_failures import AppOracle
from llama_serve_failure_transport import decode
from transformers import AutoTokenizer
import uvicorn
MODEL='/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb'
POD=Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py')

async def wire(port,body,force_close=True,disconnect=False):
    reader,writer=await asyncio.open_connection('127.0.0.1',port)
    b=json.dumps(body).encode()
    writer.write(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'+(b'Connection: close\r\n' if force_close else b'')+b'Content-Length: '+str(len(b)).encode()+b'\r\n\r\n'+b)
    await writer.drain()
    try:
        if disconnect:
            received=b''; start=time.monotonic()
            while b'"reasoning"' not in received:
                received+=await asyncio.wait_for(reader.read(4096),10)
            elapsed=time.monotonic()-start
            writer.transport.abort()
            return dict(prefix=received.decode(),first_reasoning_s=elapsed)
        data=await asyncio.wait_for(reader.read(),20)
        return decode(data)
    finally:
        writer.close()
        try: await writer.wait_closed()
        except ConnectionError: pass

async def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_LLAMA_MODEL=MODEL)
    proc=subprocess.Popen([str(P/'tcp')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,bufsize=1)
    q=queue.Queue(); log=(P/'tcp-engine.log').open('w'); mem=[]
    def reader():
        for s in proc.stdout:
            log.write(s); log.flush()
            if s.startswith('MEM '): mem.append(int(s[4:]))
            if 'PORT ' in s: q.put('PORT '+s.split('PORT ',1)[1].strip())
            elif s.startswith(('READY ','CLEAN ')): q.put(s.strip())
    threading.Thread(target=reader,daemon=True).start()
    def expect(prefix):
        s=q.get(timeout=40); assert s.startswith(prefix),s; return s
    engine_port=int(expect('PORT ')[5:])
    def control(raw,slow=False):
        proc.stdin.write(json.dumps(dict(raw=raw,slow=slow))+'\n'); proc.stdin.flush(); return expect('READY ')
    hf=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    oracle=AppOracle(POD,hf)
    original_job=oracle.env['AsyncJob']; cancelled=0
    class SlowJob(original_job):
        async def run(self):
            for t in oracle.state['chunks']:
                await asyncio.sleep(.003)
                yield dict(text=t)
            yield dict(eos=True,new_tokens=oracle.state['count'],eos_reason='stop')
        async def cancel(self):
            nonlocal cancelled
            cancelled+=1
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(); sock.setblocking(False)
    pod_port=sock.getsockname()[1]
    uv=uvicorn.Server(uvicorn.Config(oracle.app,log_level='critical',lifespan='off',http='h11'))
    server=asyncio.create_task(uv.serve(sockets=[sock]))
    while not uv.started: await asyncio.sleep(.01)
    bad=[]; count=0; recoveries=0; disconnects=[]; results=[]
    def compare(name,mode,target,got,want):
        nonlocal count
        count+=1
        keys=[k for k in ['status','body','complete'] if got[k]!=want[k]]
        if got['headers'].get('content-type')!=want['headers'].get('content-type'): keys.append('content-type')
        if keys: bad.append(dict(name=name,mode=mode,target=target,fields=keys,got=got,want=want))
    try:
        rows=json.load((P/'failures.json').open())
        good=dict(messages=[dict(role='user',content='x')],max_tokens=65000,stream_options=dict(include_usage=True))
        for i,row in enumerate(rows):
            for mode,stream in [('json',False),('sse',True)]:
                control(row['raw']); oracle.chunks(hf(row['raw'],add_special_tokens=False)['input_ids'])
                body=dict(row['request'],stream=stream)
                expected=row[mode]
                close=expected['complete'] and expected['status']!=500
                for target,port in [('engine',engine_port),('pod',pod_port)]:
                    got=await wire(port,body,force_close=close)
                    compare(row['name'],mode,target,got,expected)
                control('</think>Recovered.'); oracle.chunks(hf('</think>Recovered.',add_special_tokens=False)['input_ids'])
                body=dict(good,stream=stream); wanted=await oracle.request(body)
                for target,port in [('engine',engine_port),('pod',pod_port)]:
                    got=await wire(port,body)
                    compare(row['name']+'-recovery',mode,target,got,wanted); recoveries+=1
            if i%25==24: print('TCP',i+1,'bad',len(bad),flush=True)
        for i in range(12):
            raw=('Inspect the source and compare the result. 雪\t\n'*80)+'</think>Done.'
            ready=control(raw,True)
            if i==0: initial_cancels=int(ready.split('cancels=')[1])
            oracle.chunks(hf(raw,add_special_tokens=False)['input_ids']); oracle.env['AsyncJob']=SlowJob
            for target,port in [('engine',engine_port),('pod',pod_port)]:
                d=await wire(port,dict(good,stream=True),disconnect=True); disconnects.append(dict(target=target,**d))
            await asyncio.sleep(.03)
            control('</think>Recovered.'); oracle.env['AsyncJob']=original_job; oracle.chunks(hf('</think>Recovered.',add_special_tokens=False)['input_ids'])
            for stream in [False,True]:
                body=dict(good,stream=stream); wanted=await oracle.request(body)
                for target,port in [('engine',engine_port),('pod',pod_port)]:
                    got=await wire(port,body); compare('disconnect-recovery',str(stream),target,got,wanted); recoveries+=1
        proc.stdin.write('STOP\n');proc.stdin.flush(); clean=expect('CLEAN ')
        code=proc.wait(timeout=40); assert code==0,code
        assert cancelled==12,cancelled
        assert int(clean.split('cancels=')[1])-initial_cancels==12,clean
        print(clean,flush=True)
        summary=dict(comparisons=count,recovery_responses=recoveries,disconnects=len(disconnects),pod_cancels=cancelled,clean=clean,mismatches=len(bad),allocated_bytes=mem,engine_exit=code)
        (P/'tcp-summary.json').write_text(json.dumps(summary,indent=2));(P/'tcp-mismatches.json').write_text(json.dumps(bad,indent=2));(P/'disconnects.json').write_text(json.dumps(disconnects,indent=2))
        print({k:v for k,v in summary.items() if k!='allocated_bytes'},flush=True)
    finally:
        uv.should_exit=True;await server;sock.close()
        if proc.poll() is None:
            proc.stdin.close()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=5)
        log.close()
if __name__=='__main__': asyncio.run(main())
