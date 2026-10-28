import asyncio
import copy
import json
from pathlib import Path
import independent as m

m.ROOT = Path('/tmp/nanbeige-review-7fe2dce7/targeted')
m.ROOT.mkdir(exist_ok=True)
TRACE = Path('/home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals/outputs/primeintellect/terminal-bench-2--nanbeige-exl3-a1--pi--1a1fed7e/traces.jsonl')
rows = [json.loads(l)['traces'][0] for l in TRACE.open()]
actual_tools = [{'type':'function', 'function':{k:v for k,v in t.items() if k != 'type'}} for t in rows[0]['tools']]

def cases():
    for name, val in [
        ('named-minimal', "'\\N{SNOWMAN}'"),
        ('escape-minimal', "'\\x+1'"),
        ('underscore-minimal', "'\\u0_41'"),
        ('nul-minimal', "'\x00'"),
        ('edit-named', "[{'oldText':'x','newText':'\\N{SNOWMAN}'}]"),
        ('edit-invalid', "[{'oldText':'x','newText':'\\u+041'}]"),
        ('edit-valid-json', '[{"oldText":"x","newText":"\\\\N{SNOWMAN}"}]'),
        ('edit-valid-python', "[{'oldText':'x','newText':'\\\\N{SNOWMAN}'}]"),
        ('edit-raw-python', "[{'oldText':'x','newText':r'\\N{SNOWMAN}'}]"),
    ]:
        yield {'name':name, 'group':'minimal', 'raw':'</think>'+m.xml('edit',[('edits',val)]),
               'request':{'messages':[{'role':'user','content':'x'}], 'tools':actual_tools, 'max_tokens':500}}
    for i, trace in enumerate(rows):
        ts = [{'type':'function','function':{k:v for k,v in t.items() if k!='type'}} for t in trace['tools']]
        for j, call in enumerate(trace['calls']):
            if j not in {0, len(trace['calls'])//2, len(trace['calls'])-1} or 'node' not in call: continue
            index = trace['nodes'][call['node']].get('parent')
            messages=[]
            while index is not None:
                node=trace['nodes'][index]
                message=copy.deepcopy(node['message'])
                if message.get('tool_calls'):
                    message['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}} if 'function' not in c else c for c in message['tool_calls']]
                messages.append(message)
                index=node.get('parent')
            messages.reverse()
            yield {'name':f'recorded-{i:02}-{j:03}', 'group':'recorded', 'raw':'</think>Done.',
                   'request':{'messages':messages,'tools':ts,'max_tokens':50}}

if __name__=='__main__':
    m.new_cases=cases
    asyncio.run(m.generate())
