import ast
import asyncio
import collections
import copy
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import types
import uuid
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

ROOT = Path('/tmp/nanbeige-fix6/review5')
POD = Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py')
MODEL = '/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb'
R = random.Random(0x72FE2243)
COMMANDS = [
    "python - <<'PY'\nimport json\nprint(json.dumps({'x': '\\\"', 'emoji': '💡'}))\nPY",
    "find . -name '*.py' -print0 | xargs -0 rg 'foo\\s+bar'",
    "printf '%s\\n' '{\"path\":\"C:\\\\tmp\",\"name\":\"雪\"}' > data.json",
    "bash -lc 'cd /app && pytest -q tests/test_api.py'",
    "sed -i 's/old/new/g' src/main.rs && git diff --check",
    "curl -s -H 'Content-Type: application/json' -d '{\"messages\":[{\"content\":\"hello\"}]}' localhost:8080",
    'python -c "print(\'\\N{SNOWMAN}\')"',
    "cat <<'EOF' > fixture.xml\n<function=run><parameter=command>echo hi</parameter></function>\nEOF",
    "printf 'αβγ\\r\\n'; echo \"$HOME\"; test -f 'a b.txt'",
    "git status --porcelain=v1 && git log -1 --format='%H %s'",
]
SCHEMAS = [
    {'type': 'string'}, {'type': 'object'}, {'type': 'array'}, {'type': 'integer'},
    {'type': 'number'}, {'type': 'boolean'}, {'enum': ['true', 'false', 'null']},
    {'anyOf': [{'type': 'object'}, {'type': 'null'}]},
    {'type': ['null', 'string']}, {'type': ['null', 'object']}, {'type': ['integer', 'string']},
    {'type': 'object', 'nullable': True}, {}, {'type': ['null']}, {'type': None},
]

def tools(schema, mode=0):
    f = {'name': 'terminal', 'description': 'Run a command in the terminal.'}
    if mode != 1:
        f['parameters'] = {'type': 'object', 'properties': {'command': {'type': 'string'}, 'payload': schema,
            'timeout': {'type': 'number'}, 'background': {'type': 'boolean'}}}
    t = {'function': f}
    if mode != 2:
        t['type'] = 'function'
    return [t]


def xml(name, params):
    return '<tool_call><function=' + name + '>' + ''.join('<parameter=' + k + '>' + v + '</parameter>' for k, v in params) + '</function></tool_call>'


def new_cases():
    for i in range(21000):
        command = R.choice(COMMANDS)
        schema = copy.deepcopy(R.choice(SCHEMAS))
        obj = {'cmd': command, 'env': {'LANG': 'C.UTF-8', 'PATH': '/tmp/bin:/usr/bin'},
               'steps': [i % 19, None, bool(i % 2), {'query': R.choice(['你好', 'quotes \' "', '\\t\\n', 'a\tb\nc'])}]}
        values = [json.dumps(obj, ensure_ascii=bool(i % 2)), repr(obj), json.dumps([obj, {'x': [1, 2]}]),
                  "{'cmd': " + repr(command) + ", 'flags': (True, None, -1.25),}",
                  '{"x":1,"x":2,"name":"last"}', "{1:'first', 1.0:'last', '1':'text'}",
                  R.choice(['000', '-0.0', '6.02e23', '1_000', 'NaN', '-Infinity', '１２３', 'null', 'True']),
                  command, ' [1, 2,]\n', "{'a': [1, 2], 'b': 'x' 'y'}", '"' + command.replace('"', '\\"') + '"']
        value = R.choice(values)
        if i % 277 == 0:
            value = json.dumps({'document': command * 180, 'tail': 'END'}, ensure_ascii=False)
        offered = tools(schema, 1 if i % 31 == 0 else 2 if i % 37 == 0 else 0)
        count = R.choices([1, 2, 3, 5], [7, 3, 1, 1])[0]
        calls = []
        for j in range(count):
            name = 'not_offered' if i % 23 == 0 else 'terminal'
            params = [('command', command), ('payload', value), ('timeout', R.choice(['10', '1e2', '0.25', '٣٠'])), ('background', R.choice(['true', 'FALSE', ' null ']))]
            if i % 7 == 0:
                params += [('command', command + ' # retry'), ('payload', value)]
            if j:
                params = [('command', R.choice(COMMANDS)), ('payload', str(i + j))]
            calls.append(xml(name, params))
        raw = R.choice(['</think>', 'Inspect files.\n</think>', '<think>\n计划💡\n</think>']) + R.choice(['', 'I will run this.\n', '\n']) + '\n'.join(calls)
        if i % 11 == 0:
            raw = raw.replace('</parameter>', '', 1)
        if i % 17 == 0:
            raw = raw.removesuffix('</tool_call>')
        if i % 19 == 0:
            raw += '\nDone.\n'
        body = {'model': 'nanbeige', 'messages': [{'role': 'user', 'content': 'Inspect the repository and fix the tests.'}],
                'tools': offered, 'max_tokens': 65000, 'stream_options': {'include_usage': bool(i % 2)}}
        if i % 13 == 0:
            body['parallel_tool_calls'] = False
        yield {'name': f'output-{i:05}', 'group': 'output', 'raw': raw, 'request': body}
    arguments = [json.dumps({'command': s}, ensure_ascii=False) for s in COMMANDS]
    arguments += ['{broken', '', ' ', 'null', '[]', '42', '"hello"', '{"x":1,"x":2}',
                  '{"n":1e2,"x":-0.0}', '{"command":"echo \\u96ea"}', '{"x":NaN}', '{"x":Infinity}',
                  '{"x":1e999}', '{"x":1e-7}', None, {'command': 'echo ok'}, ['a', 'b']]
    for i in range(1050):
        args = copy.deepcopy(arguments[i % len(arguments)])
        msgs = [{'role': 'system', 'content': R.choice(['Use the terminal.', None, ['one', {'type': 'text', 'text': 'two'}]])},
                {'role': 'developer', 'content': 'Check actual results.'},
                {'role': 'user', 'content': R.choice(['Fix the tests.', ['Fix', {'type': 'text', 'text': 'tests'}], None])}]
        for j in range(1 + i % 3):
            f = {'name': 'terminal', 'arguments': args}
            if i % 29 == 0:
                f.pop('arguments')
            assistant = {'role': 'assistant', 'tool_calls': [{'id': 'call_' + str(j), 'type': 'function', 'function': f}],
                         'reasoning': 'Inspect before editing.', 'content': None}
            if i % 3 == 0:
                assistant.pop('content')
            if i % 5 == 0:
                assistant['reasoning_content'] = ''
            if i % 7 == 0:
                assistant['content'] = [{'type': 'text', 'text': 'Executing.'}]
            msgs += [assistant, {'role': 'tool', 'tool_call_id': 'call_' + str(j), 'content': R.choice(['exit 0\n雪', None, ['line 1', {'type': 'text', 'text': 'line 2'}]])}]
        msgs.append({'role': 'user', 'content': 'Continue.'})
        if i % 17 == 0:
            msgs.insert(-1, {'role': 'assistant', 'tool_calls': None})
        if i % 19 == 0:
            msgs.insert(-1, {'role': 'developer', 'content': 'Keep working.'})
        body = {'messages': msgs, 'tools': tools({'type': 'object'}), 'max_tokens': 50}
        if i % 11 == 0:
            body['chat_template_kwargs'] = {'enable_thinking': False}
        yield {'name': f'history-{i:04}', 'group': 'history', 'raw': '</think>Done.', 'request': body}
    literal_values = ["{'x': 'a\x00b'}", "{'x': '\\N{SNOWMAN}'}", "{'x': b'abc'}", "{'x': {1,2}}", "{'x': 1+2j}",
        "{'x': ...}", "{(1,2): 'x'}", "[1, 2\x00]", "{'x': '\\uD800'}", "['a'\n'b']", "{'x': 1_0}",
        '[[1]]', "{'x': - (2)}", "{'x': 0x_1}", "{'x': '\\x_1'}", "{'x': '\\u0_41'}",
        "{'x': '\\x+1'}", "{'x': '\\u+041'}", "{'x': '\\U0000_041'}", '"x\x00y"',
        '[1e10000]', "{'x': 1.e-320}", '[NaN,Infinity,-Infinity]', '9' * 4301]
    for i, val in enumerate(literal_values):
        yield {'name': f'directed-{i:02}', 'group': 'directed', 'raw': '</think>' + xml('terminal', [('payload', val)]),
               'request': {'messages': [{'role': 'user', 'content': 'x'}], 'tools': tools({'type':'object'}), 'max_tokens': 10000}}

class Request:
    def __init__(self, body): self.body = body
    async def json(self): return self.body
    async def is_disconnected(self): return False

class Job:
    def __init__(self, *args, **kw): self.kw = kw
    def __aiter__(self): return self.run()
    async def run(self):
        for s in STATE['chunks']: yield {'text': s}
        yield {'eos': True, 'new_tokens': STATE['n'], 'eos_reason': 'stop'}
    async def cancel(self): pass

STATE = {}

def scope(hf):
    tree = ast.parse(POD.read_text())
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in {'text_of','prepare','typed','split_output','error','chat'}:
            n.decorator_list = []
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'CONTEXT','STOP_IDS','THINK_END','TOOL_CALL','FUNCTION','PARAMETER'} for t in n.targets):
            nodes.append(n)
    env = dict(ast=ast, re=re, json=json, asyncio=asyncio, time=time, uuid=uuid, Request=Request,
        JSONResponse=JSONResponse, StreamingResponse=StreamingResponse, AsyncJob=Job,
        ComboSampler=lambda **kw: kw, torch=types.SimpleNamespace(long=None, tensor=lambda x, **kw:x),
        S={'names':['nanbeige'], 'hf': hf, 'gen':None})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(POD), 'exec'), env)
    return env


def normalized(b):
    s = b.decode('utf8') if isinstance(b, bytes) else b
    s = re.sub(r'chatcmpl-tool-[0-9a-f]{16}', 'chatcmpl-tool-' + '0'*16, s)
    s = re.sub(r'chatcmpl-[0-9a-f]{32}', 'chatcmpl-' + '0'*32, s)
    return re.sub(r'("created": ?)[0-9]+', r'\g<1>0', s)

async def generate():
    hf = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    env = scope(hf)
    cache = {}
    class CachedHF:
        def apply_chat_template(self, messages, **kw):
            key = json.dumps([messages, kw], ensure_ascii=False)
            if key not in cache:
                text = hf.apply_chat_template(messages, **kw)
                cache[key] = (text, hf(text, add_special_tokens=False)['input_ids'])
            STATE['prompt'] = cache[key][1]
            return cache[key][0]
        def __call__(self, prompt, **kw): return {'input_ids': STATE['prompt']}
    env['S']['hf'] = CachedHF()
    n = 0
    with (ROOT/'input.jsonl').open('w') as inp, (ROOT/'expected.jsonl').open('w') as expected:
        for c in new_cases():
            ids = hf(c['raw'], add_special_tokens=False)['input_ids']
            c['ids'] = ids
            STATE.update(n=len(ids), chunks=[], prompt=[])
            sent = ''
            for j in range(1, len(ids)+1):
                s = hf.decode(ids[:j], skip_special_tokens=False)
                if s.endswith('\ufffd'): continue
                if not s.startswith(sent): raise RuntimeError('nonprefix decode')
                STATE['chunks'].append(s[len(sent):]); sent = s
                if '</think>' in s:
                    remaining = hf.decode(ids, skip_special_tokens=False)[len(sent):]
                    if remaining: STATE['chunks'].append(remaining)
                    break
            out = {'name':c['name'], 'group':c['group']}
            for stream in (False, True):
                key = 'sse' if stream else 'json'
                try:
                    resp = await env['chat'](Request(dict(c['request'], stream=stream)))
                    if isinstance(resp, StreamingResponse):
                        b = ''.join([v async for v in resp.body_iterator]).encode('utf8')
                    elif isinstance(resp, JSONResponse): b = resp.body
                    else: b = JSONResponse(resp).body
                    out[key] = normalized(b)
                except Exception as e:
                    out[key+'_error'] = type(e).__name__ + ': ' + str(e)
            out['prompt'] = STATE['prompt']
            inp.write(json.dumps(c, ensure_ascii=False)+'\n')
            expected.write(json.dumps(out, ensure_ascii=True)+'\n')
            n += 1
            if n % 1000 == 0: print('labelled', n, flush=True)
    print('TOTAL', n, flush=True)


def compare():
    import failure_compare
    return failure_compare.compare(sys.modules[__name__])


if __name__ == '__main__':
    if sys.argv[1] == 'generate': asyncio.run(generate())
    else: compare()
