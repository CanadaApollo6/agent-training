"""Render actual Nanbeige pilot requests using the pod's message preparation."""
import ast
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

root, dest = map(Path, sys.argv[1:])
model = root / '01-inference/nanbeige-33b/quant/models/3.5-nb'
pod = root / '01-inference/nanbeige-33b/quant/serve_exl3.py'
tree = ast.parse(pod.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('text_of', 'prepare')]
namespace = {'json': json}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(pod), 'exec'), namespace)
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
histories, provenance = [], []
for source in sorted((root / '06-agents/harness-evals/outputs/primeintellect').glob('*nanbeige-exl3*/traces.jsonl')):
    for line in source.open():
        record = json.loads(line)
        for trace in record['traces']:
            call = max(trace['calls'], key=lambda c: c.get('usage', {}).get('prompt_tokens', 0))
            reported = call.get('usage', {}).get('prompt_tokens', 0)
            if reported < 100000:
                continue
            nodes = trace['nodes']
            index = nodes[call['node']].get('parent')
            chain = []
            while index is not None:
                chain.append(index)
                index = nodes[index].get('parent')
            chain.reverse()
            messages = json.loads(json.dumps([nodes[i]['message'] for i in chain]))
            for message in messages:
                if message.get('tool_calls'):
                    message['tool_calls'] = [c if 'function' in c else {'id': c.get('id', ''),
                        'type': 'function', 'function': {k: c[k] for k in ('name', 'arguments') if k in c}}
                        for c in message['tool_calls']]
            messages = namespace['prepare'](messages)
            tools = nodes[chain[0]].get('tools') or trace.get('tools') or []
            tools = [t if 'function' in t else {'type': 'function', 'function': {k: v for k, v in t.items() if k != 'type'}} for t in tools]
            prompt = tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, add_special_tokens=False)['input_ids']
            assert len(ids) == reported, (trace['id'], len(ids), reported)
            info = dict(source=str(source.relative_to(root)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        trace=trace['id'], task=record['task']['data']['name'], call_node=call['node'],
                        parent_chain=chain, reported_prompt_tokens=reported, tokens=len(ids), bos=ids[0],
                        token_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
            print(json.dumps({k: v for k, v in info.items() if k != 'parent_chain'}), flush=True)
            if len(ids) >= 100000 and ids[0] == 166100:
                histories.append(ids)
                provenance.append(info)
if len(histories) < 3:
    raise SystemExit(f'STOP: only {len(histories)} qualifying histories; upload nothing')
# Keep three different tasks, in source order, without changing their tokens.
selected, selected_info, tasks = [], [], set()
for ids, info in zip(histories, provenance):
    if info['task'] not in tasks:
        selected.append(ids)
        selected_info.append(info)
        tasks.add(info['task'])
    if len(selected) == 3:
        break
assert len(selected) == 3
assert all(type(t) is int and 0 <= t < 166912 for h in selected for t in h)
for length in (20000, 60000, 100000):
    assert len({tuple(h[:length]) for h in selected}) == 3
assert len({tuple([h[0]] + h[-127:]) for h in selected}) == 3
dest.mkdir(parents=True, exist_ok=True)
(dest / 'histories.json').write_text(json.dumps(selected, separators=(',', ':')) + '\n')
(dest.parent / 'histories-provenance.json').write_text(json.dumps(dict(tokenizer='transformers==4.57.1',
    rendering='serve_exl3.prepare, apply_chat_template tools, add_generation_prompt=True, tokenize=False; encode add_special_tokens=False',
    selected=selected_info, candidates=provenance), indent=2) + '\n')
