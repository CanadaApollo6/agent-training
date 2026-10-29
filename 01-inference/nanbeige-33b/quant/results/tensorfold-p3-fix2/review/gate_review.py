import contextlib,io,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, '/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig')
import llama_full_gate as gate
root=Path('/tmp/nanbeige-review-97d4ac9c/gate-dry');root.mkdir(exist_ok=True)
(root/'histories.json').write_text(json.dumps([[166100]+[i]*99999 for i in (1,2,3)]))
for n in ['model','draft']:
 (root/n).mkdir(exist_ok=True);(root/n/'config.json').write_text('{}')
for n in ['pod','smoke','baseline-runner','runner','binary']:(root/n).touch()
args=['gate','--sm','86']
for n in ['model','draft','pod','smoke','baseline-runner','runner','binary','work','out']:args += ['--'+n,str(root/n)]
args+=['--histories',str(root/'histories.json')]
with patch.object(sys,'argv',args+['--plan-only']):gate.main()
summary=json.loads((root/'out/summary.json').read_text());assert len(summary['jobs'])==22
print('actual plan-only: 22 jobs; no GPU query or launch')
calls=[]
def query(cmd,**kwargs):return 'head\n' if cmd[0]=='git' else 'FAKE_CPU_TEST, 8.6, 49152\n'
def crash(cmd,**kwargs):
 calls.append(cmd)
 if len(calls)==3:raise KeyboardInterrupt()
 return SimpleNamespace(returncode=0,stdout='',stderr='')
with patch.object(sys,'argv',args),patch.object(gate.subprocess,'check_output',side_effect=query),patch.object(gate.subprocess,'run',side_effect=crash):
 try:gate.main()
 except KeyboardInterrupt:pass
summary=json.loads((root/'out/summary.json').read_text());print('after interrupted job 3:',[j['state'] for j in summary['jobs'][:4]], 'overall='+summary['state'])
first=[]
def observe(cmd,**kwargs):first.append(cmd);raise KeyboardInterrupt()
with patch.object(sys,'argv',args),patch.object(gate.subprocess,'check_output',side_effect=query),patch.object(gate.subprocess,'run',side_effect=observe):
 try:gate.main()
 except KeyboardInterrupt:pass
assert first[0]==calls[0]
summary=json.loads((root/'out/summary.json').read_text());assert all(j['state']=='NOT RUN' for j in summary['jobs'])
print('restart launches p1-reference again and resets all previous PASS statuses to NOT RUN')
print('P4 prompt forwards =',sum(n*3*10 for n in (20000,60000,100000)))
