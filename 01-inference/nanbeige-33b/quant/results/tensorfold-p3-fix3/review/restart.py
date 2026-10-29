import sys,os,json,subprocess,signal,copy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama')
sys.path.insert(0,str(ROOT/'tools/zig'))
import llama_full_gate as gate
from llama_gate_state import atomic,identity,fingerprint
S=Path('/tmp/nanbeige-review-a3c96094/restart');S.mkdir(exist_ok=True)
if len(sys.argv)>1:
 mode=sys.argv[1]; root=S/'case'
 args=SimpleNamespace(out=root/'out',work=root/'work',budget_minutes=150,only=None,skip=None,plan_only=False,sm='89',timeout=20)
 for p in [args.out,args.work]:p.mkdir(parents=True,exist_ok=True)
 jobs=[dict(label=l,command=['cpu',l],state='NOT RUN') for l in ['p1-reference','p1-logits','p1-greedy']]
 bind=fingerprint(identity(root/'inputs'))
 if mode=='subset':args.only='p1-logits'
 def worker(job,log,limit):
  with (root/'calls').open('a') as f:f.write(job['label']+'\n')
  if mode=='crash' and job['label']=='p1-logits':os.kill(os.getpid(),signal.SIGKILL)
  log.write('good\n');return 0
 gate.execute(args,jobs,bind,{},worker=worker)
 sys.exit()
root=S/'case';(root/'inputs').mkdir(parents=True,exist_ok=True);(root/'inputs/weights').write_text('original')
def run(mode):
 p=subprocess.run([sys.executable,__file__,mode],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 (S/(mode+'.log')).write_text(p.stdout+p.stderr);return p
checks={}
p=run('crash');checks['sigkill_exit']=p.returncode==-9
saved=json.loads((root/'out/summary.json').read_text());checks['running_durable']=[j['state'] for j in saved['jobs']]==['PASS','RUNNING','NOT RUN']
checks['restart_success']=run('resume').returncode==0
calls=(root/'calls').read_text().splitlines();checks['pass_not_rerun_running_retried']=calls==['p1-reference','p1-logits','p1-logits','p1-greedy']
checks['idempotent']=run('again').returncode==0 and (root/'calls').read_text().splitlines()==calls
(root/'out/summary.tmp').write_text('{torn');checks['torn_temporary_ignored']=run('torn').returncode==0
(root/'inputs/weights').write_text('mutated');p=run('input');checks['input_tamper_refused']=p.returncode!=0 and 'fingerprint mismatch' in p.stderr
(root/'inputs/weights').write_text('original');(root/'out/p1-reference.log').write_text('tampered');p=run('artifact');checks['artifact_tamper_refused']=p.returncode!=0 and 'artifact mismatch' in p.stderr
(root/'out/p1-reference.log').write_text('good\n');p=run('subset');checks['subset_partial']=p.returncode==0 and json.loads((root/'out/summary.json').read_text())['state']=='PARTIAL'
checks['skip_dependency_refused']=False
try:gate.select([dict(label=l) for l in ['p1-reference','p1-logits']],None,'p1-reference')
except ValueError:checks['skip_dependency_refused']=True
# Hard calibration admission including already spent wall time.
with patch.object(gate,'estimate',return_value=9010):
 args=SimpleNamespace(out=S/'budget',work=S/'budget-work',budget_minutes=150,only=None,skip=None,plan_only=False,sm='89',timeout=20)
 args.out.mkdir(exist_ok=True);args.work.mkdir(exist_ok=True)
 try:gate.execute(args,[dict(label='p1-reference',command=[],state='NOT RUN')],'budget',{},worker=lambda *a: (_ for _ in ()).throw(AssertionError('ran')))
 except RuntimeError:checks['budget_refused']=json.loads((args.out/'summary.json').read_text())['state']=='BUDGET'
(S/'results.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2));assert all(checks.values())
