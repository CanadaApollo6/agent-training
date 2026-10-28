import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

out = Path(__file__).resolve().parent
label, selected = sys.argv[1:3]
root = Path(sys.argv[3]) if len(sys.argv) > 3 else out.parents[3] / 'tools/TensorFold-llama'
line = next(s for s in (out/'focus-verbose.log').read_text().splitlines() if s.startswith('info(verbose): ') and 'zig test ' in s)
cmd = shlex.split(line.removeprefix('info(verbose): '))
cmd.remove('--listen=-')
cmd[cmd.index('--test-filter')+1] = selected
cmd += ['-femit-bin='+str(out/'host-cache'/label)]
env = dict(os.environ, TF_LLAMA_MODEL=str(out.parents[1]/'models/3.5-nb'))
with (out/(label+'.log')).open('w') as log:
    result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
with (out/'commands.jsonl').open('a') as f:
    f.write(json.dumps(dict(label=label, command=cmd, cwd=str(root), exit=result.returncode))+'\n')
print(label, 'exit',result.returncode)
sys.exit(result.returncode)
