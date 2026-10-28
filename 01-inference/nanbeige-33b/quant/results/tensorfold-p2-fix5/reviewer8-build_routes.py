import subprocess
from pathlib import Path
p=Path(__file__).parent
r=p/'source/zig/src'
args=[str(Path.home()/'.local/share/zig-x86_64-linux-0.17.0/zig'),'test','--test-filter','independent','--test-no-exec','-femit-bin='+str(p/'routes'),'-lc','-O','ReleaseSafe']
def mod(name,path,deps=()):
    for d in list(deps) + ([] if name == 'sys' else ['sys']): args.extend(['--dep',d])
    args.append('-M'+name+'='+str(path))
mod('root',p/'routes.zig',['json','engine_api','tokenizer','template','living','dsv41_chat','drama_prompt','lanes'])
mod('json',r/'server/json.zig')
mod('engine_api',r/'core/engine_api.zig',['lanes'])
mod('tokenizer',r/'core/tokenizer/tokenizer.zig')
mod('template',r/'core/template/template.zig')
mod('living',r/'living/living.zig')
mod('dsv41_chat',r/'families/deepseek_v41/serve/template.zig',['json'])
mod('drama_prompt',r/'families/drama3/prompt.zig')
mod('lanes',r/'core/lanes/lanes.zig')
mod('sys',r/'platform/sys.zig')
with (p/'routes-build.log').open('w') as f: p1=subprocess.run(args,stdout=f,stderr=f)
print(p1.returncode)
