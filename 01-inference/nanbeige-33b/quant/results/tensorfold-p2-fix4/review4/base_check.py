from pathlib import Path
import os,subprocess,json
import differential as d
p=Path(__file__).parent
repo=p/'source'
base=p/'base/zig/src/server'
base.mkdir(parents=True,exist_ok=True)
for path in (repo/'zig/src/server').iterdir():
    dst=base/path.name
    if path.name in {'nanbeige.zig','tool_qwen.zig'}:
        dst.write_bytes(subprocess.check_output(['git','show','6d56a1c4:zig/src/server/'+path.name],cwd=repo))
    elif not dst.exists():dst.symlink_to(path.resolve(),target_is_directory=path.is_dir())
(p/'base-parser.zig').write_text((p/'parser.zig').read_text().replace('source/zig/src/server/','base/zig/src/server/'))
args=[str(Path.home()/'.local/share/zig-x86_64-linux-0.17.0/zig'),'test','--dep','json','-Mroot='+str(p/'base-parser.zig'),'-Mjson='+str(repo/'zig/src/server/json.zig'),'-lc','--test-filter','independent','--test-no-exec','-femit-bin='+str(p/'base-parser')]
subprocess.run(args,check=True)
cases=d.generate()[:5]
path=p/'directed-input.jsonl';path.write_text(''.join(json.dumps(c)+'\n' for c in cases))
for name in ['base-parser','parser']:
    proc=subprocess.run([str(p/name)],env=dict(os.environ,REVIEW_INPUT=str(path),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,check=True)
    out=[json.loads(line.split('RESULT ',1)[1]) for line in proc.stderr.splitlines() if 'RESULT ' in line]
    print(name)
    for c,g in zip(cases,out):print(c['label'],d.normalized(g)==d.normalized(d.expected(c)),json.dumps(d.normalized(g)))
