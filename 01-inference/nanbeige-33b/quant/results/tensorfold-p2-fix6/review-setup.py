from pathlib import Path
import shutil
engine=Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama')
for old,new in [('/tmp/nanbeige-review-7fe2dce7','/tmp/nanbeige-fix6/review5'),('/tmp/nanbeige-review-891015b9','/tmp/nanbeige-fix6/review6')]:
 old,new=Path(old),Path(new);new.mkdir(exist_ok=True)
 for src in old.glob('*'):
  if src.suffix in ['.py','.zig','.sh'] or src.name=='newer-names.json':
   if not (new/src.name).exists():shutil.copy(src,new/src.name)
 for directory in ['source','deps','ucd']:
  target=engine if directory=='source' else old/directory
  if target.exists() and not (new/directory).exists():(new/directory).symlink_to(target)
 for directory in ['extended','targeted']:(new/directory).mkdir(exist_ok=True)
 src=new/'independent.py';s=src.read_text().replace(str(old),str(new));start=s.index('def compare():');end=s.index("\nif __name__",start)
 s=s[:start]+'''def compare():
    import failure_compare
    return failure_compare.compare(sys.modules[__name__])

'''+s[end:];src.write_text(s)
 for name in ['rerun.sh','extended.py','targeted.py']:
  p=new/name
  if p.exists():p.write_text(p.read_text().replace(str(old),str(new)))
 shutil.copy('/tmp/nanbeige-fix6/failure_compare.py',new/'failure_compare.py')
