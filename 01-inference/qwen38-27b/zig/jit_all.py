"""capture.py's jit.json for every Triton function the 27B's CUDA modules define (no GPU work)."""
import importlib, json, pkgutil, sys
sys.path.insert(0, sys.argv[2])
import capture
import tensorfold.families.qwen3_5.cuda as fam
import tensorfold.cuda.kernels as kern
for pkg in (fam, kern):
    for m in pkgutil.iter_modules(pkg.__path__):
        try:
            importlib.import_module(f"{pkg.__name__}.{m.name}")
        except Exception as e:
            print("skip", m.name, type(e).__name__, file=sys.stderr)
spec = capture.specialization()
open(sys.argv[1], "w").write(json.dumps(spec, indent=1, sort_keys=True) + "\n")
print(len(spec), "functions")
