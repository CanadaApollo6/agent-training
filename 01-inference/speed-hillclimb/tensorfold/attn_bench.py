"""Tree attention's three kernels at one long context (4 chained rows, Qwen3.6's 16/2 heads of 256): µs each under the
torch profiler, and whether the working tree's output bits equal git HEAD's (the reference module is loaded from HEAD).

Usage: python attn_bench.py [KEYS=70000]   (in envs/tensorfold060; edit the variant list to sweep the module's knobs)
"""
import sys, torch, itertools
from torch.profiler import profile, ProfilerActivity
from tensorfold.cuda.kernels import attention as A
dev = "cuda"; torch.manual_seed(0)
H, HK, D, W = 16, 2, 256, 4
def setup(P):
    torch.manual_seed(0)
    kc = torch.randn(P + 64, HK, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(P + 64, HK, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(W, H, D, device=dev, dtype=torch.bfloat16)
    kn = torch.randn(W, HK, D, device=dev, dtype=torch.bfloat16); vn = torch.randn_like(kn)
    plan = A.plan([[-1, 0, 1, 2]], [P], H // HK, dev)
    offs = torch.tensor(A.offsets([(kc, vc)], dev), dtype=torch.int64, device=dev).view(1, 2)
    return lambda: A.attention(q, kn, vn, offs, plan, scale=D ** -0.5)
def timed(f, reps=30):
    f(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps): f()
        torch.cuda.synchronize()
    t = {}
    for e in p.key_averages():
        for k in ("_shared", "_merge", "_tail"):
            if e.key.startswith(k): t[k] = e.device_time / 1  # avg us
    return t
P = int(sys.argv[1]) if len(sys.argv) > 1 else 70000
a = torch.randn(4096, 4096, device=dev)
import time; t0 = time.time()
while time.time() - t0 < 1: a @ a
f = setup(P)
A.SHARED_STAGES, A.MERGE_STAGES, A.MERGE_COLUMNS = 1, 1, 64
import importlib.util, subprocess
src = subprocess.run(["git","-C","/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-060","show","HEAD:src/tensorfold/cuda/kernels/attention.py"],capture_output=True,text=True).stdout
open("/tmp/attention_head.py","w").write(src)
spec = importlib.util.spec_from_file_location("attention_head", "/tmp/attention_head.py"); H0 = importlib.util.module_from_spec(spec); sys.modules["attention_head"] = H0; spec.loader.exec_module(H0)
A_new = A; A = H0; f0 = setup(P); ref = f0().clone(); t0r = timed(f0); A = A_new
print("HEAD kernels:", {k: round(v,1) for k,v in t0r.items()})
print(f"P={P}  KV bytes/layer {2*P*HK*D*2/1e6:.1f} MB")
for ss, ms, mc in [(1,1,16),(1,1,8),(1,1,16),(1,1,8)]:
    A.SHARED_STAGES, A.MERGE_STAGES, A.MERGE_COLUMNS = ss, ms, mc
    out = f(); same = torch.equal(out, ref)
    t = timed(f)
    gbs = 2*P*HK*D*2/ (t["_shared"]*1e-6) / 1e9
    print(f"shared stages {ss} merge stages {ms} cols {mc:3}: shared {t['_shared']:7.1f} us ({gbs:4.0f} GB/s)  merge {t['_merge']:6.1f}  tail {t['_tail']:5.1f}  bits same {same}")
