import sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
r = GGUFReader(sys.argv[2])
for k in r.fields:
    if 'gating' in k or 'expert' in k and 'count' in k: 
        f=r.fields[k]; print(k, f.parts[f.data[0]])
tot=0; act=0; groups={}
for t in r.tensors:
    n=t.name; b=int(t.n_bytes); tot+=b
    if 'token_embd' in n: g='embed/head(tied)'; a=b
    elif '_exps' in n: g='routed experts'; a=b*4/64
    elif 'ffn_' in n and 'gate_inp' not in n and 'exp_probs' not in n: g='dense mlp (2 layers)'; a=b
    elif 'shortconv' in n: g='short conv'; a=b
    elif 'attn' in n: g='attention+norms'; a=b
    else: g='other'; a=b
    groups.setdefault(g,[0,0,set()]); groups[g][0]+=b; groups[g][1]+=a; groups[g][2].add(t.tensor_type.name)
    act+=a
for g,(b,a,ty) in groups.items(): print(f"{g:22s} stored {b/1e9:6.2f} GB  read/token {a/1e9:5.3f} GB  {sorted(ty)}")
print(f"total stored {tot/1e9:.2f} GB; read per token {act/1e9:.3f} GB; ceiling @830 GB/s = {830/(act/1e9):.0f} tok/s")
