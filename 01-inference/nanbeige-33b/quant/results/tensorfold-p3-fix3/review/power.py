import numpy as np, math, json
from pathlib import Path
rng=np.random.default_rng(20261029)
V=166912; N=8192; TRIALS=2000
radius=2*math.sqrt(math.log(4*36*8/.01)/(2*N)); cutoff=.10-radius

def norm(p): return p/p.sum()
def transform(z,t,k,p,mask=None):
    x=np.exp((z-z.max())/t)
    order=np.argsort(-x)
    if k: x[order[k:]]=0
    x=norm(x)
    order=np.argsort(-x)
    keep=np.cumsum(x[order])-x[order]<p
    x[order[~keep]]=0
    if mask is not None: x*=mask
    return norm(x)

def power(p,b):
    pb=np.bincount(ids%8,weights=p,minlength=8); bb=np.bincount(ids%8,weights=b,minlength=8)
    a=rng.multinomial(N,pb,size=TRIALS); c=rng.multinomial(N,bb,size=TRIALS)
    old=np.mean(np.max(abs(a-c),axis=1)/N>=cutoff)
    # Predeclared bug-direction event, exact Hoeffding null cutoff, 576 one-sided tests.
    event=p>b; pa=p[event].sum(); ba=b[event].sum()
    n=32768
    delta=math.sqrt(math.log(576/.01)/n)
    x=rng.binomial(n,pa,TRIALS); y=rng.binomial(n,ba,TRIALS)
    new=np.mean((x-y)/n>delta)
    return dict(tv=float(abs(p-b).sum()/2), max_bin=float(max(abs(pb-bb))), old_power=float(old),new_power=float(new), event_delta=float(pa-ba))
rows=[]
for exponent in [1.05,1.25,1.6]:
 for context in range(3):
    ids=rng.permutation(V)
    z=-exponent*np.log(np.arange(1,V+1))+rng.normal(0,.1,V)
    zd=z+rng.normal(0,.35,V)
    for variant,(t,k,pp) in enumerate([(0.3,0,1),(0.7,3,.8),(1,20,.95),(1.3,50,.9)]):
        p=transform(z,t,k,pp); q=transform(zd,t,k,pp)
        accepted=np.minimum(p,q); rejection=1-accepted.sum()
        bugs={'residual uses p':accepted+rejection*p,
              'residual reversed':accepted+rejection*norm(np.maximum(q-p,0)) if rejection>1e-12 else p,
              'top-k/p from draft':transform(z,t,0,1,mask=q>0),
              'temperature twice':transform(z,t*t,k,pp)}
        if variant==3:
            zp=z.copy(); zp[:7]-=np.array([1.3,1,.0,1.6,0,1,1])
            penalized=transform(zp,t,k,pp)
            bugs['penalties draft only']=p
        for name,b in bugs.items():
            target=penalized if name=='penalties draft only' else p
            rows.append(dict(exponent=exponent,context=context,variant=variant,bug=name,**power(target,b)))
# An 0.08 token-mass swap wholly inside a modulo bin is invisible.
ids=np.arange(V); p=np.zeros(V); p[[1,9,2,3]]=[.25,.25,.3,.2]; b=p.copy(); b[1]+=.08;b[9]-=.08
rows.append(dict(bug='within-bin swap',**power(p,b)))
summary={}
for name in dict.fromkeys(r['bug'] for r in rows):
 r=[x for x in rows if x['bug']==name and x['tv']>1e-8]
 powered=[x for x in r if x['tv']>=.03]
 summary[name]=dict(cells=len(r),tv_range=[min(x['tv'] for x in r),max(x['tv'] for x in r)],old_power_mean=np.mean([x['old_power'] for x in r]),old_power_range=[min(x['old_power'] for x in r),max(x['old_power'] for x in r)],new_tv03_cells=len(powered),new_tv03_min_power=min([x['new_power'] for x in powered],default=None))
report=dict(seed=20261029,synthetic=True,vocab=V,trials=TRIALS,draws=N,radius=radius,cutoff=cutoff,summary=summary,rows=rows)
Path('/tmp/nanbeige-review-a3c96094/power.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
