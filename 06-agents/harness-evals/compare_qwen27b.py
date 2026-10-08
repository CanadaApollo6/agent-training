"""Qwen3.8-27B (bf16, vLLM, reasoning effort medium, untrained) vs R1s-SD c32 under prime_agent on the held-out 20 x 3:
solves, lost runs, stops, and feature use (feature_audit.analyse). usage: python3 compare_qwen27b.py"""
import json, sys, glob, collections, statistics as st
sys.path.insert(0, '.')
import feature_audit as fa
def load(pat):
    rows=[]
    for f in sorted(glob.glob(f'outputs/primeintellect/terminal-bench-2--{pat}--prime_agent--*/traces.jsonl')):
        for l in open(f):
            try: r=fa.analyse(json.loads(l))
            except json.JSONDecodeError: continue
            if r: rows.append(r)
    return rows
for name,pat in [('qwen27b','qwen38-27b-pa-a[123]'),('r1s-sd c32','ornith35b-r1s-sd-c32-a[123]')]:
    R=load(pat); U=[r for r in R if not r['lost']]
    print(f"== {name}: rollouts {len(R)}, lost {len(R)-len(U)}, solved {sum(r['solved'] for r in R)}, usable solved {sum(r['solved'] for r in U)}/{len(U)}")
    print(" stops:",dict(collections.Counter(r['stop'] for r in R)))
    print(" ended on 32K cut:",sum(r['ended_on_cut'] for r in U)," median turns %.0f"%st.median([r['turns'] for r in U]), " runs>100K prompt:",sum(r['peak_prompt']>100000 for r in U))
    n=len(U)
    for k in ['edit','compact','spawn','harness_layer','bash_await','bash_background','attach_image','py_file_io']:
        print(f"  {k}: {sum(1 for r in U if r['used'].get(k))}/{n}", end='')
    print()
    print("  subprocess:",sum(1 for r in U if r['anti'].get('subprocess')),"/",n," missing-tool runs:",sum(1 for r in U if r['bad_tools']),"/",n)
    per=collections.defaultdict(list)
    for r in U: per[r['task']].append(r['solved'])
    print("  per-task solved:",{t.split('/')[-1][:22]:sum(v) for t,v in sorted(per.items())})
Q=[r for r in load('qwen38-27b-pa-a[123]')]; S=[r for r in load('ornith35b-r1s-sd-c32-a[123]')]
def per(R):
    d=collections.defaultdict(list)
    for r in R:
        if not r['lost']: d[r['task']].append(r['solved'])
    return d
q,s=per(Q),per(S)
both=[t for t in q if t in s]
qa=[sum(q[t])/len(q[t]) for t in both]; sa=[sum(s[t])/len(s[t]) for t in both]
print("tasks both:",len(both),"qwen %.0f%%  r1s-sd %.0f%%"%(100*st.mean(qa),100*st.mean(sa)))
better=sum(a>b for a,b in zip(qa,sa)); worse=sum(a<b for a,b in zip(qa,sa)); print("qwen better on",better,"worse on",worse)
lostq=collections.Counter(r['task'].split('/')[-1][:24] for r in Q if r['lost']); print("qwen lost:",dict(lostq))
print("qwen timeouts:",dict(collections.Counter(r['task'].split('/')[-1][:24] for r in Q if r['stop']=='agent_timeout')))
