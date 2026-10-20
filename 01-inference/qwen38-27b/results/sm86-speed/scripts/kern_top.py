import csv,sys
lines=open(sys.argv[1]).read().splitlines()
i=next(k for k,l in enumerate(lines) if l.startswith('Time (%)'))
rows=list(csv.DictReader(lines[i:]))
tot=sum(float(r['Total Time (ns)']) for r in rows)
print("total GPU kernel ms", round(tot/1e6,1), "kernels", sum(int(r['Instances']) for r in rows))
for r in rows[:int(sys.argv[2]) if len(sys.argv)>2 else 30]:
    print(f"{float(r['Time (%)']):5.1f}% {float(r['Total Time (ns)'])/1e6:8.1f}ms n={r['Instances']:>7} avg={float(r['Avg (ns)'])/1e3:7.1f}us med={float(r['Med (ns)'])/1e3:7.1f} {r['Name'][:100]}")
