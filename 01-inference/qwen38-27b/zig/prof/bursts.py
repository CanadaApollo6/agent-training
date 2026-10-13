import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
try: k += [(s, e, 'GRAPH') for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
except Exception: pass
k.sort()
bursts = [[k[0]]]
for x in k[1:]:
    if x[0] - bursts[-1][-1][1] > 50e6: bursts.append([x])
    else: bursts[-1].append(x)
for i, b in enumerate(bursts):
    span = (b[-1][1] - b[0][0]) / 1e9
    busy = sum(e - s for s, e, _ in b) / 1e9
    print(f"burst {i}: {len(b)} kernels span {span:.2f}s busy {busy:.2f}s")
if len(sys.argv) > 2:
    b = bursts[int(sys.argv[2])]
    tot = collections.defaultdict(lambda: [0, 0])
    for s, e, n in b: tot[n][0] += e - s; tot[n][1] += 1
    for n, (t, c) in sorted(tot.items(), key=lambda x: -x[1][0])[:25]: print(f"{t/1e6:9.1f} ms {c:7d} {n[:90]}")
