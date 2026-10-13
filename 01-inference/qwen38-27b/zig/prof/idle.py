import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
t0, t1 = k[0][0], k[-1][1]
k = [x for x in k if x[0] > t0 + 0.8 * (t1 - t0) and names[x[2]] != 'prefill_kernel']
rounds = sum(1 for i in range(1, len(k)) if k[i][3] and not k[i-1][3])
gap = collections.defaultdict(lambda: [0, 0])
for a, b in zip(k, k[1:]):
    if a[3] and b[3]: continue
    key = names[a[2]][:30] + ' -> ' + names[b[2]][:30]
    gap[key][0] += b[0] - a[1]; gap[key][1] += 1
print("rounds", rounds)
for n, (t, c) in sorted(gap.items(), key=lambda x: -x[1][0])[:20]: print(f"{t/rounds/1e3:7.1f} us/round {c/rounds:4.1f}x {n}")
g = collections.defaultdict(int)
for s, e, n, gn in k:
    if gn: g[names[n]] += e - s
print("-- graph kernels")
for n, t in sorted(g.items(), key=lambda x: -x[1])[:12]: print(f"{t/rounds/1e3:7.1f} us/round {n[:70]}")
