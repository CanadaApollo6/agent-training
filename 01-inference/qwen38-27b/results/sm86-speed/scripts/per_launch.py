"""Kernel time per graph launch (verify forwards) and per eager stretch, by kernel, averaged over launches of the median node count."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
ks = db.execute("select start, end, shortName, graphId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
per = collections.defaultdict(list)
for s, e, n, g, c in ks:
    if g is not None: per[c].append((s, e, names[n]))
sizes = collections.Counter(len(v) for v in per.values())
print("graph node counts:", sizes.most_common(6))
common = sizes.most_common(1)[0][0]
sel = [v for v in per.values() if len(v) == common]
tot = collections.Counter(); cnt = collections.Counter()
for v in sel:
    for s, e, n in v: tot[n] += e - s; cnt[n] += 1
L = len(sel); allk = sum(tot.values()) / L
print(f"{L} launches of {common} nodes: kernel time {allk/1e3:.0f} us, span {sum(max(e for _,e,_ in v)-min(s for s,_,_ in v) for v in sel)/L/1e3:.0f} us")
for n, t in tot.most_common(14): print(f"  {t/L/1e3:8.1f} us  {cnt[n]/L:5.0f}x  {n[:60]}")
eager = collections.Counter(); ecnt = collections.Counter()
for s, e, n, g, c in ks:
    if g is None: eager[names[n]] += e - s; ecnt[names[n]] += 1
nl = len(per)
print(f"eager kernels per graph launch: {sum(ecnt.values())/nl:.0f}, time {sum(eager.values())/nl/1e3:.0f} us")
for n, t in eager.most_common(12): print(f"  {t/nl/1e3:8.1f} us  {ecnt[n]/nl:5.1f}x  {n[:60]}")
