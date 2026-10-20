"""Busy fraction, launches and kernel time per round from an nsys sqlite; gaps over 20 ms count as idle between requests."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
rounds_hint = int(sys.argv[2]) if len(sys.argv) > 2 else 0
names = dict(db.execute("select id, value from StringIds"))
ks = db.execute("select start, end, shortName, graphId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
cp = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
ev = sorted([(s, e, names[n], g, c) for s, e, n, g, c in ks] + [(s, e, "memcpy", None, None) for s, e in cp])
busy = 0; span = 0; cur_s, cur_e = ev[0][0], ev[0][1]; idle_gaps = 0
for s, e, *_ in ev[1:]:
    if s > cur_e:
        gap = s - cur_e
        busy += cur_e - cur_s
        if gap < 20e6: span += s - cur_s
        else: span += cur_e - cur_s; idle_gaps += 1
        cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
busy += cur_e - cur_s; span += cur_e - cur_s
graph = [k for k in ks if k[3] is not None]
launches = collections.Counter(k[4] for k in graph)
print(f"kernels {len(ks)} (in graphs {len(graph)}, graph launches {len(launches)}), memcpys {len(cp)}")
print(f"active span {span/1e6:.1f} ms, busy {busy/1e6:.1f} ms, busy fraction {busy/span:.3f}, request gaps {idle_gaps}")
if rounds_hint:
    print(f"per round ({rounds_hint}): wall {span/1e6/rounds_hint:.2f} ms, busy {busy/1e6/rounds_hint:.2f} ms, "
          f"kernels {len(ks)/rounds_hint:.0f} ({(len(ks)-len(graph))/rounds_hint:.0f} eager), memcpys {len(cp)/rounds_hint:.1f}")
# per graph launch: span and busy
per = collections.defaultdict(list)
for s, e, n, g, c in ks:
    if g is not None: per[c].append((s, e))
spans = [max(e for _, e in v) - min(s for s, _ in v) for v in per.values()]
if spans:
    spans.sort(); print(f"graph launch span median {spans[len(spans)//2]/1e6:.2f} ms, nodes median {sorted(len(v) for v in per.values())[len(per)//2]}")
