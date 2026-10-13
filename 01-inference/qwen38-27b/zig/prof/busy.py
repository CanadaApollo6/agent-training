import sqlite3, sys, collections, statistics
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
k += [(s, e, 'GRAPH') for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
k.sort()
t0, t1 = k[0][0], k[-1][1]
k = [x for x in k if x[0] > t0 + 0.85 * (t1 - t0)]
# rounds: start at each first GRAPH after non-GRAPH
starts = [i for i in range(1, len(k)) if k[i][2] == 'GRAPH' and k[i-1][2] != 'GRAPH']
rows = []
for a, b in zip(starts, starts[1:]):
    seg = k[a:b]
    wall = seg[-1][1] - seg[0][0] + (k[b][0] - seg[-1][1])
    busy = sum(e - s for s, e, _ in seg)
    g = sum(e - s for s, e, n in seg if n == 'GRAPH')
    gaps = collections.Counter()
    for x, y in zip(seg, seg[1:] + [k[b]]):
        gaps[x[2][:24] + '>' + y[2][:24]] += y[0] - x[1]
    rows.append((wall, busy, g, gaps))
print("rounds", len(rows), "median wall us", statistics.median(r[0] for r in rows)/1e3, "busy", statistics.median(r[1] for r in rows)/1e3, "graph", statistics.median(r[2] for r in rows)/1e3)
tot = collections.Counter()
for r in rows: tot.update(r[3])
for n, v in tot.most_common(10): print(f"{v/len(rows)/1e3:7.1f} us/round {n}")
for key in ['tf_argmax_rows_i32_kerne>replay_kernel', 'tf_draft_pick>tf_tree_paths']:
    v = sorted(r[3][key] / 1e3 for r in rows)
    print(key, "pcts 10/50/90/max", [round(v[int(p * (len(v) - 1))]) for p in (0.1, 0.5, 0.9, 1.0)])
