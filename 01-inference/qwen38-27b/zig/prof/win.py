import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
hits = [(k[i][1], k[i+1][0]) for i in range(len(k)-1) if names[k[i][2]].startswith('tf_argmax') and names[k[i+1][2]] == sys.argv[2] and 100e3 <= k[i+1][0]-k[i][1] <= 250e3]
a, b = hits[-1]
ev = []
for s, e, n, st in k:
    if a - 300e3 <= s <= b + 10e3: ev.append((s, e, 'K ' + names[n][:30], st))
for s, e, by, kd, st in db.execute("select start, end, bytes, copyKind, streamId from CUPTI_ACTIVITY_KIND_MEMCPY"):
    if a - 300e3 <= s <= b + 10e3: ev.append((s, e, f'M kind{kd} {by}B', st))
for s, e, by, st in db.execute("select start, end, bytes, streamId from CUPTI_ACTIVITY_KIND_MEMSET"):
    if a - 300e3 <= s <= b + 10e3: ev.append((s, e, f'S {by}B', st))
for s, e, st in db.execute("select start, end, streamId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
    if a - 300e3 <= s <= b + 10e3: ev.append((s, e, 'GRAPH', st))
for s, e, n, st in sorted(ev): print(f"{(s-a)/1e3:8.1f} {(e-s)/1e3:7.1f} st{st} {n}")
print("--- api")
for s, e, n in db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"):
    if a - 120e3 <= s <= b: print(f"{(s-a)/1e3:8.1f} {(e-s)/1e3:7.1f} {names[n]}")
print("--- sync")
for row in db.execute("select start, end, syncType, streamId, eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION order by start"):
    if a - 120e3 <= row[0] <= b: print(f"{(row[0]-a)/1e3:8.1f} {(row[1]-row[0])/1e3:7.1f} type{row[2]} st{row[3]} ev{row[4]}")
