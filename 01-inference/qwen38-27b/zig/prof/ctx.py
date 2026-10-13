import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
cs = db.execute("select timestamp, tag, contextId, globalPid from GPU_CONTEXT_SWITCH_EVENTS order by timestamp").fetchall()
print("contexts", collections.Counter((c[2], c[3]) for c in cs))
hits = [(k[i][1], k[i+1][0]) for i in range(len(k)-1) if names[k[i][2]].startswith('tf_argmax') and names[k[i+1][2]] == 'tf_upload' and k[i+1][0]-k[i][1] > 100e3]
nhit = 0
for a, b in hits:
    inside = [c for c in cs if a <= c[0] <= b]
    nhit += bool(inside)
print(f"slow commit gaps {len(hits)}, with a context switch inside {nhit}")
a, b = hits[-1]
for t, tag, ctx, pid in cs:
    if a - 50e3 <= t <= b + 20e3: print(f"{(t-a)/1e3:8.1f} tag{tag} ctx{ctx} pid{pid}")
fast = [(k[i][1], k[i+1][0]) for i in range(len(k)-1) if names[k[i][2]].startswith('tf_argmax') and names[k[i+1][2]] == 'tf_upload' and k[i+1][0]-k[i][1] < 40e3]
print(f"fast commit gaps {len(fast)}, with a switch inside {sum(1 for a, b in fast if any(a <= c[0] <= b for c in cs))}")
