import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
api = db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
a_name, b_name, lo, hi = sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
hits = [(k[i][1], k[i+1][0]) for i in range(len(k)-1) if names[k[i][2]].startswith(a_name) and names[k[i+1][2]].startswith(b_name) and lo <= (k[i+1][0]-k[i][1])/1e3 <= hi]
for a_end, b_start in hits[-2:]:
    print(f"--- gap {(b_start-a_end)/1e3:.0f} us")
    for s, e, n in api:
        if a_end - 30e3 <= s <= b_start: print(f"{(s-a_end)/1e3:8.1f} {(e-s)/1e3:7.1f} {names[n]}")
mc = db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
for a_end, b_start in hits[-2:]:
    print(f"--- memcpys in gap {(b_start-a_end)/1e3:.0f} us")
    for s, e, b, kd in mc:
        if a_end - 30e3 <= s <= b_start: print(f"{(s-a_end)/1e3:8.1f} {(e-s)/1e3:7.1f} {b} bytes kind {kd}")
