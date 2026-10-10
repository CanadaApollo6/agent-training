import re, collections, statistics as st
DONE = re.compile(r"rollout done: id=\S+ task=(\d+) reward=(\S+) turns=(\d+) stop=(\S+)")
BAD = {"ProviderError", "HarnessError", "SandboxError"}
tasks = [l.split()[0] for l in open("/home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals/tasks.txt") if l.strip()]
src = {"base": "/home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals/logs/tb2-pilot-base-a{}-pi.log",
       "pilot": "/home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals/logs/tb2-pilot-rl-a{}-pi.log",
       "cont": "/tmp/he-old/06-agents/harness-evals/logs/tb2-cont-rl-a{}-pi.log"}
per = {}
for k, pat in src.items():
    solved = usable = total = 0; stops = collections.Counter(); pt = collections.defaultdict(list)
    for a in (1, 2, 3):
        for m in DONE.finditer(open(pat.format(a)).read()):
            t, r, turns, stop = int(m[1]), float(m[2]), int(m[3]), m[4]; total += 1; stops[stop] += 1
            if stop in BAD: continue
            usable += 1; solved += r >= 1; pt[tasks[t]].append(r >= 1)
    per[k] = {t: sum(v) / len(v) for t, v in pt.items()}
    print(f"{k:6s} solved {solved}/{usable} usable ({100*solved/usable:.0f}%), {total} logged; stops {dict(stops)}")
common = set.intersection(*[set(v) for v in per.values()])
for k in per: print(f"{k:6s} per-task mean over {len(common)} common tasks: {100*st.mean(per[k][t] for t in common):.1f}%")
for a, b in (("base", "cont"), ("pilot", "cont")):
    d = {t: per[b][t] - per[a][t] for t in common}
    print(f"{a} -> {b}: up {sorted(t for t in d if d[t] > 0)}, down {sorted(t for t in d if d[t] < 0)}")
