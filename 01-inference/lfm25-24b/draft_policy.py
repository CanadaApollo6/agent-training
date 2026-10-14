"""Drafting policies replayed on greedy outputs with measured window costs (ms by rows): speedup over plain decode.
Usage: draft_policy.py prompts.json long_prompt.json greedy.json greedy_long.json (greedy runs saved by tf-lfm2-cuda --save)."""
import json, sys
short = [3.08, 3.93, 4.40, 4.77, 5.30, 5.64, 5.96, 6.23, 7.05, 7.47, 7.82, 8.03, 8.38, 8.60, 9.04, 9.22]
long_ = [3.47, 4.55, 5.22, 6.02, 6.59, 7.11, 7.76, 8.57, 9.61, 10.02, 11.52, 11.73, 12.12, 12.45, 12.84, 14.10]
plain = {"short": 3.03, "long": 3.50}

def match(ctx, maxn=64):
    """Latest earlier occurrence of the longest suffix (up to maxn): (start of continuation, match length)."""
    best = (None, 0)
    n = len(ctx)
    last = ctx[-1]
    for s in range(n - 2, -1, -1):
        if ctx[s] != last: continue
        m = 1
        while m < maxn and s - m >= 0 and ctx[s - m] == ctx[n - 1 - m]: m += 1
        if m > best[1]: best = (s + 1, m)
        if m >= maxn: break
    return best

def run(prompt, out, cost, policy):
    ctx = list(prompt) + [out[0]]
    i, t = 1, 0.0
    k_adapt = 2
    while i < len(out):
        at, m = match(ctx)
        k = policy(m, k_adapt) if at is not None else 0
        d = ctx[at:at + k] if k else []
        a = 0
        while a < len(d) and i + a < len(out) and d[a] == out[i + a]: a += 1
        t += cost[len(d)]
        if d: k_adapt = min(15, k_adapt * 2) if a == len(d) else max(1, a)
        take = min(a + 1, len(out) - i)
        ctx += out[i:i + take]; i += take
    return t

prompts = json.load(open(sys.argv[1])); prompts.update(json.load(open(sys.argv[2])))
outs = json.load(open(sys.argv[3]))["results"]; outs.update(json.load(open(sys.argv[4]))["results"])
pols = {f"fixed K={K} m>={M}": (lambda K, M: lambda m, ka: K if m >= M else 0)(K, M) for K in (2, 4, 8) for M in (1, 2, 3, 4, 6, 8, 12)}
pols.update({f"adapt m>={M} cap{C}": (lambda M, C: lambda m, ka: min(ka, C) if m >= M else 0)(M, C) for M in (2, 3, 4, 6, 8) for C in (4, 8, 15)})
res = {}
for name, out in outs.items():
    out = out["tokens"]
    cost = long_ if name == "long" else short
    base = (len(out) - 1) * plain["long" if name == "long" else "short"]
    for p, f in pols.items(): res.setdefault(p, {})[name] = base / run(prompts[name], out, cost, f)
best = sorted(res.items(), key=lambda kv: -sum(kv[1].values()))
for p, r in best[:12]: print(f"{p:22s} " + "  ".join(f"{n} {v:.2f}x" for n, v in r.items()))
