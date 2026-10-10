"""Replay DFlash2 tree policies on recorded drafter candidates (round_bench.py DRAFT_TRACE=...): tokens a round.

Each record is one round's drafter output at a context length c (pending token included): the pending token full[c-1]
and 16 candidates a depth for full[c], full[c+1], ... Decoding is exact (greedy, or sampling with the target's keyed noise), so the tokens a
round accepts depend only on the tree and the reply the run produced. A policy is scored on the same rounds the
recording visited: mean accepted guesses + 1 (the target's own token).

  uv run python draft_sim.py ~/.cache/tmp-attn4/dtrace/base-*-t0.npz
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Round:
    c: int
    pending: int
    ids: np.ndarray          # (depths, 16) candidate ids
    unary: np.ndarray        # (depths, 16) drafter logits of the candidates
    projected: np.ndarray    # (depths, rank) selector projection
    truth: list[int]         # full[c : c+32] (copy chains can run past the drafter's depths)
    temp: float
    context: np.ndarray | None = field(default=None, repr=False)   # full[:c], for copy proposals
    noise: np.ndarray | None = field(default=None)


def load(path: str) -> tuple[list[Round], np.ndarray, np.ndarray]:
    z = np.load(path)
    full = np.concatenate([z["prompt"], z["out"]])
    temp = float(path.rsplit("-t", 1)[1].removesuffix(".npz"))
    rounds = []
    for c, pending, ids, fl in zip(z["ctx"], z["pending"], z["ids"], z["floats"]):
        c = int(c)
        assert full[c - 1] == pending, (path, c)             # the context length counts the pending token
        depths = ids.shape[0]
        rounds.append(Round(c, int(pending), ids.astype(np.int64), fl[:, :16].astype(np.float64),
                            fl[:, 16:].astype(np.float64), [int(t) for t in full[c:c + 32]], temp,
                            full[:c]))
    if "pred" in z:
        return rounds, z["pred"].astype(np.float64), z["succ"].astype(np.float64)
    books = _codebooks(path.rsplit("-", 3)[0] + "-codebooks.npy")
    return rounds, books[0], books[1]


_BOOKS: dict = {}


def _codebooks(path: str) -> np.ndarray:
    if path not in _BOOKS:
        _BOOKS[path] = np.load(path).astype(np.float64)
    return _BOOKS[path]


def add_noise(rounds: list[Round], seed: int = 1234) -> None:
    """The target's keyed Gumbel noise a candidate gets (what ``best_first`` weighs in when sampling)."""
    from tensorfold.engine.exact_sampling import uniform_rows

    for r in rounds:
        if r.temp > 0:
            positions = r.c + np.arange(r.ids.shape[0])        # as draft_decode passes len(context)
            r.noise = -np.log(-np.log(uniform_rows(seed, positions, r.ids)))


def accepted(tokens: list[int], parents: list[int], truth: list[int]) -> int:
    """Guesses the target keeps: follow the tree from the root along the true tokens."""
    children: dict[int, dict[int, int]] = {}
    for i, (t, p) in enumerate(zip(tokens, parents)):
        children.setdefault(p, {}).setdefault(t, i)
    node, n = -1, 0
    for t in truth:
        node = children.get(node, {}).get(t)
        if node is None:
            break
        n += 1
    return n


def best_first(r: Round, pred, succ, max_nodes: int, edge=0.6, noise_w=0.7, scale=1.5, branch=4, slope=0.0,
               depth_pen=0.0):
    """draft_tree.best_first with its policy constants as arguments (same pops, same ties)."""
    depth_count = r.ids.shape[0]
    temp = r.temp if r.temp > 0 else 1.0
    heap: list = []
    tokens: list[int] = []
    parents: list[int] = []
    successor = [succ[r.ids[d]] for d in range(depth_count)]

    def expand(token, depth, parent, path_score):
        e = successor[depth] @ (pred[token] * r.projected[depth]) if edge else 0.0
        values = (r.unary[depth] + edge * e) / temp
        if r.noise is not None:
            values = values + noise_w * r.noise[depth]
        values = values / (scale * (1 + slope * depth))
        values = values - values.max()
        logp = values - np.log(np.exp(values).sum()) - depth_pen * depth
        for i in np.argsort(-logp)[:branch]:
            if not np.isfinite(logp[i]):
                break
            heapq.heappush(heap, (path_score - float(logp[i]), parent, int(r.ids[depth, i]), depth))

    expand(r.pending, 0, -1, 0.0)
    while heap and len(tokens) < max_nodes:
        score, parent, token, depth = heapq.heappop(heap)
        me = len(tokens)
        tokens.append(token)
        parents.append(parent)
        if depth + 1 < depth_count:
            expand(token, depth + 1, me, score)
    return tokens, parents


def diagnose(rounds: list[Round], pred, succ) -> None:
    """Where acceptance is lost: the true token's rank among the candidates, by depth, given a right prefix."""
    depths = rounds[0].ids.shape[0]
    print("depth | right prefix rounds | true token is #1 (unary) | #1 with edge | in top 4 (edge) | in top 16")
    for d in range(depths):
        n = top1 = top1e = top4 = top16 = 0
        for r in rounds:
            if len(r.truth) <= d:
                continue
            n += 1
            row = list(r.ids[d])
            t = r.truth[d]
            if t not in row:
                continue
            top16 += 1
            prev = r.pending if d == 0 else r.truth[d - 1]
            e = succ[r.ids[d]] @ (pred[prev] * r.projected[d])
            v = r.unary[d] + 0.6 * e
            top1 += int(np.argmax(r.unary[d]) == row.index(t))
            order = list(np.argsort(-v))
            top1e += int(order[0] == row.index(t))
            top4 += int(row.index(t) in order[:4])
        print(f"{d + 1:5d} | {n:19d} | {top1 / n:24.1%} | {top1e / n:12.1%} | {top4 / n:15.1%} | {top16 / n:9.1%}")


def score(rounds, pred, succ, max_nodes=7, **kw) -> float:
    return 1 + float(np.mean([accepted(*best_first(r, pred, succ, max_nodes, **kw), r.truth) for r in rounds]))


def copy_proposal(context: np.ndarray, min_match: int, max_nodes: int) -> list[int]:
    """decode.CopyIndex's proposal: the longest continuation after an earlier occurrence of the last ``min_match``
    tokens (latest first), at least ``min_match`` long unless ``short`` (any length)."""
    n = min_match
    if len(context) < 2 * n:
        return []
    needle = context[-n:]
    best: list[int] = []
    ctx = context.tolist()
    nd = needle.tolist()
    for start in range(len(ctx) - n - 1, -1, -1):
        if ctx[start:start + n] == nd:
            cont = ctx[start + n:start + n + max_nodes]
            if len(cont) > len(best):
                best = cont
                if len(best) == max_nodes:
                    break
    return best


def with_copy(tokens, parents, chain: list[int], keep: int, max_nodes: int):
    """The draft tree's first ``keep`` pops, then the copy chain hung from the root (sharing nodes it matches), then the
    tree's next pops until ``max_nodes``."""
    tokens, parents = list(tokens), list(parents)
    out_t, out_p = tokens[:keep], parents[:keep]
    node = -1
    for t in chain:
        if len(out_t) >= max_nodes:
            break
        hit = next((i for i, (tt, pp) in enumerate(zip(out_t, out_p)) if tt == t and pp == node), None)
        if hit is None:
            out_t.append(t)
            out_p.append(node)
            hit = len(out_t) - 1
        node = hit
    remap = {i: i for i in range(keep)}
    for i in range(keep, len(tokens)):                  # the rest of the tree, while its parent is in
        if len(out_t) >= max_nodes:
            break
        p = parents[i]
        if p == -1 or p in remap:
            np_ = -1 if p == -1 else remap[p]
            hit = next((j for j, (tt, pp) in enumerate(zip(out_t, out_p)) if tt == tokens[i] and pp == np_), None)
            if hit is None:
                out_t.append(tokens[i])
                out_p.append(np_)
                hit = len(out_t) - 1
            remap[i] = hit
    return out_t, out_p


def copy_study(rounds, pred, succ, max_nodes=7) -> None:
    """Tokens a round when a copy chain from the context joins the draft tree, by match length and the tree's share."""
    trees = [best_first(r, pred, succ, 64) for r in rounds]
    base = 1 + np.mean([accepted(t[:max_nodes], p[:max_nodes], r.truth) for (t, p), r in zip(trees, rounds)])
    print(f"copy + tree, {max_nodes} nodes (tree alone {base:.3f}):")
    for m in (2, 3, 4, 6, 8):
        chains = [copy_proposal(r.context, m, 32) for r in rounds]
        hit = np.mean([bool(c) for c in chains])
        right = [accepted(c, list(range(-1, len(c) - 1)), r.truth) for c, r in zip(chains, rounds) if c]
        row = []
        for keep in (0, 2, 4, 5, 6):
            acc = [accepted(*with_copy(t, p, c, keep, max_nodes), r.truth) if c else
                   accepted(t[:max_nodes], p[:max_nodes], r.truth) for (t, p), c, r in zip(trees, chains, rounds)]
            row.append(f"tree {keep}: {1 + np.mean(acc):.3f}")
        print(f"  match {m}: chain in {hit:.0%} of rounds, its first token right {np.mean([a > 0 for a in right]) if right else 0:.0%}"
              f" | " + " | ".join(row))


def depth_sweep(rounds, pred, succ, max_nodes=7) -> list[tuple[float, dict]]:
    """Per-depth calibration around the default: the scale growing with depth, a log-probability penalty a depth."""
    out = []
    for slope in (-0.1, 0.0, 0.1, 0.2, 0.4):
        for depth_pen in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.4):
            for scale in (1.0, 1.5, 2.0):
                kw = dict(slope=slope, depth_pen=depth_pen, scale=scale)
                out.append((score(rounds, pred, succ, max_nodes, **kw), kw))
    return sorted(out, key=lambda t: -t[0])


def sweep(rounds, pred, succ, max_nodes=7) -> list[tuple[float, dict]]:
    """Every (edge, scale, branch) on a grid: tokens a round, best first."""
    out = []
    for edge in (0.0, 0.3, 0.6, 0.9, 1.2):
        for scale in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            for branch in (1, 2, 3, 4, 6):
                kw = dict(edge=edge, scale=scale, branch=branch)
                out.append((score(rounds, pred, succ, max_nodes, **kw), kw))
    return sorted(out, key=lambda t: -t[0])


def main(args: list[str]) -> None:
    flags = {a for a in args if a.startswith("--")}
    paths = [a for a in args if not a.startswith("--")]
    runs, pred, succ = {}, None, None
    for p in paths:
        rs, pred, succ = load(p)
        runs[p] = rs
    rounds = [r for rs in runs.values() for r in rs]
    add_noise(rounds)
    print(f"{len(rounds)} rounds from {len(paths)} runs")
    diagnose(rounds, pred, succ)
    print(f"default policy, 7 nodes: {score(rounds, pred, succ):.3f} tokens a round")
    if "--copy" in flags:
        copy_study(rounds, pred, succ)
    if "--sweep" not in flags:
        return
    names = sorted(runs)
    halves = [[r for p in names[i::2] for r in runs[p]] for i in (0, 1)]
    for i, (fit, test) in enumerate(((halves[0], halves[1]), (halves[1], halves[0]))):
        best = (depth_sweep if "--depth" in flags else sweep)(fit, pred, succ)
        top, kw = best[0]
        print(f"half {i}: best on fit {top:.3f} {kw}; on held-out {score(test, pred, succ, **kw):.3f} "
              f"vs default {score(test, pred, succ):.3f}")
        for t, k in best[:5]:
            print(f"   fit {t:.3f} {k}")


def nodes_study(paths: list[str]) -> None:
    """Tokens a round by the number of guesses (rows - 1), default policy."""
    rounds, pred, succ = [], None, None
    for p in paths:
        rs, pred, succ = load(p)
        rounds += rs
    add_noise(rounds)
    trees = [best_first(r, pred, succ, 64) for r in rounds]
    for n in (3, 5, 7, 9, 11, 13, 15, 19, 23, 31):
        acc = [accepted(t[:n], p[:n], r.truth) for (t, p), r in zip(trees, rounds)]
        print(f"rows {n + 1:2d}: {1 + np.mean(acc):.3f} tokens a round")


if __name__ == "__main__":
    if sys.argv[1] == "--nodes":
        nodes_study(sys.argv[2:])
    else:
        main(sys.argv[1:])
