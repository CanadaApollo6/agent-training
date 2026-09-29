"""How many of the model's own output tokens a draft vocabulary of N ids covers: the ids ranked by frequency in
Ornith's agent traces (terminal-bench, 35B and 9B, split by task), by BPE id, and TensorFold's default 79,591 ids.

A draft can only be right when its token is in the draft vocabulary, so coverage bounds the acceptance kept."""

import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
TRACES = Path.home() / "Projects/Personal/Research/agent-training/06-agents/harness-evals/outputs/primeintellect"
RESULTS = HERE.parent / "results"
tok = Tokenizer.from_file(str(Path.home() / "models/ornith/r2/tokenizer.json"))


def texts(path):
    """(task name, assistant text) pairs: reasoning, reply and tool-call arguments as the model wrote them."""
    for line in open(path):
        r = json.loads(line)
        for tr in r["traces"]:
            name = tr["task"]["data"]["name"]
            for n in tr["nodes"]:
                m = n["message"]
                if m["role"] != "assistant":
                    continue
                parts = [m.get("reasoning_content") or "", m.get("content") or ""]
                for c in m.get("tool_calls") or []:
                    try:
                        args = json.loads(c["arguments"])
                        parts += [v if isinstance(v, str) else json.dumps(v) for v in args.values()]
                    except (ValueError, AttributeError):
                        parts.append(c["arguments"])
                yield name, "\n".join(p for p in parts if p)


def ids(text):
    return tok.encode(text, add_special_tokens=False).ids


by_task = collections.defaultdict(list)
for f in sorted(glob.glob(str(TRACES / "terminal-bench-2--ornith*/traces.jsonl"))):
    for name, t in texts(f):
        by_task[name].extend(ids(t))
tasks = sorted(by_task)
train = collections.Counter()
held = []
for i, name in enumerate(tasks):                  # alternate tasks: rank on half, measure on the other half
    (train.update(by_task[name]) if i % 2 == 0 else held.extend(by_task[name]))
greedy = []
for f in ("greedy-3090-r2-tail.json", "greedy-3090-new.json"):
    for g in json.load(open(RESULTS / f)):
        greedy.append(ids(g["text"]))
vocab = tok.get_vocab_size()
default = np.unique(np.loadtxt(Path(sys.argv[1]), dtype=np.int64).reshape(-1)) if len(sys.argv) > 1 else None
print(f"{len(tasks)} tasks; ranked on {sum(train.values())} tokens ({len(train)} distinct), held-out traces "
      f"{len(held)} tokens, benchmark greedy texts {sum(map(len, greedy))} tokens; vocab {vocab}")

# rank: trace frequency, then BPE id (earlier merges are more frequent in the tokenizer's corpus)
seen = sorted(train, key=lambda t: (-train[t], t))
rest = [t for t in range(vocab) if t not in train]
ranked = np.array(seen + rest, dtype=np.int64)
np.save(HERE / "ranked.npy", ranked)
np.save(HERE / "seen.npy", len(seen))
json.dump(tasks[1::2], open(HERE / "held_out_tasks.json", "w"), indent=1)


def cover(sub, toks):
    s = np.zeros(vocab + 1024, bool)
    s[sub] = True
    a = np.asarray(toks)
    return s[a].mean()


rows = []
for n in (4096, 8192, 16384, 24576, 32768, 49152, 65536):
    f = cover(ranked[:n], held)
    b = cover(np.arange(n), held)
    g = [cover(ranked[:n], x) for x in greedy]
    print(f"N {n:6d}: held-out traces {100 * f:6.2f}% (ids<N {100 * b:6.2f}%)  greedy texts "
          + " ".join(f"{100 * x:6.2f}" for x in g))
if default is not None:
    # the A/B's order: trace-seen ids by frequency, then the default list's other ids by id
    order = np.array(seen + [t for t in default.tolist() if t not in train], dtype=np.int64)
    for n in (16384, 24576, 32768, 49152):
        print(f"seen+default top {n:6d}: held-out traces {100 * cover(order[:n], held):6.2f}%  greedy texts "
              + " ".join(f"{100 * cover(order[:n], x):6.2f}" for x in greedy))
    print(f"default {len(default)}: held-out traces {100 * cover(default, held):6.2f}%  greedy texts "
          + " ".join(f"{100 * cover(default, x):6.2f}" for x in greedy))
