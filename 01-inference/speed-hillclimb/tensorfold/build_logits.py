"""Teacher-forced next-token logits of one build, for comparing two TensorFold versions on the same weights.

    python build_logits.py MODEL_DIR GREEDY_JSON OUT.pt

For each greedy reply in GREEDY_JSON (greedy.py's output, from a reference build), the plain-text prompt plus the
reply's first k characters is prefilled serially (no drafts) at 12 cut points, and the last row's full-vocabulary
logits are kept. Two builds' files then give KL and top-1 agreement on identical inputs (compare_logits.py).
"""
import json
import sys

import torch
from tokenizers import Tokenizer

from tensorfold.families.qwen3_5.cuda.decode import State, prefill_stops
from tensorfold.families.qwen3_5.cuda.forward import _mm
from tensorfold.families.qwen3_5_moe.cuda.weights import load

model, greedy, out = sys.argv[1:4]
tok = Tokenizer.from_file(f"{model}/tokenizer.json")
w = load(model)
rows = []
with torch.no_grad():
    for item in json.load(open(greedy)):
        reply = item["text"].split("\n<<>>\n", 1)[-1]
        for cut in torch.linspace(0, len(reply) * 0.6, 12).long().tolist():
            ids = tok.encode(item["prompt"] + reply[:cut], add_special_tokens=False).ids
            normed = prefill_stops(w, ids, State(w))
            rows.append(_mm(normed[-1:], w.head).float().cpu()[0])
torch.save(torch.stack(rows), out)
print("saved", len(rows), "rows")
