"""Renders one row through SpecForge's Nanbeige parser and prints the token spans it trains on, so the template and
loss mask can be checked by eye: every assistant turn (thinking, answer, tool calls, the closing <|im_end|>) should
be marked, and no system, user or tool text.

    python check_mask.py <tokenizer dir> <rows.jsonl> [row index]
"""
import json
import sys

from transformers import AutoTokenizer

from specforge.data.parse import ThinkingParser
from specforge.data.template import TEMPLATE_REGISTRY

tok = AutoTokenizer.from_pretrained(sys.argv[1])
row = [json.loads(l) for l in open(sys.argv[2])][int(sys.argv[3]) if len(sys.argv) > 3 else 0]
ids, mask = ThinkingParser(tok, TEMPLATE_REGISTRY.get("nanbeige-thinking")).parse(
    row["conversations"], 8192, tool=row.get("tools") or [])
ids, mask = ids.view(-1).tolist(), mask.view(-1).tolist()
print(f"{len(ids)} tokens, {sum(mask)} trained; first ids {ids[:4]}")
spans, start = [], None
for i, m in enumerate(mask + [0]):
    if m and start is None:
        start = i
    elif not m and start is not None:
        spans.append((start, i))
        start = None
for a, b in spans[:4]:
    print(f"--- trained [{a}:{b}] after ...{tok.decode(ids[max(0, a - 6):a])!r}")
    print(repr(tok.decode(ids[a:min(b, a + 40)])), "...", repr(tok.decode(ids[max(a, b - 12):b])))
print(f"{len(spans)} spans")
