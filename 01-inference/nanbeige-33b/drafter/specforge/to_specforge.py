"""gen_responses.py output -> SpecForge online training rows: {"id", "conversations": [..., assistant turn with
reasoning_content]}. Responses cut at the token cap keep their text; the template then closes them with a few
tokens the model didn't write, a handful per sample.

    python3 to_specforge.py out.jsonl gen-*.jsonl
"""
import json
import sys

dst, srcs = sys.argv[1], sys.argv[2:]
n = cut = 0
seen = set()
with open(dst, "w") as out:
    for src in srcs:
        for line in open(src):
            r = json.loads(line)
            if r["id"] in seen or not (r["reasoning"] or r["content"]):
                continue
            seen.add(r["id"])
            cut += r["finish"] == "length"
            turn = {"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]}
            out.write(json.dumps({"id": r["id"], "conversations": r["messages"] + [turn]}) + "\n")
            n += 1
print(f"{n} rows ({cut} cut at the cap) -> {dst}")
