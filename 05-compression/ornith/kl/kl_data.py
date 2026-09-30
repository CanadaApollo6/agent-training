"""Token sequences for the KL harness: Q8's recorded Terminal-Bench conversations, as the server rendered them.

Each conversation is rendered with the model's chat template, the way TensorFold does for a request (earlier turns'
thinking kept), and tokenized once. The mask marks the positions whose next token the model wrote itself: its
thinking, answers and tool calls. That's where a quantized build's drift matters, not in re-reading tool output.

Run in the TensorFold env (it renders with TensorFold's template code):

    cd 01-inference/envs/tensorfold && uv run python ../../../05-compression/ornith/kl/kl_data.py

Writes kl-data.pt: a list of {"task", "attempt", "ids" int32 [T], "mask" bool [T]} (mask[t]: token t+1 is the model's).
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "06-agents/harness-evals"))
from length_replay import MODEL_DIR, openai_message, openai_tools, traces  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="q8")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", default=str(Path(__file__).parent / "kl-data.pt"))
    args = ap.parse_args()

    from tokenizers import Tokenizer

    import tensorfold.cuda.server as S
    from tensorfold.server.messages import normalize_messages

    template, tok = S.ChatTemplate(MODEL_DIR), Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    def render(msgs, tools, gen):
        m = S._normalize_tool_call_arguments(normalize_messages([openai_message(x) for x in msgs],
                                                                late_system=template.late_system))
        return template.template.render(**template.specials, messages=m, tools=openai_tools(tools),
                                        add_generation_prompt=gen, enable_thinking=True)

    rows, skipped = [], 0
    for attempt, task, t in traces(args.build):
        msgs = [n["message"] for n in t["nodes"]]
        if any(isinstance(m.get("content"), list) and any(p.get("type") != "text" for p in m["content"]) for m in msgs):
            skipped += 1
            continue
        full = render(msgs, t["tools"], False)
        spans = []                                   # character spans of the model's own turns
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            start = render(msgs[:j], t["tools"], True)
            end = render(msgs[:j + 1], t["tools"], False)
            assert full.startswith(start) and full.startswith(end), (task, attempt, j)
            spans.append((len(start), len(end)))
        enc = tok.encode(full, add_special_tokens=False)
        ids = torch.tensor(enc.ids[:args.max_tokens], dtype=torch.int32)
        own = torch.zeros(len(enc.ids), dtype=torch.bool)
        starts = torch.tensor([o[0] for o in enc.offsets])
        for a, b in spans:
            own |= (starts >= a) & (starts < b)
        own = own[:len(ids)]
        mask = torch.zeros(len(ids), dtype=torch.bool)
        mask[:-1] = own[1:]
        rows.append({"task": task, "attempt": attempt, "ids": ids, "mask": mask})
    torch.save(rows, args.out)
    n, m = sum(len(r["ids"]) for r in rows), sum(int(r["mask"].sum()) for r in rows)
    print(f"{len(rows)} conversations ({skipped} with images skipped), {n} tokens, {m} scored positions -> {args.out}")


if __name__ == "__main__":
    main()
