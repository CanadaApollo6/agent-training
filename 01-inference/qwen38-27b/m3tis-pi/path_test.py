"""Which code path breaks BF16 Qwen3.8-27B in HF: next-token loss on the first 1024 tokens of kl.jsonl text 0."""
import json, sys
mode = sys.argv[1]
if mode == "nofla":
    sys.modules["fla"] = None
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
sys.path.insert(0, "/home/ubuntu/pod")
from kl_q27 import dict_args
tok = AutoTokenizer.from_pretrained("/home/ubuntu/qwen27")
row = json.loads(open("/home/ubuntu/pod/kl.jsonl").readline())
ids = tok.apply_chat_template(dict_args(row["messages"]), tools=row["tools"], tokenize=True, return_dict=False,
                              enable_thinking=row["thinking"])
if hasattr(ids, "keys"):
    ids = ids["input_ids"]
ids = list(ids)[:int(sys.argv[2]) if len(sys.argv) > 2 else 1024]
kw = {"attn_implementation": "eager"} if mode == "eager" else {}
m = Qwen3_5ForConditionalGeneration.from_pretrained("/home/ubuntu/qwen27", dtype=torch.bfloat16, device_map="auto",
                                                    **kw).eval()
x = torch.tensor([ids]).to(m.device)
with torch.no_grad():
    lp = torch.log_softmax(m(input_ids=x).logits[0, :-1].float(), -1)
nll = -lp.gather(-1, x[0, 1:, None].to(lp.device))[:, 0].cpu()
print(mode, len(ids), "mean NLL %.3f" % nll.mean(), "first 200 %.3f" % nll[:200].mean(),
      "at 59/71/95:", [round(nll[p].item(), 2) for p in (59, 71, 95)],
      "top at 59:", repr(tok.decode([lp[59].argmax().item()])))
