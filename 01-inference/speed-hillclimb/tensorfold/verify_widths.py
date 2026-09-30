import collections, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Projects/Personal/Research/agent-training/01-inference/speed-hillclimb/tensorfold"))
from tensorfold.cuda import capacity
capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(0.3 * capacity.GIB))
from tokenizers import Tokenizer
from tensorfold.cuda.markers import TemplateTokens
from tensorfold.cuda.server import ChatTemplate
from tensorfold.families.qwen3_5_moe.cuda import decode, engine as eng
from seg_ab import PROMPTS
sys.path.insert(0, os.path.expanduser("~/Projects/Personal/Research/agent-training/01-inference/speed-hillclimb/tensorfold/draftvocab"))
from ab import agent_prompts
model = os.path.expanduser("~/models/ornith/r2")
raw = Tokenizer.from_file(model + "/tokenizer.json"); tok = TemplateTokens(raw, ChatTemplate(Path(model)))
widths = collections.Counter()
orig = decode.mtp_decode
def wrap(*a, **k):
    r = orig(*a, **k); widths.update(r.widths); return r
eng_mod = sys.modules["tensorfold.families.qwen3_5_moe.cuda.decode"]; eng_mod.mtp_decode = wrap
e = eng.Qwen36Engine(model, context=32768, context_explicit=True)
for kind, text in PROMPTS + agent_prompts():
    ids = list(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)) if kind == "chat" else raw.encode(text, add_special_tokens=False).ids
    e.generate(ids, 1024, None, lambda t: False, stop_eos=False)
tot = sum(widths.values())
print("verify widths (rows: share of rounds):", {w: f"{100 * c / tot:.1f}%" for w, c in sorted(widths.items())})
