"""bf16 reference check: perplexity of Nanbeige under HF transformers on the same wiki2 rows exllamav3's model_diff
scores (same exllamav3 tokenization), with and without a BOS token in front."""
import math, sys, torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, "/home/ubuntu/qkit")
from exllamav3 import Config, Tokenizer
import model_diff
cfg = Config.from_directory("/home/ubuntu/nanbeige")
tok = Tokenizer.from_config(cfg)
ids = model_diff.get_test_tokens(tok, 8, 2048, 2048)
m = AutoModelForCausalLM.from_pretrained("/home/ubuntu/nanbeige", torch_dtype=torch.bfloat16, device_map="cuda:0")
m.eval()
for bos in (False, True):
    nll = n = 0
    for r in ids:
        x = r.view(1, -1).cuda()
        if bos:
            x = torch.cat([torch.tensor([[166100]], device="cuda"), x[:, :-1]], 1)
        with torch.no_grad():
            lg = m(x).logits.float()
        s = 1 if bos else 0
        lp = torch.log_softmax(lg[0, s:-1], -1).gather(-1, x[0, s + 1:].unsqueeze(-1))
        nll -= lp.sum().item(); n += lp.numel()
    print(f"HF bf16 wiki2 8 rows, bos={bos}: ppl {math.exp(nll / n):.3f}", flush=True)
