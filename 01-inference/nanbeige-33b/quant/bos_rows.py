"""BOS variants of the KL rows: <|im_start|> (Nanbeige's BOS, 166100) at position 0, row shifted right by one."""
import sys, torch
sys.path.insert(0, "/home/ubuntu/qkit")
from safetensors.torch import load_file, save_file
from exllamav3 import Config, Tokenizer
import model_diff
BOS = 166100
def bos(x):
    return torch.cat([torch.full((x.shape[0], 1), BOS, dtype=x.dtype), x[:, :-1]], 1).contiguous()
a = load_file("/home/ubuntu/rows/kl-agent.safetensors")["input_ids"]
save_file({"input_ids": bos(a)}, "/home/ubuntu/rows/kl-agentbos.safetensors")
tok = Tokenizer.from_config(Config.from_directory("/home/ubuntu/nanbeige"))
w = model_diff.get_test_tokens(tok, 100, 2048, 2048).to(torch.long)
save_file({"input_ids": bos(w)}, "/home/ubuntu/rows/kl-wikibos.safetensors")
print("ok", a.shape, w.shape, int((a[:, 0] == BOS).sum()), "agent rows already start with BOS")
