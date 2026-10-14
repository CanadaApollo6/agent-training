"""A long code prompt (about 14.7K tokens) for decode speed at long context: two CUDA sources pasted into one
chat-templated user turn. Usage: make_long_prompt.py MODEL_DIR SOURCE_TEXT OUT_JSON (long_prompt.json is its output
for TensorFold lfm2 d94dc787's lfm2.cu and window_attention.cu)."""

import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
src = open(sys.argv[2]).read()
msgs = [{"role": "user", "content": "Here is a CUDA source file:\n\n```cuda\n" + src + "\n```\n\n"
         "Write a Python reference implementation of tf_lfm2_attn using numpy, with a short test."}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok.encode(text, add_special_tokens=False)
print(len(ids))
json.dump({"long": list(ids)}, open(sys.argv[3], "w"))
