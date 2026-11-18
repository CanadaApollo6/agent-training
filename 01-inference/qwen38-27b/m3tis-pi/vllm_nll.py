"""Reference next-token loss from vLLM (BF16, TP4) on the same token ids as path_test.py: kl.jsonl text 0, 6144 tokens."""
import json, sys
sys.path.insert(0, "/home/ubuntu/pod")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from kl_q27 import dict_args


def main():
    tok = AutoTokenizer.from_pretrained("/home/ubuntu/qwen27")
    row = json.loads(open("/home/ubuntu/pod/kl.jsonl").readline())
    ids = tok.apply_chat_template(dict_args(row["messages"]), tools=row["tools"], tokenize=True, return_dict=False,
                                  enable_thinking=row["thinking"])
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = list(ids)[:6144]
    llm = LLM("/home/ubuntu/qwen27", tensor_parallel_size=4, max_model_len=8192, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0}, gpu_memory_utilization=0.85)
    out = llm.generate(TokensPrompt(prompt_token_ids=ids), SamplingParams(max_tokens=1, prompt_logprobs=0))[0]
    nll = [-d[t].logprob for d, t in zip(out.prompt_logprobs[1:], ids[1:])]
    m = lambda xs: sum(xs) / len(xs)
    print("vllm", len(ids), "mean NLL %.3f" % m(nll), "first 200 %.3f" % m(nll[:200]), "first 1023 %.3f" % m(nll[:1023]),
          "1023-2047 %.3f" % m(nll[1023:2047]), "at 59/71/95:", [round(nll[p], 2) for p in (59, 71, 95)])


if __name__ == "__main__":
    main()
