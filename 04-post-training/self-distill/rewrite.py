"""Rewrite the reasoning in teacher samples with Ornith's own, keeping every teacher action (best of N).

Teacher reasoning is far off Ornith's distribution (offpolicy.py: about -1.1 nats/token against -0.5 for its own),
its actions much less so. For each teacher turn, in order, Ornith samples N reasoning drafts from the context so far
(earlier turns already rewritten) and stops at </think>. Each draft is scored by the log-probability Ornith gives the
teacher's own tail after it (</think>, answer text, tool calls). The best draft replaces the teacher's reasoning.
The drafts are samples from Ornith, so picking among them by how well they explain the action approximates sampling
from p(reasoning | context, action) without the teacher reasoning ever being shown. The teacher's reasoning is kept
when no draft closes its </think> within --max-reason tokens, or when the best draft makes the tail less likely than
the teacher's reasoning did by more than --fallback nats per tail token.

Runs vLLM in-process (AsyncLLM): scoring needs prompt log-probs with the prefix cache on, which the OpenAI server
doesn't expose (vLLM skips the cache for prompt_logprobs requests by default, so every score would re-read the whole
context). Only tokens past the cached prefix get log-probs; the tail is always new text after a fresh draft.

    python rewrite.py --samples data/teacher.jsonl --model ~/ornith-bf16 --out data/teacher_rewritten.jsonl

Writes rewritten samples (harvest.py's format, without messages) to --out and one record per turn to --stats.
"""
import argparse
import asyncio
import hashlib
import itertools
import json
import time
from pathlib import Path

from tokenizers import Tokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

HERE = Path(__file__).resolve().parent
THINK_END = "</think>"


class Rewriter:
    def __init__(self, a):
        self.a = a
        self.tok = Tokenizer.from_file(str(Path(a.model).expanduser() / "tokenizer.json"))
        self.end_id = self.tok.token_to_id(THINK_END)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(Path(a.model).expanduser()), max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
            tensor_parallel_size=a.tp, enable_prefix_caching=True, max_logprobs=1, language_model_only=True,
            trust_remote_code=True, max_num_seqs=a.max_seqs))
        self.ids = itertools.count()

    async def run(self, ids: list, sp: SamplingParams):
        sp.output_kind = RequestOutputKind.FINAL_ONLY
        out = None
        async for out in self.engine.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=str(next(self.ids))):
            pass
        return out

    async def drafts(self, prefix: str) -> list:
        """Up to N reasoning drafts that close with </think>: (text, tokens, sum of their log-probs)."""
        ids = self.tok.encode(prefix, add_special_tokens=False).ids
        sp = SamplingParams(n=self.a.n, temperature=1.0, top_p=0.95, top_k=20, max_tokens=self.a.max_reason,
                            stop_token_ids=[self.end_id], logprobs=0, detokenize=False)
        res = await self.run(ids, sp)
        out = []
        for c in res.outputs:
            toks = list(c.token_ids)
            if not toks or toks[-1] != self.end_id:     # ran out of budget before closing its reasoning
                continue
            lps = [d[t].logprob for d, t in zip(c.logprobs[:-1], toks[:-1])]
            out.append((self.tok.decode(toks[:-1], skip_special_tokens=False), len(toks) - 1, sum(lps)))
        return out

    async def score(self, prefix: str, reasoning: str, tail: str, skip_cache=False) -> dict:
        """Log-probs of the reasoning and of the tail after prefix + reasoning."""
        text = prefix + reasoning + tail
        enc = self.tok.encode(text, add_special_tokens=False)
        cut_r, cut_t = len(prefix), len(prefix) + len(reasoning)
        first_r = next(i for i, (s, _) in enumerate(enc.offsets) if s >= cut_r)
        first_t = next(i for i, (s, _) in enumerate(enc.offsets) if s >= cut_t)
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False, skip_reading_prefix_cache=skip_cache)
        res = await self.run(enc.ids, sp)
        # with c tokens served from the prefix cache, prompt_logprobs covers only tokens c.. (entry i is token c + i,
        # entry 0 None); checked against an uncached run, the values are identical
        c = res.num_cached_tokens or 0
        if first_t <= c:
            return await self.score(prefix, reasoning, tail, skip_cache=True)
        plp = res.prompt_logprobs
        assert len(plp) == len(enc.ids) - c
        lp = lambda j: plp[j - c][enc.ids[j]].logprob
        ok_r = first_r > c
        return {"tail": sum(lp(j) for j in range(first_t, len(enc.ids))), "n_tail": len(enc.ids) - first_t,
                "reasoning": sum(lp(j) for j in range(first_r, first_t)) if ok_r else None, "n_reasoning": first_t - first_r}

    async def sample(self, r: dict, stats) -> dict:
        text, out, loss, cur = r["text"], "", [], 0
        for k, (a, b) in enumerate(r["loss"]):
            out += text[cur:a]
            cur = b
            span = text[a:b]
            e = span.find(THINK_END)
            rec = {"task": r["task"], "rollout": r["rollout"], "segment": r["segment"], "turn": k}
            if e < 0:                                    # no reasoning in this turn
                rec["chosen"] = "no-think"
            else:
                teacher, tail = span[:e], span[e:]
                drafts, base = await asyncio.gather(self.drafts(out), self.score(out, teacher, tail))
                scores = await asyncio.gather(*(self.score(out, d[0], tail) for d in drafts))
                rec.update(teacher_tail=base["tail"], n_tail=base["n_tail"], teacher_reasoning=base["reasoning"],
                           teacher_n_reasoning=base["n_reasoning"], drafts=len(drafts),
                           draft_tails=[s["tail"] for s in scores])
                best = max(range(len(drafts)), key=lambda i: scores[i]["tail"]) if drafts else None
                if best is None or (base["tail"] - scores[best]["tail"]) / base["n_tail"] > self.a.fallback:
                    rec["chosen"] = "teacher"
                else:
                    d = drafts[best]
                    rec.update(chosen="draft", tail=scores[best]["tail"], reasoning=d[2], n_reasoning=d[1])
                    span = d[0] + tail
            stats.write(json.dumps(rec) + "\n")
            stats.flush()
            loss.append([len(out), len(out) + len(span)])
            out += span
        out += text[cur:]
        enc = self.tok.encode(out, add_special_tokens=False)
        loss_tokens = sum(1 for s, _ in enc.offsets if any(lo <= s < hi for lo, hi in loss))
        return {**{k: v for k, v in r.items() if k != "messages"}, "text": out, "loss": loss, "tokens": len(enc.ids),
                "loss_tokens": loss_tokens}


async def main_async(a):
    rows = [json.loads(line) for line in open(a.samples)]
    i, k = map(int, a.shard.split("/"))
    # by rollout id, so re-harvesting with more runs keeps every finished sample in its shard
    rows = [r for r in rows if int(hashlib.md5(r["rollout"].encode()).hexdigest(), 16) % k == i][:a.limit or None]
    done = {(json.loads(line)["rollout"], json.loads(line)["segment"]) for line in open(a.out)} if a.out.exists() else set()
    todo = [r for r in rows if (r["rollout"], r["segment"]) not in done]
    print(f"{len(todo)} samples to rewrite ({len(rows) - len(todo)} done), {sum(len(r['loss']) for r in todo)} turns",
          flush=True)
    rw = Rewriter(a)
    sem = asyncio.Semaphore(a.concurrency)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.stats.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with a.out.open("a") as out, a.stats.open("a") as stats:
        async def one(r):
            async with sem:
                res = await rw.sample(r, stats)
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(f"{time.time() - t0:7.0f}s {r['task'][:28]:28} {len(r['loss']):3} turns  "
                  f"{r['loss_tokens']} -> {res['loss_tokens']} loss tokens", flush=True)
        await asyncio.gather(*(one(r) for r in todo))
    rw.engine.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", default="~/ornith-bf16")
    ap.add_argument("--out", type=Path, default=HERE / "data/teacher_rewritten.jsonl")
    ap.add_argument("--stats", type=Path, default=HERE / "results/rewrite.jsonl")
    ap.add_argument("-n", type=int, default=4, help="reasoning drafts per turn")
    ap.add_argument("--max-reason", type=int, default=8192, help="token budget per draft")
    ap.add_argument("--fallback", type=float, default=1.0,
                    help="keep the teacher's reasoning when the best draft costs the tail more than this, nats/token")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="I/K: this engine's share of the samples")
    ap.add_argument("--concurrency", type=int, default=64, help="samples in flight")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
