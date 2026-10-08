"""Does a 2-bit build of Qwen3.8-27B collapse on long reasoning? A small, heat-limited probe.

20 level-5 MATH-500 problems (fixed seed), thinking on, one sample each at the model card's sampling settings, a
16K-token cap. Records accuracy, reasoning length, and how often the model runs to the cap without answering: the
signature of the collapse Prism reports for untrained 2-bit builds.

    # EXL3, in-process (8 sequences at once)
    uv run --project 01-inference/envs/exl3 01-inference/qwen38-27b/reasoning_probe.py exl3 --revision 2.00bpw
    # Bonsai 2 (or anything) behind an OpenAI-compatible server, e.g. PrismML's llama-server
    uv run --project 01-inference/envs/prism-llama 01-inference/qwen38-27b/reasoning_probe.py server --label bonsai2-pq2
"""
import argparse
import gzip
import json
import random
import time
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify

OUT = Path(__file__).parent / "results" / "reasoning"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}  # Qwen3.8 card; Prism's eval used min_p 0


def problems(n, seed):
    rows = [r for r in load_dataset("HuggingFaceH4/MATH-500", split="test") if r["level"] == 5]
    return random.Random(seed).sample(rows, n)


def grade(row, text, finish):
    """finish: 'stop' (the model ended its turn), 'cap' (ran out of tokens) or 'deadline' (cut off by the timer).
    Only the text after </think> counts as the answer; a model still thinking at the cap has no answer."""
    answer = text.rsplit("</think>", 1)[1] if "</think>" in text else ""
    correct = bool(answer.strip()) and verify(parse("\\boxed{" + row["answer"] + "}"), parse(answer))
    outcome = ("correct" if correct else "runaway" if finish == "cap" else "cut off" if finish == "deadline"
               else "wrong")
    return {"id": row["unique_id"], "subject": row["subject"], "gold": row["answer"], "outcome": outcome,
            "finished_thinking": "</think>" in text, "answer_tail": answer.strip()[-200:]}


def run_exl3(args, rows):
    from huggingface_hub import snapshot_download
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler import ComboSampler

    config = Config.from_directory(snapshot_download("turboderp/Qwen3.8-27B-exl3", revision=args.revision))
    model = Model.from_config(config)
    # 8-bit KV: int4 is the kit's default, but 8 bits keeps the cache out of the comparison
    cache = Cache(model, max_num_tokens=args.slots * (args.cap + 512), layer_type=CacheLayer_quant,
                  k_bits=8, v_bits=8, max_batch_size=args.slots)
    model.load(progressbar=True)
    tok = Tokenizer.from_config(config)
    gen = Generator(model, cache, tok, max_batch_size=args.slots)
    stops = [tok.single_id("<|im_end|>"), tok.single_id("<|endoftext|>")]

    def prompt(text, think=True):
        return (f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
                + ("<think>\n" if think else "<think>\n\n</think>\n\n"))

    def ids(text):
        return tok.encode(text, encode_special_tokens=True)

    # The sanity check bench_exl3.py got wrong: its chat markers were encoded as plain text
    sanity = ""
    job = Job(input_ids=ids(prompt("A farmer has 17 sheep. All but 9 run away. How many are left? Answer with one "
                                   "number.", think=False)), max_new_tokens=16, stop_conditions=stops,
              sampler=ComboSampler(temperature=0.0), identifier="sanity")
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            sanity += r.get("text", "")
    print(f"== sanity (thinking off, greedy): {sanity!r}")

    sampler = ComboSampler(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"],
                           min_p=SAMPLING["min_p"])
    texts, info, jobs = {}, {}, {}
    for i, row in enumerate(rows):
        jobs[i] = Job(input_ids=ids(prompt(row["problem"] + SUFFIX)), max_new_tokens=args.cap, stop_conditions=stops,
                      sampler=sampler, seed=args.seed + i, identifier=i)
        gen.enqueue(jobs[i])
        texts[i] = ""
    t0, deadline, last = time.perf_counter(), time.perf_counter() + args.deadline_min * 60, 0
    while gen.num_remaining_jobs():
        if time.perf_counter() > deadline:
            for i, j in jobs.items():
                if i not in info:
                    gen.cancel(j)
                    info[i] = {"finish": "deadline", "tokens": len(tok.encode(texts[i])), "s": args.deadline_min * 60}
            break
        for r in gen.iterate():
            i = r["identifier"]
            texts[i] += r.get("text", "")
            if r.get("eos"):
                info[i] = {"finish": "cap" if r["eos_reason"] == "max_new_tokens" else "stop",
                           "tokens": r["new_tokens"], "s": time.perf_counter() - t0}
                print(f"   problem {i:>2} done: {r['new_tokens']:>5} tokens, {info[i]['finish']},"
                      f" {len(info)}/{len(rows)} after {time.perf_counter() - t0:.0f} s", flush=True)
        if time.perf_counter() - last > 60:
            last = time.perf_counter()
            print(f"   ... {time.perf_counter() - t0:.0f} s, {len(info)}/{len(rows)} finished")
    return [(texts[i], info[i]) for i in range(len(rows))], {"sanity": sanity}


def run_server(args, rows):
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    client = OpenAI(base_url=args.url, api_key="none")
    t0 = time.perf_counter()

    def one(i):
        remaining = args.deadline_min * 60 - (time.perf_counter() - t0)
        try:
            r = client.chat.completions.create(
                model="local", messages=[{"role": "user", "content": rows[i]["problem"] + SUFFIX}],
                max_tokens=args.cap, temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], seed=args.seed + i,
                extra_body={"top_k": SAMPLING["top_k"], "min_p": SAMPLING["min_p"],
                            "chat_template_kwargs": args.template_kwargs,
                            **({"draft": False} if args.no_draft else {})}, timeout=min(max(remaining, 1), 86400))
        except Exception as e:  # the timer ran out
            return "", "", {"finish": "deadline", "tokens": None, "s": time.perf_counter() - t0, "error": str(e)[:200]}
        c = r.choices[0]
        reasoning = getattr(c.message, "reasoning_content", None) or ""
        finish = "cap" if c.finish_reason == "length" else "stop"
        print(f"   problem {i:>2} done: {r.usage.completion_tokens:>5} tokens, {finish}, after"
              f" {time.perf_counter() - t0:.0f} s", flush=True)
        return reasoning, c.message.content or "", {"finish": finish, "tokens": r.usage.completion_tokens,
                                                    "s": time.perf_counter() - t0}

    with ThreadPoolExecutor(args.slots) as pool:
        out = list(pool.map(one, range(len(rows))))
    sanity = client.chat.completions.create(
        model="local", max_tokens=2048, temperature=0.0,
        messages=[{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many are left? "
                                              "Answer with one number."}]).choices[0].message.content
    print(f"== sanity (thinking on, greedy): {sanity!r}")
    return [((reasoning + "</think>" if reasoning else "") + content, info) for reasoning, content, info in out], \
        {"sanity": sanity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["exl3", "server"])
    parser.add_argument("--revision", default="2.00bpw", help="EXL3 branch")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--label", default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--cap", type=int, default=16384, help="max new tokens, thinking included")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1,
                        help="attempts per problem, each with its own sampling seed; one sample swings a 20-problem "
                             "score by 2-3 problems between runs")
    parser.add_argument("--out", type=Path, default=OUT, help="results directory")
    parser.add_argument("--template-kwargs", type=json.loads, default={},
                        help='server backend: chat template variables, e.g. \'{"reasoning_effort": "medium"}\'')
    parser.add_argument("--no-draft", action="store_true",
                        help='send "draft": false (TensorFold: plain one-token decoding, no draft model)')
    parser.add_argument("--deadline-min", type=float, default=1e9, help="optional heat guard: stop the run after this long")
    args = parser.parse_args()
    label = args.label or f"exl3-{args.revision}"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = [row for row in problems(args.n, args.seed) for _ in range(args.samples)]  # attempt i uses seed + i
    t0 = time.perf_counter()
    outputs, extra = (run_exl3 if args.backend == "exl3" else run_server)(args, rows)
    wall = time.perf_counter() - t0

    graded = [grade(row, text, info["finish"]) | info | {"sample": i % args.samples}
              for i, (row, (text, info)) in enumerate(zip(rows, outputs))]
    counts = {k: sum(g["outcome"] == k for g in graded) for k in ("correct", "wrong", "runaway", "cut off")}
    lengths = sorted(g["tokens"] for g in graded if g["tokens"])
    summary = {"label": label, "n": len(rows), "problems": args.n, "samples": args.samples, "cap": args.cap,
               "sampling": SAMPLING,
               "template_kwargs": args.template_kwargs, "draft": not args.no_draft, "wall_s": wall, **extra,
               **counts, "accuracy": counts["correct"] / len(rows),
               "median_tokens": lengths[len(lengths) // 2] if lengths else None,
               "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
               "total_tokens": sum(lengths), "per_problem": graded}
    (out / f"{label}.json").write_text(json.dumps(summary, indent=1))
    with gzip.open(out / f"{label}.traces.jsonl.gz", "wt") as f:
        for row, (text, info) in zip(rows, outputs):
            f.write(json.dumps({"id": row["unique_id"], "problem": row["problem"], "gold": row["answer"],
                                "output": text, **info}) + "\n")  # attempts in order: problem-major, sample-minor
    print(f"== {label}: {counts} of {len(rows)}; accuracy {summary['accuracy']:.0%};"
          f" median {summary['median_tokens']} tokens; {summary['total_tokens']} tokens in {wall / 60:.1f} min")
    for g in graded:
        print(f"   {g['id']:<40} {g['outcome']:<8} {g['tokens']!s:>6} tok  gold {g['gold']!r:<20}"
              f" got {g['answer_tail'][-60:]!r}")


if __name__ == "__main__":
    main()
