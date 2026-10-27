"""OpenAI-style chat server for an EXL3 Nanbeige4.3-33B quant, built to match the BF16 TB2 pilot's vLLM 0.32 setup
(06-agents/harness-evals/serve_nanbeige_pod.sh) everywhere except the weights, so a pilot run compares quants only:

- Prompts are rendered by the model's own chat template (HF tokenizer, earlier thinking kept) and tokenized by the HF
  tokenizer, as vLLM does. Tool-call arguments arrive as JSON strings and go to the template as mappings.
- Sampling defaults to the card's agent settings, T 1.0 and top_p 0.95; a request's own values win.
- Output is split like vLLM's `qwen3` reasoning parser (everything before </think> is reasoning) and `qwen3_coder`
  tool parser (<tool_call><function=NAME><parameter=P>value</parameter>...), with values typed by the tool schema.
- Image parts get vLLM's 400 ("... is not a multimodal model"), which ends the run the same way it did for BF16.
- 262,144-token context per request. The KV cache is 8-bit (our KL test: no measurable drift), shared by all requests.

Streaming sends reasoning as it is generated, then the answer and tool calls in one event at the end (SSE comment
pings keep the connection alive meanwhile).

    python serve_exl3.py --model ~/q/3.5-nb --name nanbeige-exl3-a1 [--name ...] [--port 8000] [--cache-tokens N]
"""
import argparse
import ast
import asyncio
import json
import re
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from exllamav3 import AsyncGenerator, AsyncJob, Cache, CacheLayer_quant, ComboSampler, Config, Model, Tokenizer

CONTEXT = 262144
STOP_IDS = [166101, 166102]  # <|im_end|>, <|endoftext|>
THINK_END = "</think>"
TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)

app = FastAPI()
S = {}  # model, tokenizers, generator, names


def error(status, message, kind="BadRequestError"):
    return JSONResponse({"error": {"message": message, "type": kind, "param": None, "code": status}}, status)


def text_of(content):
    """Message content as one string; None if it holds media (the caller refuses those)."""
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif p.get("type") == "text":
            parts.append(p.get("text", ""))
        else:
            return None
    return "\n".join(parts)


def prepare(messages):
    out = []
    for m in messages:
        m = dict(m)
        text = text_of(m.get("content"))
        if text is None:
            raise ValueError("media")
        m["content"] = text
        if m.get("role") == "assistant":
            m["reasoning_content"] = m.get("reasoning_content") or m.get("reasoning")
            calls = []
            for c in m.get("tool_calls") or []:
                c = json.loads(json.dumps(c))
                args = c.get("function", {}).get("arguments")
                if isinstance(args, str):
                    try:
                        c["function"]["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        c["function"]["arguments"] = {}
                calls.append(c)
            if calls:
                m["tool_calls"] = calls
        out.append(m)
    return out


def typed(value, schema):
    """A parameter's text as the type its tool schema declares (qwen3_coder's conversion)."""
    kind = (schema or {}).get("type", "string")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if value.lower() == "null" and kind != "string":
        return None
    try:
        if kind in ("string", "str", "text"):
            return value
        if kind in ("integer", "int"):
            return int(value)
        if kind in ("number", "float"):
            f = float(value)
            return int(f) if f.is_integer() and "." not in value else f
        if kind in ("boolean", "bool"):
            return value.strip().lower() == "true"
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def split_output(text, tools):
    """(reasoning, content, tool_calls) from the raw completion."""
    reasoning, sep, rest = text.partition(THINK_END)
    if not sep:  # no end of thinking: all of it is reasoning, as vLLM streams it
        return text.removeprefix("<think>").strip("\n"), "", []
    reasoning = reasoning.removeprefix("<think>").strip("\n")
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {})
               for t in tools or [] if t.get("type", "function") == "function"}
    calls = []
    first = rest.find("<tool_call>") if tools else -1
    if first >= 0:
        for block in TOOL_CALL.findall(rest[first:]):
            f = FUNCTION.search(block)
            if not f:
                continue
            name = f.group(1).strip()
            args = {}
            for p, v in PARAMETER.findall(f.group(2)):
                v = v.removeprefix("\n").removesuffix("\n")
                args[p.strip()] = typed(v, schemas.get(name, {}).get(p.strip()))
            calls.append({"id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    content = (rest[:first] if calls else rest).strip()
    return reasoning, content, calls


@app.get("/health")
async def health():
    return JSONResponse({})


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "local", "max_model_len": CONTEXT}
                                       for n in S["names"]]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model") or S["names"][0]
    try:
        messages = prepare(body.get("messages", []))
    except ValueError:
        return error(400, f"{model} is not a multimodal model")
    tools = body.get("tools") or None
    kwargs = body.get("chat_template_kwargs") or {}
    prompt = S["hf"].apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, **kwargs)
    ids = S["hf"](prompt, add_special_tokens=False)["input_ids"]
    room = CONTEXT - len(ids)
    want = body.get("max_completion_tokens") or body.get("max_tokens") or room
    if room <= 0 or want > room:
        return error(400, f"This model's maximum context length is {CONTEXT} tokens. However, you requested "
                          f"{len(ids) + want} tokens ({len(ids)} in the messages, {want} in the completion).")
    def param(key, default):
        return default if body.get(key) is None else body[key]

    sampler = ComboSampler(temperature=param("temperature", 1.0), top_p=param("top_p", 0.95),
                           top_k=param("top_k", 0), min_p=param("min_p", 0.0))
    job = AsyncJob(S["gen"], input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=want, sampler=sampler,
                   stop_conditions=STOP_IDS, seed=body.get("seed"))
    rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    def chunk(delta, finish=None, usage=None):
        c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
        if usage is not None:
            c["choices"], c["usage"] = [], usage
        return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

    async def run(on_reasoning=None):
        """Generate; feed reasoning text to on_reasoning as it arrives. Returns (text, finish, completion tokens)."""
        text, sent, thought, n, reason = "", 0, False, 0, "stop"
        async for r in job:
            text += r.get("text", "")
            if on_reasoning and not thought:
                end = text.find(THINK_END)
                upto = end if end >= 0 else len(text) - (len(THINK_END) - 1)  # hold back a split </think>
                if upto > sent:
                    await on_reasoning(text[sent:upto])
                    sent = upto
                thought = end >= 0
            if r.get("eos"):
                n = r.get("new_tokens", 0)
                reason = "length" if r.get("eos_reason") == "max_new_tokens" else "stop"
        if on_reasoning and not thought and len(text) > sent:
            await on_reasoning(text[sent:])
        return text, reason, n

    if not body.get("stream"):
        try:
            text, reason, n = await run()
        except asyncio.CancelledError:
            await job.cancel()
            raise
        reasoning, content, calls = split_output(text, tools)
        msg = {"role": "assistant", "content": content or None, "reasoning": reasoning or None,
               "reasoning_content": reasoning or None, "tool_calls": calls}
        return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "logprobs": None,
                             "finish_reason": "tool_calls" if calls else reason}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}

    include_usage = (body.get("stream_options") or {}).get("include_usage")

    async def events():
        q = asyncio.Queue()

        async def on_reasoning(t):
            await q.put(chunk({"reasoning": t, "reasoning_content": t}))

        async def produce():
            try:
                await q.put(("done", await run(on_reasoning)))
            except Exception as e:  # noqa: BLE001 - reported to the client as a stream error
                await q.put(("fail", e))

        task = asyncio.create_task(produce())
        yield chunk({"role": "assistant", "content": ""})
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), 10)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    yield ": ping\n\n"
                    continue
                if isinstance(item, str):
                    yield item
                    continue
                kind, val = item
                if kind == "fail":
                    yield f"data: {json.dumps({'error': {'message': str(val), 'type': 'InternalServerError'}})}\n\n"
                    break
                text, reason, n = val
                _, content, calls = split_output(text, tools)
                if content:
                    yield chunk({"content": content})
                for i, c in enumerate(calls):
                    yield chunk({"tool_calls": [{"index": i, **c}]})
                yield chunk({}, "tool_calls" if calls else reason)
                if include_usage:
                    yield chunk({}, usage={"prompt_tokens": len(ids), "completion_tokens": n,
                                           "total_tokens": len(ids) + n})
                yield "data: [DONE]\n\n"
                break
        finally:
            if not task.done():
                task.cancel()
                await job.cancel()

    return StreamingResponse(events(), media_type="text/event-stream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cache-tokens", type=int, default=1_000_448, help="shared KV pool, tokens (multiple of 256)")
    ap.add_argument("--kv-bits", type=int, default=8)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens, layer_type=CacheLayer_quant, max_batch_size=32,
                  k_bits=args.kv_bits, v_bits=args.kv_bits)
    model.load(progressbar=True)
    S.update(names=args.name, hf=AutoTokenizer.from_pretrained(args.model))
    tok = Tokenizer.from_config(config)

    @app.on_event("startup")
    async def start():  # the async generator binds to the running loop
        S["gen"] = AsyncGenerator(model=model, cache=cache, tokenizer=tok, max_batch_size=32)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
