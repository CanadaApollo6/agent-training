"""Mid-Harness (Kang et al., NVIDIA, arXiv 2609.39982) as a proxy: per-action best-of-N between the harness and an
OpenAI-compatible vLLM server, with the model judging its own candidates. Neither the harness nor the model changes.

    python midharness_proxy.py --upstream http://127.0.0.1:8000 --port 8001 -n 8 --log logs/midharness.jsonl

Each /v1/chat/completions request is sampled n times in one upstream request (one shared prefix, one batched decode)
with the client's own sampling settings. Identical actions (same answer text and tool calls) are merged. With more than
one distinct action, each is judged pointwise and decision-only, as in the paper's cheapest verifier that still helps:
the same model, thinking off, sees the conversation without any reasoning, then the candidate's text and tool calls
(never its reasoning), and answers Yes or No to "is this a correct, useful next step toward passing the task's tests?".
The score is P(Yes) from the first token's log-probs. The best-scoring action goes back to the client as the only
choice; ties go to the action sampled most often. Every judged step is logged to --log.

-n 1 passes requests straight through, so a baseline can run through the same proxy. Needs vLLM (n > 1, log-probs,
batched decode); TensorFold on CUDA decodes one request at a time and returns no log-probs. Other paths pass through.
"""
import argparse
import json
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JUDGE = """You are reviewing an agent working on the task above in a terminal sandbox. Its next action has been \
proposed but not run yet. The task is graded only by hidden tests run after the agent finishes.

Is the proposed action a correct and useful next step toward making those tests pass? Consider whether the command \
will run as intended in this environment, whether it does what the task needs, and whether it repeats or undoes \
earlier progress.

PROPOSED ACTION:
{action}

Answer with one word: Yes or No."""


def action_key(msg: dict) -> str:
    calls = []
    for c in msg.get("tool_calls") or []:
        f = c.get("function", {})
        try:
            args = json.dumps(json.loads(f.get("arguments") or "{}"), sort_keys=True)
        except json.JSONDecodeError:
            args = f.get("arguments") or ""
        calls.append([f.get("name"), args])
    return json.dumps([(msg.get("content") or "").strip(), calls])


def render(msg: dict) -> str:
    text = (msg.get("content") or "").strip()
    calls = "\n".join(f"[tool call] {c.get('function', {}).get('name')}: {c.get('function', {}).get('arguments')}"
                      for c in msg.get("tool_calls") or [])
    return "\n".join(x for x in (text, calls) if x) or "(no text and no tool calls: the agent would stop here)"


def strip_reasoning(messages: list) -> list:
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
        if m["role"] == "assistant" and isinstance(m.get("content"), str) and "</think>" in m["content"]:
            m["content"] = m["content"].split("</think>")[-1].lstrip()
        out.append(m)
    return out


class Proxy:
    def __init__(self, a):
        self.a, self.url = a, a.upstream.rstrip("/")
        self.pool = ThreadPoolExecutor(64)
        self.lock = threading.Lock()
        self.log = open(a.log, "a") if a.log else None

    def post(self, body: dict, timeout=4 * 3600) -> dict:
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))

    def judge(self, body: dict, history: list, msg: dict) -> float:
        req = {"model": body["model"], "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
               "messages": history + [{"role": "user", "content": JUDGE.format(action=render(msg))}],
               "chat_template_kwargs": {"enable_thinking": False}}
        if body.get("tools"):
            req["tools"] = body["tools"]
        top = self.post(req, timeout=600)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        p = {"yes": 0.0, "no": 0.0}
        for t in top:
            w = t["token"].strip().lower()
            if w in p:
                p[w] += math.exp(t["logprob"])
        return p["yes"] / (p["yes"] + p["no"]) if p["yes"] + p["no"] > 0 else 0.5

    def complete(self, body: dict) -> dict:
        if self.a.n <= 1:
            return self.post(body)
        t0 = time.time()
        res = self.post({**body, "n": self.a.n, "stream": False})
        t_gen = time.time() - t0
        groups = {}
        for c in res["choices"]:
            groups.setdefault(action_key(c["message"]), []).append(c)
        reps = [g[0] for g in groups.values()]
        counts = [len(g) for g in groups.values()]
        if len(reps) == 1:
            best, scores = 0, [None]
        else:
            history = strip_reasoning(body["messages"])
            scores = list(self.pool.map(lambda c: self.judge(body, history, c["message"]), reps))
            best = max(range(len(reps)), key=lambda i: (scores[i], counts[i]))
        if self.log:
            rec = {"time": time.time(), "turn": len(body["messages"]), "n": self.a.n, "distinct": len(reps),
                   "counts": counts, "scores": scores, "chosen": best,
                   "majority": max(range(len(reps)), key=lambda i: counts[i]), "gen_s": round(t_gen, 2),
                   "judge_s": round(time.time() - t0 - t_gen, 2),
                   "finish": [c.get("finish_reason") for c in reps]}
            with self.lock:
                self.log.write(json.dumps(rec) + "\n")
                self.log.flush()
        choice = {**reps[best], "index": 0}
        return {**res, "choices": [choice]}


def sse_chunks(res: dict):
    """A finished completion as stream chunks: one delta with the whole message, the finish reason, the usage."""
    c, base = res["choices"][0], {k: res[k] for k in ("id", "created", "model") if k in res}
    msg = c["message"]
    delta = {"role": "assistant", **{k: msg[k] for k in ("content", "reasoning", "reasoning_content") if msg.get(k)}}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
    yield {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    yield {**base, "object": "chat.completion.chunk",
           "choices": [{"index": 0, "delta": {}, "finish_reason": c.get("finish_reason")}]}
    if res.get("usage"):
        yield {**base, "object": "chat.completion.chunk", "choices": [], "usage": res["usage"]}


def handler(proxy: Proxy):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, code: int, data: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def forward(self, method: str, data: bytes | None):
            req = urllib.request.Request(proxy.url + self.path, data, {"Content-Type": "application/json"},
                                         method=method)
            try:
                with urllib.request.urlopen(req, timeout=4 * 3600) as r:
                    self.send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as e:
                self.send(e.code, e.read())

        def do_GET(self):
            self.forward("GET", None)

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self.forward("POST", data)
            body = json.loads(data)
            stream = body.pop("stream", False)
            body.pop("stream_options", None)
            try:
                res = proxy.complete(body)
            except urllib.error.HTTPError as e:
                return self.send(e.code, e.read())
            except Exception as e:  # noqa: BLE001 - the client gets the error, the proxy keeps serving
                return self.send(502, json.dumps({"error": {"message": f"midharness: {e!r}"}}).encode())
            if not stream:
                return self.send(200, json.dumps(res).encode())
            out = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in sse_chunks(res)) + b"data: [DONE]\n\n"
            self.send(200, out, "text/event-stream")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("-n", type=int, default=8, help="candidate actions per step (1: pass through)")
    ap.add_argument("--log", default=None, help="JSONL, one record per step with more than one candidate")
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(Proxy(a))).serve_forever()


if __name__ == "__main__":
    main()
