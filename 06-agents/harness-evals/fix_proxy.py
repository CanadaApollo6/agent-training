"""Harness fixes as a proxy between vf-eval and the model server: neither prime_agent nor the model changes.

    python fix_proxy.py --upstream https://<job>--8080.hf.jobs --key-var HF_TOKEN --port 8101 \
        --fixes cutoff,check,image --log logs/fixes-NAME.jsonl

Three fixes, each aimed at a measured way R1s-SD loses prime_agent runs (README, "Reply cap under prime_agent"):

- cutoff: a reply that hits the length cap before any tool call ends the run as agent_completed (the model thought for
  the whole 32K). The reply is dropped and the request sent again with a note after the last message: you ran out of
  room while thinking, act now with code. Up to --cutoff-retries times a turn.
- check: the first time a run's reply has no tool call (the model says it's done), its reply is kept and a note asks
  it to re-read the task, list every output the tests will look for, and check each with code. Its next reply goes
  back to the harness instead. Once per run.
- image: image parts in the conversation (the model viewing a picture) become a text note, instead of the server's
  "image input requires a supported vision checkpoint" 400 that ends the run.

The harness never sees the notes, so each later request of that run gets them put back where the model saw them: an
injection is remembered by its position and by the reply that followed it (tool call ids, else the text), and matched
on the run's first user message. --fixes "" passes everything through, so a baseline runs through the same proxy.
Every fix that fires is logged to --log (time, fix, turn, what came back).
"""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CUTOFF_NOTE = ("Your last reply hit the length limit while you were still thinking, before you did anything, so it was "
               "thrown away. Don't work it all out in your head: take the next concrete step now with a tool call, and "
               "let code do the decoding, searching and checking.")
CHECK_NOTE = ("Before you finish: this task is graded by hidden tests that run after you stop, on the files and state "
              "you leave behind. Re-read the task and list every output it asks for (files, paths, names, formats, "
              "running services). Check each one now with code, and fix anything missing or wrong. If everything checks "
              "out, reply with a one-line summary and no tool call.")
IMAGE_NOTE = "[image omitted: this model can't view images. Inspect the file with code instead.]"


def text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return (c or "").split("</think>")[-1].strip()


def sig(msg: dict) -> str:
    """What identifies the reply that followed an injection: its tool call ids, else its text."""
    ids = [c.get("id") for c in msg.get("tool_calls") or [] if c.get("id")]
    return "ids:" + ",".join(ids) if ids else "text:" + hashlib.sha1(text_of(msg).encode()).hexdigest()


def run_key(messages: list) -> str:
    first = next((m for m in messages if m.get("role") == "user"), {})
    return hashlib.sha1(json.dumps(first.get("content"), sort_keys=True).encode()).hexdigest()


def no_images(messages: list) -> tuple[list, int]:
    out, n = [], 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") not in ("text", None) for p in c):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") not in ("text", None):
                    parts.append({"type": "text", "text": IMAGE_NOTE})
                    n += 1
                else:
                    parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out, n


class Proxy:
    def __init__(self, a):
        self.a, self.url = a, a.upstream.rstrip("/")
        self.fixes = {f for f in a.fixes.split(",") if f}
        self.key = os.environ.get(a.key_var, "") if a.key_var else ""
        self.lock = threading.Lock()
        self.injections = {}    # run key -> [(position, [notes], signature of the reply after them)]
        self.checked = set()    # run keys (+ first reply signature) whose finish was already checked
        self.log = open(a.log, "a") if a.log else None

    def note(self, **rec):
        if self.log:
            with self.lock:
                self.log.write(json.dumps({"time": round(time.time(), 1), **rec}) + "\n")
                self.log.flush()

    def post(self, body: dict) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", json.dumps(body).encode(), h)
        return json.load(urllib.request.urlopen(req, timeout=4 * 3600))

    def restore(self, key: str, messages: list) -> list:
        """The run's conversation as the model saw it: the notes put back before the replies that followed them."""
        with self.lock:
            inj = sorted(self.injections.get(key, []), key=lambda x: x[0])
        out, shift = list(messages), 0
        for pos, notes, s in inj:
            if pos < len(messages) and sig(messages[pos]) == s:
                out[pos + shift:pos + shift] = notes
                shift += len(notes)
        return out

    def remember(self, key: str, pos: int, notes: list, reply: dict):
        with self.lock:
            self.injections.setdefault(key, []).append((pos, notes, sig(reply)))

    def complete(self, body: dict) -> dict:
        harness_msgs = body["messages"]
        key = run_key(harness_msgs)
        msgs = self.restore(key, harness_msgs) if self.fixes else harness_msgs
        if "image" in self.fixes:
            msgs, n = no_images(msgs)
            if n:
                self.note(fix="image", turn=len(harness_msgs), parts=n)
        res = self.post({**body, "messages": msgs})
        pos, notes = len(harness_msgs), []
        msg, fin = res["choices"][0]["message"], res["choices"][0].get("finish_reason")
        if "cutoff" in self.fixes:
            for i in range(self.a.cutoff_retries):
                if fin != "length" or msg.get("tool_calls"):
                    break
                notes.append({"role": "user", "content": CUTOFF_NOTE})
                res = self.post({**body, "messages": msgs + notes})
                msg, fin = res["choices"][0]["message"], res["choices"][0].get("finish_reason")
                self.note(fix="cutoff", turn=pos, retry=i + 1, finish=fin, acted=bool(msg.get("tool_calls")))
        if "check" in self.fixes and not msg.get("tool_calls") and fin == "stop":
            first_reply = next((m for m in harness_msgs if m.get("role") == "assistant"), None)
            ck = key + (sig(first_reply) if first_reply else "")
            with self.lock:
                fresh = ck not in self.checked
                self.checked.add(ck)
            if fresh:
                notes += [{"role": "assistant", "content": text_of(msg)}, {"role": "user", "content": CHECK_NOTE}]
                res = self.post({**body, "messages": msgs + notes})
                msg, fin = res["choices"][0]["message"], res["choices"][0].get("finish_reason")
                self.note(fix="check", turn=pos, finish=fin, acted=bool(msg.get("tool_calls")))
        if notes:
            self.remember(key, pos, notes, msg)
        return res


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
            h = {"Content-Type": "application/json"}
            if proxy.key:
                h["Authorization"] = f"Bearer {proxy.key}"
            req = urllib.request.Request(proxy.url + self.path, data, h, method=method)
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
                return self.send(502, json.dumps({"error": {"message": f"fix_proxy: {e!r}"}}).encode())
            if not stream:
                return self.send(200, json.dumps(res).encode())
            out = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in sse_chunks(res)) + b"data: [DONE]\n\n"
            self.send(200, out, "text/event-stream")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="server root, without /v1")
    ap.add_argument("--key-var", default=None, help="env var holding the upstream API key (HF_TOKEN for HF jobs)")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--fixes", default="cutoff,check,image", help='comma list of cutoff, check, image ("" = none)')
    ap.add_argument("--cutoff-retries", type=int, default=2)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(Proxy(a))).serve_forever()


if __name__ == "__main__":
    main()
