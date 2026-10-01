"""The Anthropic Messages API (``/v1/messages``) in front of an OpenAI-compatible server (vLLM), so Claude Code can drive
it with the same translation TensorFold's own endpoint uses (tensorfold/server/anthropic.py, loaded from its file).

    python messages_proxy.py --upstream http://127.0.0.1:8000 --port 8080 [--context 262144]

Each request is tokenized first (vLLM's ``/tokenize``), so the stream opens with the real input token count, a prompt
over ``--context`` is refused in the API's words ("prompt is too long", Claude Code's cue to compact) and ``max_tokens``
is clamped to the room left. The upstream reply always streams; the client gets a Message or Messages events, with a
ping every 10 s while it waits.
"""
import argparse
import importlib.util
import json
import queue
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSLATOR = HERE.parents[1] / "01-inference/tools/TensorFold/src/tensorfold/server/anthropic.py"
PING_S = 10.0


def load(path: Path):
    spec = importlib.util.spec_from_file_location("anthropic_messages", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Upstream:
    def __init__(self, url: str, context: int, model: str | None):
        self.url, self.context = url.rstrip("/"), context
        self.model = model or json.load(urllib.request.urlopen(f"{self.url}/v1/models", timeout=30))["data"][0]["id"]

    def post(self, path: str, body: dict, timeout: float = 30):
        request = urllib.request.Request(f"{self.url}{path}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=timeout)

    @staticmethod
    def messages(chat: dict) -> list:
        """The chat messages as TensorFold renders them for this template: leading system messages merged into one,
        later ones (Claude Code sends some) as user messages, since Qwen's template refuses a system message after the
        start; and earlier reasoning under both keys, since vLLM 0.30 renders ``reasoning`` and silently drops
        ``reasoning_content``, which older servers (and the model's template) read."""

        out, lead = [], []
        for m in chat["messages"]:
            if m["role"] == "system":
                if not out:
                    lead.append(m["content"])
                    continue
                m = {**m, "role": "user"}
            out.append({**m, "reasoning": m["reasoning_content"]} if m.get("reasoning_content") else m)
        return ([{"role": "system", "content": "\n\n".join(lead)}] if lead else []) + out

    def count(self, chat: dict) -> int:
        body = {"model": self.model, "messages": self.messages(chat), "add_generation_prompt": True}
        for key in ("tools", "chat_template_kwargs"):
            if key in chat:
                body[key] = chat[key]
        return int(json.load(self.post("/tokenize", body))["count"])

    def stream(self, chat: dict):
        """(kind, value) events of one streamed chat reply: ("delta", {...}), then ("done", result)."""

        body = {**chat, "messages": self.messages(chat), "model": self.model, "stream": True,
                "stream_options": {"include_usage": True}}
        reasoning, content, calls, finish, usage = [], [], {}, "stop", {}
        with self.post("/v1/chat/completions", body, timeout=4 * 3600) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    think = delta.get("reasoning_content") or delta.get("reasoning")
                    out = {}
                    if think:
                        reasoning.append(think)
                        out["reasoning_content"] = think
                    if delta.get("content"):
                        content.append(delta["content"])
                        out["content"] = delta["content"]
                    if out:
                        yield "delta", out
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(call.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        fn = call.get("function") or {}
                        slot["id"] = call.get("id") or slot["id"]
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish
        calls_out = [{"id": c["id"] or f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": c["name"], "arguments": c["arguments"]}} for _, c in sorted(calls.items())]
        yield "done", {"final": {}, "calls": calls_out or None, "finish": "tool_calls" if calls_out else finish,
                       "content": "".join(content), "reasoning": "".join(reasoning),
                       "prompt_tokens": usage.get("prompt_tokens", 0),
                       "completion_tokens": usage.get("completion_tokens", 0)}


def handler(up: Upstream, am):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/health"):
                return self._json(200, {"object": "list", "data": [{"id": up.model, "object": "model"}]})
            self._json(404, am.error("not_found_error", "not found"))

        def do_POST(self):
            route = self.path.split("?", 1)[0].rstrip("/")
            if not (route.endswith("/messages") or route.endswith("/messages/count_tokens")):
                return self._json(404, am.error("not_found_error", "not found"))
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                chat = am.to_chat(body)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                return self._json(400, am.error("invalid_request_error", f"malformed Messages request: {exc}"))
            try:
                prompt = up.count(chat)
            except urllib.error.HTTPError as exc:
                return self._json(400, am.error("invalid_request_error", exc.read().decode()[:2000]))
            if route.endswith("/count_tokens"):
                return self._json(200, {"input_tokens": prompt})
            if prompt >= up.context:
                return self._json(400, am.error("invalid_request_error",
                                                f"prompt is too long: {prompt} tokens > {up.context} maximum"))
            chat["max_tokens"] = max(1, min(int(chat.get("max_tokens") or up.context), up.context - prompt))
            rid, model = f"msg_{uuid.uuid4().hex[:24]}", str(body.get("model") or up.model)
            events: queue.Queue = queue.Queue()

            def work():
                try:
                    for event in up.stream(chat):
                        events.put(event)
                except Exception as exc:          # noqa: BLE001  handed to the writer
                    events.put(("error", exc))

            threading.Thread(target=work, daemon=True).start()
            if not chat["stream"]:
                while True:
                    kind, item = events.get()
                    if kind == "done":
                        item["prompt_tokens"] = item["prompt_tokens"] or prompt
                        return self._json(200, am.message(item, model, rid))
                    if kind == "error":
                        detail = item.read().decode()[:2000] if isinstance(item, urllib.error.HTTPError) else str(item)
                        return self._json(500, am.error("api_error", detail))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            reply = am.EventStream(model, rid, prompt)

            def write(text: str) -> None:
                self.wfile.write(text.encode())
                self.wfile.flush()

            try:
                write(reply.start())
                while True:
                    try:
                        kind, item = events.get(timeout=PING_S)
                    except queue.Empty:
                        write(am.sse("ping", {"type": "ping"}))
                        continue
                    if kind == "delta":
                        write(reply.delta(item))
                    elif kind == "done":
                        return write(reply.finish(item))
                    else:
                        detail = item.read().decode()[:2000] if isinstance(item, urllib.error.HTTPError) else str(item)
                        return write(am.sse("error", am.error("api_error", detail)))
            except OSError:                       # the client left; the upstream request finishes on its own
                return

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--context", type=int, default=262144, help="the upstream's max model length")
    ap.add_argument("--model", help="upstream model id (default: the first it lists)")
    ap.add_argument("--translator", type=Path, default=TRANSLATOR)
    a = ap.parse_args()
    up = Upstream(a.upstream, a.context, a.model)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(up, load(a.translator)))
    print(f"messages proxy on 127.0.0.1:{a.port} -> {a.upstream} ({up.model}, context {a.context})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
