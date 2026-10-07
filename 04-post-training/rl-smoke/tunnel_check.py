"""Can an HF Job open a Prime tunnel? (prime-rl's sandboxed agents reach the model through one, via TCP 7000 out.)
Serves a tiny HTTP page locally, exposes it with prime_tunnel, fetches it back through the public URL."""
import asyncio, http.server, threading, httpx
from prime_tunnel import Tunnel

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"tunnel-ok")
    def log_message(self, *a): pass

async def main():
    srv = http.server.HTTPServer(("127.0.0.1", 9000), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    t = Tunnel(local_port=9000, name="hf-egress-check")
    try:
        url = await t.start()
        print("tunnel url", url, flush=True)
        for _ in range(10):
            try:
                r = httpx.get(url, timeout=15); print("fetch", r.status_code, r.text[:40], flush=True)
                if r.text.startswith("tunnel-ok"): print("RESULT: OK"); break
            except Exception as e:
                print("fetch error", e, flush=True)
            await asyncio.sleep(3)
    except Exception as e:
        print("RESULT: FAIL", type(e).__name__, e, "\n", "\n".join(t.recent_output[-20:]) if hasattr(t, "recent_output") else "")
    finally:
        await t.stop()

asyncio.run(main())
