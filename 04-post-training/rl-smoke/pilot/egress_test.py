# Does a Prime sandbox block GitHub while restricted, and reopen it on prepare_execution(None)?
import asyncio
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig

PROBE = """
import urllib.request, sys
for u in sys.argv[1:]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=15); print("OK  ", r.status, u)
    except urllib.error.HTTPError as e: print("OK  ", e.code, u)
    except Exception as e: print("FAIL", type(e).__name__, str(e)[:80], u)
"""
URLS = ["https://github.com", "https://raw.githubusercontent.com/ocaml/ocaml/trunk/README.adoc",
        "https://api.github.com", "https://astral.sh/uv/0.9.5/install.sh", "https://pypi.org/simple/",
        "https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz"]

async def probe(rt, label):
    r = await rt.run(["python3", "-c", PROBE, *URLS], {})
    print(f"--- {label}\n{r.stdout}{r.stderr}")

async def main():
    cfg = PrimeConfig(block=["github.com", "*.github.com", "githubusercontent.com", "*.githubusercontent.com"], labels=["egress-test"])
    async with provision_runtime(cfg, name="egress-test") as rt:
        await rt.prepare_setup()
        await probe(rt, "setup (open)")
        await rt.prepare_execution(["https://example.com/v1"])
        await probe(rt, "agent phase (GitHub blocked)")
        await rt.prepare_execution(None)
        await probe(rt, "after reopen (scoring)")

asyncio.run(main())
