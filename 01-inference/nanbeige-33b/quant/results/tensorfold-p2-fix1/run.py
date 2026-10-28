import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3] / "tools/TensorFold-llama"
label, *command = sys.argv[1:]
env = dict(os.environ, TF_LLAMA_MODEL=str(OUT.parents[1] / "models/3.5-nb"))
if command[0] == "zig":
    command[0] = str(Path.home() / ".local/share/zig-x86_64-linux-0.17.0/zig")
    command += ["-j4", "--prefix", str(OUT / "install"), "--cache-dir", str(OUT / "host-cache"), "--summary", "all"]
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (OUT / (label + ".log")).open("w") as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
row = dict(label=label, start=start, command=command, cwd=str(ROOT), model=env["TF_LLAMA_MODEL"], exit=result.returncode)
with (OUT / "commands.jsonl").open("a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row), flush=True)
sys.exit(result.returncode)
