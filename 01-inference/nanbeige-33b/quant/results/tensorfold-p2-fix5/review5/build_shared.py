from pathlib import Path
import subprocess
p = Path(__file__).parent
base = (p / "shared.zig").read_text()
for label, folder in [("head", "source"), ("parent", "parent")]:
    source = p / ("shared-" + label + ".zig")
    source.write_text(base.replace("source/zig/", folder + "/zig/"))
    r = p / folder / "zig/src"
    args = [str(Path.home() / ".local/share/zig-x86_64-linux-0.17.0/zig"), "test", "--test-no-exec",
            "--test-filter", "independent", "-O", "ReleaseSafe", "-femit-bin=" + str(p / ("shared-" + label)),
            "-lc", "--dep", "json", "--dep", "sys", "-Mroot=" + str(source), "--dep", "sys",
            "-Mjson=" + str(r / "server/json.zig"), "-Msys=" + str(r / "platform/sys.zig")]
    with (p / ("shared-build-" + label + ".log")).open("w") as f:
        subprocess.run(args, stdout=f, stderr=f, check=True)
