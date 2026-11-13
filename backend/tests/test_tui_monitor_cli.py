"""Shared native palette integration through both executable entry points."""
import os
import subprocess
import sys

exe = sys.argv[1]
env = {key: value for key, value in os.environ.items() if key != "FYODOR_THEME"}

def run(args, expected=0, overrides=None):
    result = subprocess.run([exe, *args], env={**env, **(overrides or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    assert result.returncode == expected, (args, result.returncode, result.stderr)
    return result.stdout

ids = run(["--themes"]).decode().splitlines()
assert ids and len(ids) == len(set(ids))
assert run(["--monitor", "--themes"]).decode().splitlines() == ids
assert run(["--monitor", "--licenses"]) == run(["--licenses"])
for theme in ids:
    output = run(["--monitor", "--theme", theme, "--color", "truecolor", "--snapshot"])
    assert b"FYODOR" in output and b"\x1b" not in output
run(["--monitor", "--theme", "missing", "--snapshot"], 2)
run(["--monitor", "--color", "missing", "--snapshot"], 2)
run(["--monitor", "--theme"], 2)
run(["--monitor", "--snapshot"], 2, {"FYODOR_THEME": "missing"})
assert b"FYODOR" in run(["--monitor", "--snapshot", "--theme", ids[0]],
                         overrides={"FYODOR_THEME": "missing", "NO_COLOR": "1"})
print(f"Shared palettes, licenses, {len(ids)} plain snapshots, invalid options and environment override passed")
