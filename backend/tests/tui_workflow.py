"""Exercise the real TUI, shared CLI store, and safe redirected fallback."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

tui, cli = map(lambda p: str(Path(p).resolve()), sys.argv[1:3])
with tempfile.TemporaryDirectory(prefix="fyodor-tui-") as directory:
    db = str(Path(directory) / "workspace-\u6587.db")
    base = ["--store", db, "--namespace", "test"]
    uris = sorted("fyodor://writing/notes/" + str(uuid.uuid4()) for _ in range(22))
    for index, uri in enumerate(uris):
        package = dict(schema=1, uri=uri, title=f"Note {index:02} \u6587", content="body\n\x1b]52;injected\x07", metadata={}, provenance={})
        run = subprocess.run([cli, *base, "resource", "import"], input=json.dumps(package), text=True, encoding="utf-8", capture_output=True)
        assert run.returncode == 0, run.stderr
    commands = "list\nopen 1\nback\nnext\nopen 2\n:resource list\n:resource import\n:resource export 'unterminated\n" + "x" * 9000 + "\nhelp\nquit\n"
    for options in ([], ["--plain"]):
        run = subprocess.run([tui, *base, *options], input=commands, text=True, encoding="utf-8", capture_output=True, timeout=20)
        assert run.returncode == 0, run.stderr
        assert "Note 00" in run.stdout and "Note 21" in run.stdout
        assert "\\x1b]52;injected\\x07" in run.stdout
        assert "Command exit 0" in run.stdout and "Command syntax error" in run.stdout
        assert "Import packages through the CLI" in run.stdout and "Command too long; discarded" in run.stdout
        assert "\x1b" not in run.stdout and "\\xe6\\x96\\x87" in run.stdout
    run = subprocess.run([tui, *base], input="", text=True, capture_output=True, timeout=10)
    assert run.returncode == 0
    run = subprocess.run([tui], text=True, capture_output=True, timeout=10)
    assert run.returncode == 2
    for options in (["--theme", "not-a-theme"], ["--color", "invalid"]):
        run = subprocess.run([tui, *base, *options], input="quit\n", text=True, capture_output=True, timeout=10)
        assert run.returncode == 2
    run = subprocess.run([tui, "--themes"], text=True, capture_output=True, timeout=10)
    assert run.returncode == 0 and len(run.stdout.splitlines()) == 10
    run = subprocess.run([tui, "--licenses"], text=True, capture_output=True, timeout=10)
    assert run.returncode == 0 and "Copyright (c) 2021 Catppuccin" in run.stdout
print("Native TUI fallback, paging, shared commands and safe text passed")
