"""Real native CLI resource lifecycle; Python is test-only."""
import json
import pathlib
import subprocess
import sys
import tempfile

binary = str(pathlib.Path(sys.argv[1]).resolve())
model = str(pathlib.Path(sys.argv[2]).resolve()) if len(sys.argv)>2 else None
def run(args, code=0, data=None):
    result = subprocess.run([binary, *args], input=data, capture_output=True, timeout=30)
    assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
    if code == 0:
        assert not result.stderr, result.stderr
    else:
        assert not result.stdout and result.stderr, (result.stdout, result.stderr)
    return result.stdout

with tempfile.TemporaryDirectory(prefix="fyodor-cli-") as directory:
    path = pathlib.Path(directory) / "\ud55c\uae00-\U0001f30d.db"
    args = ["--store", str(path), "--namespace", "workspace", "resource"]
    run(["--help"])
    run(["nonsense"], 2)
    uri = json.loads(run(["resource", "id", "document"]))["uri"]
    run(args + ["delete", uri, "-1"], 2)
    assert not path.exists(), "invalid syntax created database"
    package = {"schema":1,"uri":uri,"title":"Unicode \ud55c\uae00 \U0001f30d", "content":"line\nwith \"quotes\"\tand <script>",
               "metadata":{"tags":["test"]}, "provenance":{"source":"fixture"}}
    encoded = json.dumps(package, ensure_ascii=False).encode("utf-8")
    result = json.loads(run(args + ["import"], data=encoded))
    assert result == {"uri":uri,"revision":1}
    assert json.loads(run(args + ["export", uri])) == package
    rows = [json.loads(row) for row in run(args + ["list"]).splitlines()]
    assert rows == [{"uri":uri,"title":package["title"],"revision":1}]
    context = args[:-1] + ["context"]
    principal = "fedcba98-7654-4321-8123-456789abcdef"
    run(context + ["inspect", principal, uri], 6)
    assert run(context + ["search", principal, "line"]) == b""
    assert json.loads(run(context + ["permissions", principal, uri, "5"])) == {"permissions":5}
    assert json.loads(run(context + ["inspect", principal, uri]))["content"] == package["content"]
    assert json.loads(run(context + ["search", principal, "line"]))["uri"] == uri
    explanation = json.loads(run(context + ["explain", principal, "4", uri]))
    assert explanation["text"] == "line" and explanation["bytes"] == 4
    assert explanation["sources"][0]["revision"] == 1 and explanation["sources"][0]["length"] == 4
    receipt_id = None
    if model:
        generation = json.loads(run(context + ["generate", principal, model, "4", "2", "a", uri]))
        receipt_id = generation["receipt_id"]
        assert generation["generated_tokens"] > 0 and generation["prompt_tokens"] > 0
        saved = json.loads(run(context + ["receipt", principal, receipt_id]))
        assert saved["prompt"] == "line\n\na" and saved["output"] == generation["text"]
        assert saved["metadata"]["prompt_tokens"] == generation["prompt_tokens"]
        assert saved["sources"][0]["uri"] == uri and saved["sources"][0]["revision"] == 1
    run(context + ["permissions", principal, uri, "0"])
    run(context + ["explain", principal, "0", uri], 6)
    if receipt_id:
        run(context + ["receipt", principal, receipt_id], 6)
        run(context + ["generate", principal, model, "4", "2", "a", uri], 6)
    # Mutate a separate resource as an append-only principal without READ.
    separate = dict(package)
    separate["uri"] = json.loads(run(["resource", "id", "document"]))["uri"]
    target = separate["uri"]
    run(args + ["import"], data=json.dumps(separate).encode())
    run(context + ["append", principal, target, "1", "addition"], 6)
    run(context + ["permissions", principal, target, "8"])
    assert json.loads(run(context + ["append", principal, target, "1", "addition"])) == {"revision":2}
    run(context + ["inspect", principal, target], 6)
    run(context + ["append", principal, target, "1", "stale"], 4)
    assert json.loads(run(args + ["export", target]))["content"] == separate["content"] + "addition"
    run(context + ["delete", principal, target, "2"], 6)
    run(context + ["permissions", principal, target, "16"])
    assert json.loads(run(context + ["delete", principal, target, "2"])) == {"deleted":True}
    run(args + ["import"], 4, encoded)
    package["content"] = "updated"
    updated = json.dumps(package).encode()
    assert json.loads(run(args + ["import", "1"], data=updated))["revision"] == 2
    run(args + ["import", "1"], 4, updated)
    run(args + ["import"], 2, b'{"schema":1}')
    run(args + ["import"], 2, b"x" * (8*1024*1024 + 1))
    assert json.loads(run(args + ["export", uri])) == package
    run(args + ["delete", uri, "1"], 4)
    assert json.loads(run(args + ["delete", uri, "2"])) == {"deleted":True}
    run(args + ["export", uri], 3)
    assert run(args + ["list"]) == b""
    run(args + ["import"], 4, encoded)
print("native CLI: Unicode paths, piped JSON, lifecycle, conflicts, limits and exit contracts passed")
