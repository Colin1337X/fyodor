"""Verify extracted NSIS contents and run the native stop suite on its trainer."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
installer = root / "frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe"
extracted = root / "build-train-profile/installer-executor-verified"
assert extracted.resolve().is_relative_to((root / "build-train-profile").resolve())
extracted.mkdir(exist_ok=False)
command = ["C:/Program Files/7-Zip/7z.exe", "x", str(installer), "-o" + str(extracted), "-y"]
result = subprocess.run(command, capture_output=True, text=True, timeout=120)
(out / "installer-extract.log").write_text(result.stdout + result.stderr, encoding="utf-8")
assert result.returncode == 0


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


contents = {}
resources = root / "frontend/src-tauri/resources/backend"
for source in resources.iterdir():
    if source.is_file():
        name = "backend/" + source.name
        assert sha(source) == sha(extracted / name), name
        contents[name] = sha(source)
app = (root / "frontend/src-tauri/target/release/fyodor-desktop.exe").read_bytes()
marker = b"__TAURI_BUNDLE_TYPE_VAR_UNK"
assert app.count(marker) == 1
assert (extracted / "fyodor-desktop.exe").read_bytes() == app.replace(marker, b"__TAURI_BUNDLE_TYPE_VAR_NSS", 1)
contents["fyodor-desktop.exe"] = sha(extracted / "fyodor-desktop.exe")
test = [sys.executable, str(root / "backend/tests/train_stop.py"),
        str(extracted / "backend/fyodor-train.exe"), "--evidence", str(out / "packaged-stop-workflows.json")]
result = subprocess.run(test, capture_output=True, text=True, timeout=120)
assert result.returncode == 0, result.stdout + result.stderr
unicode_test = [sys.executable, str(root / "backend/tests/train_paths.py"),
                str(extracted / "backend/fyodor-train.exe"), "--evidence", str(out / "packaged-paths.json")]
unicode_result = subprocess.run(unicode_test, capture_output=True, text=True, timeout=120)
assert unicode_result.returncode == 0, unicode_result.stdout + unicode_result.stderr
record = {"unicode_test_command": unicode_test, "unicode_test_stdout": unicode_result.stdout, "installer": str(installer), "bytes": installer.stat().st_size, "sha256": sha(installer),
          "verified_contents_sha256": contents, "extraction_command": command,
          "desktop_comparison": "Identical after the single expected Tauri UNK -> NSS bundle marker replacement.",
          "test_command": test, "test_stdout": result.stdout, "installed": False}
cli = [str(root / ".tools/w64devkit/bin/cmake.exe"), "-DTRAINER=" + str(extracted / "backend/fyodor-train.exe"),
       "-DFIXTURES=" + str(extracted), "-P", str(root / "backend/tests/TrainCli.cmake")]
cli_result = subprocess.run(cli, capture_output=True, text=True, timeout=120)
(out / "packaged-cli.log").write_text(cli_result.stdout + cli_result.stderr, encoding="utf-8")
assert cli_result.returncode == 0
record["cli_command"] = cli
record["cli_exit_code"] = cli_result.returncode
(out / "desktop-package.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
