"""Test-only driver: independent native processes, Unicode paths, hot journal recovery."""
import pathlib
import subprocess
import sys
import tempfile

binary = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="fyodor-store-") as directory:
    # The fixture's ASCII marker selects a UTF-8 C literal, directly testing the
    # public API path without depending on the test helper's narrow Windows argv.
    root = pathlib.Path(directory) / "\ud55c\uae00-\U0001f30d"
    root.mkdir()
    for filename, argument in [("workspace.db", "workspace.db"), ("workspace-\ud55c\uae00-\U0001f30d.db", "__unicode__")]:
        path = root / filename
        for mode, expected in [("--seed", 0), ("--verify", 0), ("--crash", 23), ("--verify", 0)]:
            result = subprocess.run([binary, mode, argument], cwd=root, capture_output=True, timeout=30)
            if result.returncode != expected:
                raise AssertionError((mode, result.returncode, result.stdout, result.stderr))
            if mode == "--crash":
                journal = pathlib.Path(str(path) + "-journal")
                assert journal.is_file() and journal.stat().st_size > 512, "crash did not preserve a recovery journal"
            print(f"{mode}: expected exit {expected}")
        assert not pathlib.Path(str(path) + "-journal").exists(), "recovery left a journal"
        processes = [subprocess.Popen([binary, "--update", argument], cwd=root,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        try:
            outputs = [process.communicate(timeout=30) for process in processes]
            assert sorted(process.returncode for process in processes) == [0, 24], outputs
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        result = subprocess.run([binary, "--verify-updated", argument], cwd=root, capture_output=True, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
print("independent process persistence, concurrent revision conflict, UTF-8 and hot-journal recovery passed")
