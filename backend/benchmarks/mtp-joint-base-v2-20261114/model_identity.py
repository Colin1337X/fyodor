"""Read-only, bounded-memory identity checks for external benchmark models."""
import hashlib
import json
import os
from pathlib import Path


def verify(paths):
    result = {}
    for name, expected in paths.items():
        path = Path(name)
        with path.open('rb') as source:
            before = os.fstat(source.fileno())
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
            after = os.fstat(source.fileno())
            current = path.stat()
        def signature(stat):
            return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev
        if signature(before) != signature(after) or signature(after) != signature(current):
            raise ValueError(f'Model changed during hashing: {path}')
        if actual != expected:
            raise ValueError(f'Model identity mismatch: {path}')
        result[str(path)] = dict(sha256=actual, size=after.st_size, mtime_ns=after.st_mtime_ns)
    return result


def verify_gemma():
    provenance = Path(__file__).resolve().parent.parent / 'mtp-resident-target-20261113/model-results.json'
    recorded = json.loads(provenance.read_text())
    assistant = 'E:\\models\\mtp-gemma-4-12B-it.gguf'
    models = verify({recorded['target']: recorded['target_sha256'], assistant: recorded['hashes'][assistant]})
    return dict(models=models, provenance=str(provenance),
                provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest())
