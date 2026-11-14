"""The replay runner must refuse incomplete/stale evidence before CUDA import."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_replay


class Prerequisites(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'backend/benchmarks/batch-graph-request-20261114'
        self.out.mkdir(parents=True)
        self.model = self.out.parent/'integer-metadata-20261114/model-memcheck-short/result.json'
        self.model.parent.mkdir(parents=True)
        self.input = self.root/'input.c'
        self.input.write_bytes(b'original source')
        self.proof = dict(passed=True,exit_code=0,
            sha256={'input.c':hashlib.sha256(self.input.read_bytes()).hexdigest()})
        for path in (self.model, self.out/'host.json', self.out/'compile.json'):
            path.write_text(json.dumps(self.proof))
        self.root_patch = patch.object(check_replay,'root',self.root)
        self.out_patch = patch.object(check_replay,'out',self.out)
        self.root_patch.start(); self.out_patch.start()
        self.addCleanup(self.root_patch.stop); self.addCleanup(self.out_patch.stop)

    def test_valid_evidence_includes_hashed_input(self):
        self.assertIn(self.input, check_replay.prerequisites())

    def test_live_without_terminal_evidence_is_refused(self):
        self.model.unlink()
        with self.assertRaisesRegex(RuntimeError,'Wait for completed prerequisite'):
            check_replay.prerequisites()

    def test_failed_proof_is_refused(self):
        self.proof['passed'] = False
        self.model.write_text(json.dumps(self.proof))
        with self.assertRaisesRegex(RuntimeError,'Failed prerequisite'):
            check_replay.prerequisites()

    def test_failed_sanitizer_exit_is_refused(self):
        self.proof['exit_code'] = 99
        self.model.write_text(json.dumps(self.proof))
        with self.assertRaisesRegex(RuntimeError,'did not exit successfully'):
            check_replay.prerequisites()

    def test_changed_source_is_refused(self):
        self.input.write_bytes(b'changed source')
        with self.assertRaisesRegex(RuntimeError,'Changed prerequisite'):
            check_replay.prerequisites()


if __name__ == '__main__':
    unittest.main()
