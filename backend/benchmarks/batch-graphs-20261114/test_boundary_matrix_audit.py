import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit_boundary_matrix as matrix


class MatrixAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'proofs'
        self.out.mkdir()
        (self.out / 'run_boundary_memcheck.py').write_text('fixture runner')
        self.source = self.out / 'audit_boundary_matrix.py'
        self.source.write_text('fixture audit')
        self.patcher = patch.object(matrix, '__file__', str(self.source))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        output = '\n'.join(f'reset={r} position={p} count={c} trial={t} parity=1'
            for r in range(2) for p,c in ((250,4),(254,2),(256,2),(258,4),(8190,2),(8192,4),(8196,2))
            for t in range(3))
        trace = 'CUDA batched graph captures: 6 replays: 42\n========= ERROR SUMMARY: 0 errors\n'
        for standard, width, ring in itertools.product(('c17','c23'),('256','512'),('0','1')):
            folder = self.out / f'boundary-memcheck-{standard}-{width}-ring{ring}'
            folder.mkdir()
            (folder / 'stdout').write_text(output)
            (folder / 'stderr').write_text(trace)
            paths = [folder / 'stdout', folder / 'stderr', self.out / 'run_boundary_memcheck.py']
            hashes = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            result = dict(passed=True, accepted_benchmark=False, errors=[], exit_code=0, sha256=hashes,
                command=['sanitizer','--tool','memcheck','--error-exitcode','99',f'boundaries-{standard}.exe','fixture',width,ring])
            (folder / 'result.json').write_text(json.dumps(result))
        self.target = self.out / 'boundary-memcheck-c23-512-ring1'

    def test_complete(self):
        result = matrix.audit(self.root, self.out)
        self.assertEqual(result['parity_cases'], 336)
        self.assertEqual(len(result['configurations']), 8)
        self.assertFalse(result['accepted_benchmark'])

    def test_missing_configuration(self):
        (self.target / 'result.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Pending boundary'):
            matrix.audit(self.root, self.out)

    def test_mismatched_executable(self):
        path = self.target / 'result.json'
        result = json.loads(path.read_text())
        result['command'][5] = 'boundaries-c17.exe'
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'Wrong boundary'):
            matrix.audit(self.root, self.out)

    def test_changed_evidence(self):
        with (self.target / 'stdout').open('a') as output:
            output.write('\nextra diagnostic line\n')
        with self.assertRaisesRegex(ValueError, 'Changed boundary evidence'):
            matrix.audit(self.root, self.out)

    def test_missing_hash(self):
        path = self.target / 'result.json'
        result = json.loads(path.read_text())
        del result['sha256'][str((self.target / 'stderr').relative_to(self.root))]
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'Missing evidence hash'):
            matrix.audit(self.root, self.out)


if __name__ == '__main__':
    unittest.main()
