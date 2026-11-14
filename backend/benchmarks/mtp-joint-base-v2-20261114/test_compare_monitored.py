import json
from pathlib import Path
import tempfile
import unittest
from compare_monitored import load_audit


class CoverageGate(unittest.TestCase):
    def test_reject_incomplete_or_contaminated_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audit = dict(sha256={}, errors=[], sampled_coverage_clear=True,
                         runs=[dict(context='short', observations=[dict(coverage=dict(sampled_coverage_clear=True))])])
            (folder / 'timing.json').write_text(json.dumps(dict(gate_passed=True)))
            (folder / 'gpu-coverage.json').write_text(json.dumps(audit))
            self.assertEqual(load_audit(folder)['runs'][0]['context'], 'short')
            for field, value in (('sampled_coverage_clear', False), ('errors', ['collector failed']), ('runs', [])):
                broken = dict(audit, **{field: value})
                (folder / 'gpu-coverage.json').write_text(json.dumps(broken))
                with self.assertRaises(ValueError):load_audit(folder)
            audit['runs'][0]['observations'][0]['coverage']['sampled_coverage_clear'] = False
            (folder / 'gpu-coverage.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):load_audit(folder)

    def test_idle_gate_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audit = dict(sha256={}, errors=[], sampled_coverage_clear=True, runs=[])
            (folder / 'timing.json').write_text(json.dumps(dict(gate_passed=False)))
            (folder / 'gpu-coverage.json').write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'Idle gate'):load_audit(folder)
            (folder / 'timing.json').write_text(json.dumps(dict(gate_passed=True, quiet_gap_seconds=2)))
            with self.assertRaisesRegex(ValueError, 'Quiet-gap'):load_audit(folder)
            (folder / 'input').write_bytes(b'changed')
            audit['sha256'] = {'input': '0' * 64}
            (folder / 'gpu-coverage.json').write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'Changed audit'):load_audit(folder)


if __name__ == '__main__':
    unittest.main()
