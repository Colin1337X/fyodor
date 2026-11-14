import tempfile
import unittest
from pathlib import Path
from run_monitored import snapshot, changed_inputs


class InputIntegrity(unittest.TestCase):
    def test_mutation_preserves_original_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'candidate.exe'
            binary.write_bytes(b'validated candidate')
            before = snapshot(root, [binary])
            original = dict(before)
            self.assertEqual(changed_inputs(root, before), [])
            binary.write_bytes(b'different candidate')
            self.assertEqual(changed_inputs(root, before), ['candidate.exe'])
            self.assertEqual(before, original)

    def test_deleted_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'collector.py'
            source.write_bytes(b'original')
            before = snapshot(root, [source])
            source.unlink()
            self.assertEqual(changed_inputs(root, before), ['collector.py'])

    def test_outputs_do_not_change_input_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'proof.json'
            source.write_bytes(b'{}')
            before = snapshot(root, [source])
            (root / 'stdout').write_bytes(b'completed')
            self.assertEqual(changed_inputs(root, before), [])
            self.assertEqual(list(before), ['proof.json'])


if __name__ == '__main__':
    unittest.main()
