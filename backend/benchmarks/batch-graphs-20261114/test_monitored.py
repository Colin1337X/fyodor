import tempfile
import unittest
from pathlib import Path
from run_monitored import snapshot, changed_inputs, graph_inventory


class InputIntegrity(unittest.TestCase):
    def test_graph_capture_and_reuse_required(self):
        line='CUDA batched graph captures: 2 replays: 10\n'
        zero='CUDA batched graph captures: 0 replays: 0\n'
        self.assertTrue(graph_inventory(line*6+zero*6)[1])
        for trace in ('',line*5,line*7,line.replace('10','2')*6,zero*12):
            with self.subTest(trace=trace):self.assertFalse(graph_inventory(trace)[1])
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
