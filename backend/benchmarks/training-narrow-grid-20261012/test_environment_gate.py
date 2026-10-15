"""Exercise desktop-gate acceptance and rejection without accessing the GPU."""
import importlib.util
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parent.parent / 'score-refresh-20261011/environment.py'
spec = importlib.util.spec_from_file_location('environment_gate', path)
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


class DesktopGate(unittest.TestCase):
    def run_gate(self, utilization, powers=None, window=True, competitor=False):
        environment.record = {}
        environment.desktop_window = window
        values = itertools.cycle(utilization)
        power = itertools.cycle(powers or [7])
        def sample():
            return {'utilization': next(values), 'power': f'{next(power)} W'}
        with patch.object(environment, 'snapshot', side_effect=sample), \
                patch.object(environment, 'competitors', return_value=['other'] if competitor else []), \
                patch.object(environment, 'save'), patch.object(environment.time, 'sleep'):
            return environment.idle()

    def test_original_protocol_unchanged(self):
        self.assertEqual(len(self.run_gate([10], window=False)), 3)
        with self.assertRaises(RuntimeError):
            self.run_gate([15, 38], window=False)

    def test_complete_window_required(self):
        self.assertEqual(len(self.run_gate([15, 45])), 10)

    def test_average_limit(self):
        with self.assertRaises(RuntimeError):
            self.run_gate([31])

    def test_burst_limit(self):
        with self.assertRaises(RuntimeError):
            self.run_gate([15] * 9 + [46])

    def test_every_power_sample_checked(self):
        with self.assertRaises(RuntimeError):
            self.run_gate([15], [7] * 9 + [15.1])

    def test_competitor_always_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Competing GPU process'):
            self.run_gate([0], competitor=True)


if __name__ == '__main__':
    unittest.main()
