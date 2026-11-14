"""Exercise the real collector with a deterministic fake PDH provider, no GPU."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from recovery import Recovery, TRANSIENT

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original_coverage', HERE.parent / 'gpu-monitor-20261114/coverage.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)
LUID = 'luid_0x00000000_0x00012670'


class RecoveryTests(unittest.TestCase):
    def capture(self, outcomes, close_error=False):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'samples.jsonl'
            stop = Path(temp) / 'stop'
            pending = iter(outcomes)
            closed = []
            ticks = iter(range(1000))

            class Counter:
                def collect(self):
                    value = next(pending)
                    if isinstance(value, Exception):
                        raise value
                    if value == 'stop':
                        stop.touch()
                    return [dict(instance=f'pid_123_{LUID}_phys_0_eng_0_engtype_Compute',
                                 status=0, engine_percent=50)]

                def close(self):
                    closed.append(True)
                    if close_error:
                        raise RuntimeError('close failure')

            provider = types.ModuleType('pdh_counter')
            provider.Counter = Counter
            args = ['capture_pdh.py', '--output', str(output), '--stop', str(stop),
                    '--luid', LUID, '--samples', '20']
            exit_code = 0
            with patch.dict(sys.modules, {'pdh_counter': provider}), patch.object(sys, 'argv', args), \
                 patch('time.sleep'), patch('time.perf_counter', side_effect=lambda: next(ticks)), \
                 patch.object(sys, 'path', list(sys.path)):
                try:
                    runpy.run_path(str(HERE / 'capture_pdh.py'), run_name='__main__')
                except SystemExit as error:
                    exit_code = error.code
            self.assertEqual(closed, [True])
            return exit_code, [json.loads(line) for line in output.read_text().splitlines()]

    def test_clean_cooperative_stop(self):
        code, rows = self.capture(['ok', 'stop'])
        self.assertEqual(code, 0)
        self.assertFalse(rows[-1]['failed'])
        self.assertTrue(rows[-1]['requested_stop'])

    def test_recovered_capture_stays_rejected(self):
        code, rows = self.capture(['ok', RuntimeError(TRANSIENT), 'ok', 'stop'])
        self.assertEqual(code, 0)
        self.assertTrue(rows[-1]['failed'])
        self.assertTrue(rows[-1]['requested_stop'])
        errors = [row for row in rows if row['kind'] == 'error']
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['index'], 2)
        self.assertLess(errors[0]['qpc_before_collect'], errors[0]['qpc_observed_seconds'])
        self.assertFalse(errors[0]['accepted_benchmark'])
        samples = [row for row in rows if row['kind'] == 'sample']
        # Bracket the interval and tolerate gaps to isolate persistent error rejection.
        result = coverage.audit(rows, luid=LUID, start=samples[0]['qpc_observed_seconds'],
                                end=samples[-1]['qpc_observed_seconds'],
                                lifetimes={123: (0, 1000)}, max_gap=100)
        self.assertFalse(result['sampled_coverage_clear'])
        self.assertEqual(result['errors'], ['collector error', 'missing successful cooperative termination'])

    def test_third_consecutive_error_is_fatal(self):
        code, rows = self.capture([RuntimeError(TRANSIENT) for _ in range(3)])
        self.assertEqual(code, 1)
        self.assertEqual(rows[-1]['samples'], 3)
        self.assertFalse(rows[-1]['requested_stop'])
        self.assertTrue(rows[-1]['failed'])
        self.assertIn('Three consecutive', rows[-2]['message'])

    def test_success_resets_consecutive_errors(self):
        code, rows = self.capture([RuntimeError(TRANSIENT), RuntimeError(TRANSIENT),
                                  'ok', RuntimeError(TRANSIENT), 'stop'])
        self.assertEqual(code, 0)
        self.assertEqual([r['recovery_attempt'] for r in rows if r['kind'] == 'error'], [1, 2, 1])
        self.assertTrue(rows[-1]['failed'])

    def test_unknown_error_is_immediately_fatal(self):
        code, rows = self.capture([RuntimeError('different PDH failure')])
        self.assertEqual(code, 1)
        self.assertEqual(rows[-1]['samples'], 1)
        self.assertTrue(rows[-1]['failed'])

    def test_close_failure_is_fatal(self):
        code, rows = self.capture(['ok', 'stop'], close_error=True)
        self.assertEqual(code, 1)
        self.assertTrue(rows[-1]['failed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
