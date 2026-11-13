"""Collector death must interrupt polling without serializing process handles."""
import json
import subprocess
import sys
import unittest
from unittest.mock import Mock
from monitor_capture import MonitorState

class MonitorHealthTests(unittest.TestCase):
    def test_running_is_serializable(self):
        state = MonitorState(ready=True)
        state.worker = Mock()
        state.worker.poll.return_value = None
        state.check_alive()
        self.assertEqual(json.loads(json.dumps(state)), {'ready': True})

    def test_exit_zero_also_rejected(self):
        for status in (0, 1, -1):
            with self.subTest(status=status):
                state = MonitorState()
                state.worker = Mock()
                state.worker.poll.return_value = status
                with self.assertRaisesRegex(RuntimeError, 'exited during workload'):
                    state.check_alive()

    def test_missing_worker(self):
        with self.assertRaisesRegex(RuntimeError, 'no collector'):
            MonitorState().check_alive()

    def test_real_child_stopped_on_dead_monitor(self):
        state = MonitorState()
        state.worker = subprocess.Popen([sys.executable, '-c', 'pass'])
        state.worker.wait(timeout=10)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            with self.assertRaisesRegex(RuntimeError, 'exited during workload'):
                while child.poll() is None:
                    state.check_alive()
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
        self.assertIsNotNone(child.returncode)

if __name__ == '__main__':
    unittest.main()
