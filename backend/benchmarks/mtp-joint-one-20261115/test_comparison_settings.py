import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from comparison_settings import load_settings


class Settings(unittest.TestCase):
    def check(self, environments, change=None):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/str(i) for i in range(len(environments))]
            checked = {}
            for path, environment in zip(paths, environments):
                path.write_text(json.dumps({'environment': environment}))
                checked[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            if change == 'missing':
                checked.pop(str(paths[0].resolve()))
            if change == 'changed':
                paths[0].write_text('{}')
            return load_settings(paths, checked)

    def test_equal(self):
        self.assertEqual(self.check([{'NYA_A': '1'}]*2), {'NYA_A': '1'})

    def test_different(self):
        with self.assertRaises(ValueError):
            self.check([{'NYA_A': '1'}, {'NYA_A': '0'}])

    def test_missing_or_changed(self):
        for change in ('missing', 'changed'):
            with self.assertRaises(ValueError):
                self.check([{'NYA_A': '1'}]*2, change)

    def test_invalid(self):
        for environment in ({}, {'NYA_A': 1}):
            with self.assertRaises(ValueError):
                self.check([environment]*2)

    def test_requires_two(self):
        with self.assertRaises(ValueError):
            self.check([{'NYA_A': '1'}])


if __name__ == '__main__':
    unittest.main()
