import json
import unittest
from run_model_memcheck import validate

MODES = ('ordinary', 'mtp-serial2', 'mtp-batch2', 'mtp-joint3')
ROWS = [dict(round=r, mode=m, parity=True, generated=32, prompt=25)
        for r in range(3) for m in (reversed(MODES) if r == 1 else MODES)]
ZERO = '========= ERROR SUMMARY: 0 errors\n'
SELECTED = 'MTP joint base: enabled; proposal window=3\n' * 3


def output(rows):
    return '\n'.join(json.dumps(row) for row in rows) + '\n'


class Evidence(unittest.TestCase):
    def test_valid_streams(self):
        validate(0, output(ROWS) + ZERO, SELECTED, 'short')
        validate(0, output(ROWS), SELECTED + ZERO, 'short')

    def test_failed_exit(self):
        for code in (False, None, 1, 99):
            with self.assertRaises(ValueError):
                validate(code, output(ROWS) + ZERO, SELECTED, 'short')

    def test_invalid_summary(self):
        for summary in ('', ZERO * 2, ZERO.replace('0 errors', '1 errors')):
            with self.assertRaises(ValueError):
                validate(0, output(ROWS) + summary, SELECTED, 'short')

    def test_missing_reordered_or_failed_cases(self):
        for rows in (ROWS[:-1], list(reversed(ROWS)),
                     [dict(row, parity=False) for row in ROWS],
                     [dict(row, generated=31) for row in ROWS]):
            with self.assertRaises(ValueError):
                validate(0, output(rows) + ZERO, SELECTED, 'short')

    def test_wrong_context_or_dispatch(self):
        with self.assertRaises(ValueError):
            validate(0, output(ROWS) + ZERO, SELECTED, 'long')
        with self.assertRaises(ValueError):
            validate(0, output(ROWS) + ZERO, '', 'short')


if __name__ == '__main__':
    unittest.main()
