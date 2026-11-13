"""Reject incomplete/failed sanitizer evidence; no GPU work."""
import unittest
from audit_memcheck import check_logs

META = dict(exit_code=0, memcheck=True, prefix=513, ring=1, count=4, aligned_admitted=True)
OUT = ('reset=0 prefix=513 tokens=4 max_scaled=0.00001 execution=resident\n'
       'reset=1 prefix=513 tokens=4 max_scaled=0.00001 execution=resident\n'
       '========= ERROR SUMMARY: 0 errors\n')
ERR = ('CUDA aligned prefix: partial_queries=512 FDPP_batch_calls=40\n'
       'CUDA batched split attention calls: 48\n') * 2


class Audit(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(check_logs(META, OUT, ERR), [.00001, .00001])

    def test_no_terminal_success(self):
        for code in (None, 1, 99, False):
            with self.subTest(code=code), self.assertRaises(ValueError):
                check_logs(dict(META, exit_code=code), OUT, ERR)

    def test_bad_summary(self):
        for text in (OUT.split('=========')[0], OUT.replace('0 errors', '1 errors'), OUT + '========= ERROR SUMMARY: 0 errors\n'):
            with self.assertRaises(ValueError): check_logs(META, text, ERR)

    def test_incomplete_lifecycle(self):
        with self.assertRaises(ValueError): check_logs(META, OUT.replace('reset=1', 'reset=0'), ERR)

    def test_bad_numerics(self):
        for value in ('nan', 'inf', '-1', '0.0005001'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_logs(META, OUT.replace('0.00001', value), ERR)

    def test_dispatch_missing(self):
        with self.assertRaises(ValueError): check_logs(META, OUT, '')


if __name__ == '__main__':
    unittest.main()
