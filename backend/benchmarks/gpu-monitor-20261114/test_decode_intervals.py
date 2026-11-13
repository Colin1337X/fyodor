import unittest
from decode_intervals import parse

GOOD = ('Generation profile: mode=ordinary prompt=5 generated=3 decode_loop_ms=125.000000\n'
        'Generation interval: mode=ordinary prompt=5 generated=3 clock=qpc start_seconds=20.000000000 end_seconds=20.125000000 valid=1\n')


class Intervals(unittest.TestCase):
    def test_complete(self):
        row = parse('unrelated diagnostic\n' + GOOD, clock='qpc')[0]
        self.assertEqual((row['start'], row['end'], row['milliseconds']), (20, 20.125, 125))

    def test_missing_or_duplicate(self):
        for text in ('', GOOD.splitlines()[0], GOOD.splitlines()[1], GOOD + GOOD.splitlines()[1]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text, clock='qpc')

    def test_call_identity(self):
        with self.assertRaises(ValueError):
            parse(GOOD.replace('Generation interval: mode=ordinary', 'Generation interval: mode=mtp'), clock='qpc')

    def test_clock(self):
        for text in (GOOD.replace('clock=qpc', 'clock=monotonic'), GOOD.replace('valid=1', 'valid=0')):
            with self.assertRaises(ValueError):
                parse(text, clock='qpc')

    def test_nonfinite_or_reversed(self):
        for value in ('nan', 'inf', '-1', '19', '20'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(GOOD.replace('20.125000000', value), clock='qpc')

    def test_duration_mismatch(self):
        with self.assertRaises(ValueError):
            parse(GOOD.replace('125.000000', '126.000000'), clock='qpc')

    def test_overlap(self):
        with self.assertRaises(ValueError):
            parse(GOOD + GOOD, clock='qpc')


if __name__ == '__main__':
    unittest.main()
