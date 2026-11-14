import copy
import unittest
from analyze import analyze


def fixture():
    rows = [dict(kind='header', samples=3)]
    for i, value in enumerate((10, 20, 30)):
        rows.append(dict(kind='sample', index=i, before=float(i), after=i+.1,
            counters=[dict(formatted=dict(instance='test', status=0, engine_percent=1),
                           raw=dict(status=0, first=value, second=100+i))]))
    return rows + [dict(kind='end', failed=False, changed_inputs=[])]


class CounterAnalysis(unittest.TestCase):
    def test_increasing(self):
        result = analyze(fixture())
        self.assertEqual(result['raw_decreases'], [])
        self.assertFalse(result['accepted_benchmark'])

    def test_decrease_retains_both_samples(self):
        rows = fixture()
        rows[2]['counters'][0]['raw']['first'] = 1
        result = analyze(rows)
        self.assertEqual(len(result['raw_decreases']), 1)
        self.assertEqual(result['raw_decreases'][0]['previous_raw']['first'], 10)

    def test_missing_or_invalid_raw_not_compared(self):
        for missing in (True, False):
            rows = fixture()
            rows[3]['counters'][0]['raw']['first'] = 1
            if missing:
                rows[2]['counters'] = []
            else:
                rows[2]['counters'][0]['raw']['status'] = 0xC0000BBA
            self.assertEqual(analyze(rows)['raw_decreases'], [])

    def test_invalid_formatted_retained(self):
        rows = fixture()
        rows[2]['counters'][0]['formatted'].update(status=0xC0000BBA, engine_percent=None)
        result = analyze(rows)
        self.assertEqual(len(result['invalid_formatted']), 1)
        self.assertEqual(result['positive_samples']['test'], 2)

    def test_incomplete_failed_or_changed_capture_rejected(self):
        original = fixture()
        variants = [original[:-1], original[:2]+original[3:]]
        for field, value in [('failed', True), ('changed_inputs', ['source'])]:
            rows = copy.deepcopy(original)
            rows[-1][field] = value
            variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                analyze(rows)


if __name__ == '__main__':
    unittest.main()
