import unittest
from summarize_coverage import summarize
from test_audit_joint_one import one, LUID, audit_run


def fixture():
    run = audit_run(*one(), luid=LUID, lifetimes={7: (.1, 14.)}, mode_set='joint-one')
    return dict(mode_set='joint-one', errors=[], runs=[run])


class Summary(unittest.TestCase):
    def test_all_rounds_unaccepted(self):
        result = summarize(fixture())
        self.assertFalse(result['accepted_benchmark'])
        self.assertEqual(len(result['runs'][0]['cases']), 12)
        for mode in result['runs'][0]['modes'].values():
            self.assertEqual(mode, dict(rounds=3, all_rounds_mean_ms=200., sample_stdev_ms=0.))

    def test_failed_coverage_retained(self):
        audit = fixture()
        row = audit['runs'][0]['observations'][0]
        row['coverage']['sampled_coverage_clear'] = False
        row['interval']['milliseconds'] = 500.
        result = summarize(audit)['runs'][0]
        self.assertFalse(result['cases'][0]['coverage']['sampled_coverage_clear'])
        self.assertEqual(result['modes']['ordinary']['all_rounds_mean_ms'], 300.)

    def test_missing_round(self):
        audit = fixture(); audit['runs'][0]['observations'].pop()
        with self.assertRaises(ValueError): summarize(audit)

    def test_invalid_timing(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            audit = fixture(); audit['runs'][0]['observations'][0]['interval']['milliseconds'] = value
            with self.assertRaises(ValueError): summarize(audit)

    def test_incomplete_audit(self):
        audit = fixture(); audit['errors'] = ['collector failed']
        with self.assertRaises(ValueError): summarize(audit)

    def test_duplicate_context(self):
        audit = fixture(); audit['runs'] *= 2
        with self.assertRaises(ValueError): summarize(audit)


if __name__ == '__main__': unittest.main()
