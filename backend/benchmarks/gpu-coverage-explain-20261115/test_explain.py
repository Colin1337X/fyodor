import unittest
from explain import explain


class ExplanationTest(unittest.TestCase):
    def test_stop_and_error_scope_are_separate(self):
        observation = dict(round=0, mode='ordinary', interval=dict(start=10, end=11),
            coverage=dict(sampled_coverage_clear=False,
                errors=['collector error', 'missing successful cooperative termination', 'unattributed GPU activity'],
                positive_engines=[dict(process_id=4, known_workload=False)]))
        errors = [dict(qpc_before_collect=20, qpc_observed_seconds=21)]
        result = explain(observation, errors, dict(kind='end', requested_stop=True, failed=True))
        self.assertTrue(result['requested_stop_record_present'])
        self.assertEqual(result['collector_errors_overlapping_interval'], [])
        self.assertEqual(result['capture_wide_error_count'], 1)
        self.assertEqual(result['unattributed_process_ids'], [4])
        self.assertFalse(result['sampled_coverage_clear'])
        self.assertFalse(result['accepted_benchmark'])
        self.assertIn('retained a failure flag', result['telemetry_reasons'][-1])
        errors.append(dict(qpc_before_collect=9, qpc_observed_seconds=10.5))
        self.assertEqual(len(explain(observation, errors, {})['collector_errors_overlapping_interval']), 1)
        self.assertIn('missing requested-stop end record', explain(observation, errors, {})['telemetry_reasons'])


if __name__ == '__main__':
    unittest.main()
