"""Synthetic evidence contract tests; these are not live benchmark results."""
import json
import unittest
from audit_generation import ORDER, audit_run, lifetime

LUID = 'luid_0x00000000_0x00013020'


def fixture():
    run = dict(context='short', process_id=7, exit_code=0, observer_clock='qpc',
               spawn_before_seconds=0.1, spawn_after_seconds=0.2, exit_observed_seconds=14.)
    stdout, stderr = [], []
    for index, (round_, mode) in enumerate(ORDER):
        start = index + 1.
        native = 'ordinary' if mode == 'ordinary' else 'mtp'
        stdout.append(json.dumps(dict(round=round_, mode=mode, prompt=128, generated=32, parity=True)))
        stderr += [f'CASE round={round_} mode={mode}',
                   f'Generation profile: mode={native} prompt=128 generated=32 decode_loop_ms=200.000000',
                   f'Generation interval: mode={native} prompt=128 generated=32 clock=qpc start_seconds={start} end_seconds={start + .2} valid=1']
    samples = [dict(kind='header', provider='PDH wildcard array', adapter_luid=LUID)]
    for index in range(15):
        samples.append(dict(kind='sample', qpc_observed_seconds=float(index), target_counters=[
            dict(instance='pid_7_' + LUID + '_engtype_compute', process_id=7, status=0, engine_percent=50)]))
    samples.append(dict(kind='end', failed=False, requested_stop=True))
    return run, '\n'.join(stdout), '\n'.join(stderr), samples


class GenerationAudit(unittest.TestCase):
    def check(self, items):
        return audit_run(*items, luid=LUID, lifetimes={7: (.1, 14.)})

    def test_complete_stays_unaccepted(self):
        result = self.check(fixture())
        self.assertTrue(result['sampled_coverage_clear'])
        self.assertFalse(result['accepted_benchmark'])
        self.assertEqual(len(result['observations']), 12)

    def test_competitor(self):
        items = list(fixture())
        counter = items[3][3]['target_counters'][0]
        counter.update(process_id=8, instance='pid_8_' + LUID + '_engtype_compute')
        self.assertFalse(self.check(items)['sampled_coverage_clear'])

    def test_activity_ending_before_decode(self):
        items = list(fixture())
        counter = items[3][2]['target_counters'][0]
        counter.update(process_id=8, instance='pid_8_' + LUID + '_engtype_compute')
        self.assertTrue(self.check(items)['sampled_coverage_clear'])

    def test_missing_monitor_end(self):
        items = list(fixture()); items[3].pop()
        self.assertFalse(self.check(items)['sampled_coverage_clear'])

    def test_missing_case(self):
        items = list(fixture()); items[2] = '\n'.join(items[2].splitlines()[:-3])
        with self.assertRaises(ValueError): self.check(items)

    def test_wrong_case_order(self):
        items = list(fixture()); items[2] = items[2].replace('CASE round=0 mode=ordinary', 'CASE round=1 mode=ordinary')
        with self.assertRaises(ValueError): self.check(items)

    def test_parity_failure(self):
        items = list(fixture()); items[1] = items[1].replace('"parity": true', '"parity": false', 1)
        with self.assertRaises(ValueError): self.check(items)

    def test_prompt_mismatch(self):
        items = list(fixture()); items[2] = items[2].replace('prompt=128', 'prompt=129')
        with self.assertRaises(ValueError): self.check(items)

    def test_unbounded_child(self):
        for field, value in [('exit_code', None), ('process_id', True), ('observer_clock', 'wall'),
                             ('spawn_before_seconds', float('nan')), ('spawn_after_seconds', 15.),
                             ('observer_error', 'snapshot failure'), ('terminated_by_observer', True)]:
            run = fixture()[0]; run[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): lifetime(run)

    def test_interval_outside_lifetime(self):
        items = list(fixture()); items[0]['exit_observed_seconds'] = 2.
        with self.assertRaises(ValueError):
            audit_run(*items, luid=LUID, lifetimes={7: (.1, 2.)})

    def test_old_duration_only_rejected(self):
        items = list(fixture()); items[2] = '\n'.join(line for line in items[2].splitlines() if not line.startswith('Generation interval:'))
        with self.assertRaises(ValueError): self.check(items)


class JointGenerationAudit(unittest.TestCase):
    def joint_fixture(self):
        run, stdout, stderr, samples = fixture()
        return run, stdout.replace('mtp-batch4', 'mtp-joint3'), stderr.replace('mtp-batch4', 'mtp-joint3'), samples

    def test_explicit_joint_inventory(self):
        result = audit_run(*self.joint_fixture(), luid=LUID, lifetimes={7: (.1, 14.)}, mode_set='joint-base')
        self.assertTrue(result['sampled_coverage_clear'])
        self.assertFalse(result['accepted_benchmark'])
        self.assertEqual(result['mode_set'], 'joint-base')
        self.assertEqual(len(result['observations']), 12)

    def test_default_rejects_joint_labels(self):
        with self.assertRaises(ValueError):
            audit_run(*self.joint_fixture(), luid=LUID, lifetimes={7: (.1, 14.)})

    def test_joint_rejects_old_labels(self):
        with self.assertRaises(ValueError):
            audit_run(*fixture(), luid=LUID, lifetimes={7: (.1, 14.)}, mode_set='joint-base')

    def test_unknown_mode_set(self):
        with self.assertRaises(ValueError):
            audit_run(*fixture(), luid=LUID, lifetimes={7: (.1, 14.)}, mode_set='automatic')


if __name__ == '__main__':
    unittest.main()
