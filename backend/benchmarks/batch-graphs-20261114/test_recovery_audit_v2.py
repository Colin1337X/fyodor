import copy
import unittest
from audit_recovery_v2 import cases, validate


class RecoveryAudit(unittest.TestCase):
    def example(self, site='readback', cache='cold'):
        result = dict(passed=True, exit_code=0, errors=[], accepted_benchmark=False)
        graphs = 2 if cache == 'cached' or site in ('request', 'begin') else 3
        execs = 3 if cache == 'cold' and site in ('launch', 'sync', 'readback') else 2
        stdout = f'recovery=1 site={site} cache={cache} width=256 ring=0\n========= ERROR SUMMARY: 0 errors\n'
        stderr = f'INJECTED graph failure: {site}\nGRAPH resources: graphs={graphs}/{graphs} execs={execs}/{execs} live=0 errors=0\n'
        return result, stdout, stderr

    def test_inventory_and_valid_histories(self):
        matrix = list(cases())
        self.assertEqual(len(matrix), 96)
        self.assertEqual(len(set(matrix)), 96)
        for _, _, _, site, cache in matrix:
            validate(*self.example(site, cache), '256', '0', site, cache)

    def test_reject_bad_terminal_state(self):
        result, stdout, stderr = self.example()
        for key, value in [('passed', False), ('exit_code', False), ('exit_code', 99), ('errors', ['failure']), ('accepted_benchmark', True)]:
            broken = copy.deepcopy(result)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate(broken, stdout, stderr, '256', '0', 'readback', 'cold')

    def test_reject_incomplete_or_duplicate_evidence(self):
        result, stdout, stderr = self.example()
        for output, trace in [(stdout.replace('0 errors', '1 errors'), stderr),
                              (stdout + '========= ERROR SUMMARY: 0 errors\n', stderr),
                              (stdout.replace('ring=0', 'ring=1'), stderr),
                              (stdout, stderr.replace('INJECTED graph failure: readback\n', '')),
                              (stdout, stderr + 'INJECTED graph failure: readback\n'),
                              (stdout, stderr.replace('readback', 'launch'))]:
            with self.assertRaises(ValueError):
                validate(result, output, trace, '256', '0', 'readback', 'cold')

    def test_reject_bad_resource_accounting(self):
        result, stdout, stderr = self.example()
        for trace in [stderr.replace('live=0', 'live=1'), stderr.replace('errors=0', 'errors=1'),
                      stderr.replace('graphs=3/3', 'graphs=3/2'), stderr.replace('execs=3/3', 'execs=2/2'),
                      stderr.split('GRAPH resources:')[0], stderr + stderr.splitlines()[-1] + '\n']:
            with self.assertRaises(ValueError):
                validate(result, stdout, trace, '256', '0', 'readback', 'cold')


if __name__ == '__main__':
    unittest.main()
