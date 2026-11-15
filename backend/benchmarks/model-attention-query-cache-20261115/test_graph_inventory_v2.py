import unittest
from run_monitored_v2 import graph_inventory

class GraphInventoryTests(unittest.TestCase):
    def test_exact_inventory(self):
        line = 'CUDA batched graph captures: 1 replays: 2'
        good = '\n'.join([line] * 6)
        cases = [(good, True), ('unrelated log\n' + good, True), ('', False),
                 ('\n'.join([line] * 5), False), (good + '\n' + line, False)]
        for bad in ('0 replays: 2', '1 replays: 1', 'invalid replays: 2',
                    '1 replays: 2 trailing', '-1 replays: 2'):
            malformed = 'CUDA batched graph captures: ' + bad
            cases.extend([(good + '\n' + malformed, False),
                          (good.replace(line, malformed, 1), False)])
        cases.append((good.replace(line, 'prefix ' + line, 1), False))
        for trace, expected in cases:
            with self.subTest(trace=trace):
                self.assertEqual(graph_inventory(trace)[1], expected)

if __name__ == '__main__':
    unittest.main(verbosity=2)
