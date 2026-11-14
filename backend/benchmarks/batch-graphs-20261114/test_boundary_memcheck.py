import unittest
from run_boundary_memcheck import validate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.output = '\n'.join(f'reset={r} position={p} count={c} trial={t} parity=1'
            for r in range(2) for p,c in ((250,4),(254,2),(256,2),(258,4),(8190,2),(8192,4),(8196,2))
            for t in range(3))
        self.trace = 'CUDA batched graph captures: 6 replays: 42\n========= ERROR SUMMARY: 0 errors\n'

    def test_complete(self):
        validate(0,self.output,self.trace)

    def test_sanitizer_failures(self):
        for trace in ('',self.trace*2,self.trace.replace('0 errors','1 errors')):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                validate(0,self.output,trace)
        with self.assertRaises(ValueError):
            validate(99,self.output,self.trace)

    def test_case_inventory(self):
        rows = self.output.splitlines()
        for output in ('\n'.join(rows[:-1]),'\n'.join(reversed(rows)),self.output.replace('parity=1','parity=0',1)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate(0,output,self.trace)

    def test_graph_reuse(self):
        for trace in (self.trace.replace('6 replays','7 replays'),self.trace.replace('42','41')):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                validate(0,self.output,trace)


if __name__ == '__main__':
    unittest.main()
