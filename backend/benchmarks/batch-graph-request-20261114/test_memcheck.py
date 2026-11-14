import copy
import itertools
import unittest
from run_memcheck import validate


class Evidence(unittest.TestCase):
    def setUp(self):
        self.oracle = dict(passed=True,errors=[],cases=[
            dict(dtype=t,width=w,count=c,round=r,parity=True,guards=True,
                 tokens=[(r*3+q*5)%17 for q in range(4)],
                 position=(0,255,256,8191,8192,0xfffffffb)[r%6])
            for t,w,c,r in itertools.product((0,1,12,14),(256,3840),(2,4),range(12))])
        self.summary = '========= ERROR SUMMARY: 0 errors\n'

    def test_complete_sweep(self):
        validate(0,'',self.summary,self.oracle)

    def test_summary_failures(self):
        for summary in ('',self.summary*2,self.summary.replace('0 errors','1 errors')):
            with self.subTest(summary=summary), self.assertRaises(ValueError):
                validate(0,'',summary,self.oracle)

    def test_failed_exit(self):
        for code in (1,99,False,None):
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate(code,'',self.summary,self.oracle)

    def test_duplicate_case(self):
        self.oracle['cases'][1] = copy.deepcopy(self.oracle['cases'][0])
        with self.assertRaises(ValueError): validate(0,'',self.summary,self.oracle)

    def test_missing_case(self):
        self.oracle['cases'].pop()
        with self.assertRaises(ValueError): validate(0,'',self.summary,self.oracle)

    def test_bad_case_fields(self):
        for field,value in [('parity',False),('guards',False),('tokens',[0]*4),('position',999)]:
            oracle=copy.deepcopy(self.oracle)
            oracle['cases'][-1][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(0,'',self.summary,oracle)

    def test_cleanup_error(self):
        self.oracle['errors'] = ['cleanup failed']
        with self.assertRaises(ValueError): validate(0,'',self.summary,self.oracle)


if __name__ == '__main__':
    unittest.main()
