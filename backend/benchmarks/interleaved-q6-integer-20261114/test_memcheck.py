"""Offline checks of terminal sanitizer evidence, without loading CUDA."""
import unittest
from run_memcheck import validate

class Evidence(unittest.TestCase):
    def oracle(self):
        return dict(passed=True,cases=[dict(byte_exact=True,guards_clear=True) for _ in range(712)])
    def test_both_streams(self):
        for stdout,stderr in [('========= ERROR SUMMARY: 0 errors\n',''),('', '========= ERROR SUMMARY: 0 errors\n')]:
            validate(0,stdout,stderr,self.oracle())
    def test_exit(self):
        for code in (99,1,None,False):
            with self.assertRaises(ValueError):validate(code,'========= ERROR SUMMARY: 0 errors','',self.oracle())
    def test_summary(self):
        for summary in ('','========= ERROR SUMMARY: 1 errors','========= ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 0 errors'):
            with self.assertRaises(ValueError):validate(0,summary,'',self.oracle())
    def test_incomplete(self):
        oracle=self.oracle();oracle['cases'].pop()
        with self.assertRaises(ValueError):validate(0,'========= ERROR SUMMARY: 0 errors','',oracle)
    def test_parity(self):
        oracle=self.oracle();oracle['cases'][9]['guards_clear']=False
        with self.assertRaises(ValueError):validate(0,'========= ERROR SUMMARY: 0 errors','',oracle)

if __name__=='__main__':unittest.main()
