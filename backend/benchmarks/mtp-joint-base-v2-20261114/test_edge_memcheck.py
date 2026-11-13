import unittest
from run_edge_memcheck import validate
GOOD='Joint edge cases: 22; matched_errors=5; eos=1; context=2\n'
SELECTED='MTP joint base: enabled\n'
ZERO='========= ERROR SUMMARY: 0 errors\n'
class Evidence(unittest.TestCase):
    def test_summary_streams(self):
        validate(0,GOOD+ZERO,SELECTED);validate(0,GOOD,SELECTED+ZERO)
    def test_bad_exit(self):
        for code in (False,None,1,99):
            with self.assertRaises(ValueError):validate(code,GOOD+ZERO,SELECTED)
    def test_summaries(self):
        for summary in ('',ZERO+ZERO,'========= ERROR SUMMARY: 1 errors\n'):
            with self.assertRaises(ValueError):validate(0,GOOD+summary,SELECTED)
    def test_edge_coverage(self):
        for report in ('',GOOD+GOOD,GOOD.replace('22','21'),GOOD.replace('eos=1','eos=0')):
            with self.assertRaises(ValueError):validate(0,report+ZERO,SELECTED)
    def test_missing_dispatch(self):
        with self.assertRaises(ValueError):validate(0,GOOD+ZERO,'')
if __name__=='__main__':unittest.main()
