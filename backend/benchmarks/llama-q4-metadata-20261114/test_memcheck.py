"""CPU-only checks that incomplete raw sanitizer evidence cannot pass."""
import copy
import unittest
from run_memcheck import validate

def fixture():
    shapes = [(256,1),(256,7),(256,9),(3840,512),(3840,2048),(3840,4096),
              (3840,8192),(3840,15360),(4096,3840),(8192,3840),(15360,3840)]
    bits = [0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
            0x3c00,0xbc00,0x7bff,0xfbff,0x27ae,0xa75a]
    cases = [(k,r,None) for k,r in shapes]+[(256,9,[d,m]) for d in bits for m in bits]
    return dict(passed=True, cases=[dict(columns=k,rows=r,half_bits=e,byte_exact=True,
        guards_clear=True,float64_checked=k==256,maximum_fraction_of_float64_bound=0.25)
        for k,r,e in cases])

class EvidenceTests(unittest.TestCase):
    def test_complete(self):
        validate(0,'','========= ERROR SUMMARY: 0 errors\n',fixture())

    def test_terminal_failures(self):
        for status,summary in [(99,'0'),(0,'1'),(0,None),(False,'0')]:
            with self.subTest(status=status,summary=summary), self.assertRaises(ValueError):
                validate(status,'','' if summary is None else f'========= ERROR SUMMARY: {summary} errors\n',fixture())

    def test_duplicate_summary(self):
        with self.assertRaises(ValueError):
            validate(0,'========= ERROR SUMMARY: 0 errors\n','========= ERROR SUMMARY: 0 errors\n',fixture())

    def test_duplicate_case(self):
        oracle=fixture(); oracle['cases'][-1]=copy.deepcopy(oracle['cases'][-2])
        with self.assertRaises(ValueError):
            validate(0,'','========= ERROR SUMMARY: 0 errors\n',oracle)

    def test_missing_or_failed_bound(self):
        for field,value in [('float64_checked',False),('maximum_fraction_of_float64_bound',None),
                            ('maximum_fraction_of_float64_bound',float('nan')),
                            ('maximum_fraction_of_float64_bound',1.01)]:
            oracle=fixture(); oracle['cases'][0][field]=value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                validate(0,'','========= ERROR SUMMARY: 0 errors\n',oracle)

    def test_failed_guard(self):
        oracle=fixture(); oracle['cases'][0]['guards_clear']=False
        with self.assertRaises(ValueError):
            validate(0,'','========= ERROR SUMMARY: 0 errors\n',oracle)

if __name__ == '__main__':
    unittest.main()
