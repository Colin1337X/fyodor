import copy
import unittest
from compare_generation import ORDER, compare, native_text, statistics_ms


class Comparison(unittest.TestCase):
    def setUp(self):
        self.native = [dict(round=r,mode=m,interval=dict(prompt=25,generated=32,milliseconds=500 if r else 9999)) for r,m in ORDER]
        self.reference = [dict(round=r,prompt=25,generated=32,eos=False,text_hex='410a',decode_ms=625 if r else 1) for r in range(3)]

    def test_warmup_excluded_and_ratio_direction(self):
        result=compare(self.native,self.reference,b'A\n')
        self.assertEqual(result['llama']['mean_ms'],625)
        for cell in result['fyodor'].values():
            self.assertEqual(cell['diagnostic_speed_percent_of_llama'],125)
            self.assertEqual(cell['stdev_ms'],0)

    def test_text_record_preserves_trailing_newline(self):
        self.assertEqual(native_text('{}\nTEXT A\n\n{"round":1}\n'),b'A\n')
        for value in ('TEXT A', '{}\nTEXT A', '{}\nTEXT A\nTEXT B\n{"round":1}'):
            with self.assertRaises(ValueError): native_text(value)

    def test_workload_mismatch(self):
        for key,value in [('prompt',26),('generated',31),('eos',True),('text_hex','42')]:
            reference=copy.deepcopy(self.reference); reference[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): compare(self.native,reference,b'A\n')
        with self.assertRaises(ValueError): compare(self.native[:-1],self.reference,b'A\n')
        with self.assertRaises(ValueError): compare(self.native,self.reference[::-1],b'A\n')

    def test_invalid_times(self):
        for values in ([1], [1,0], [1,float('nan')], [1,float('inf')], [1,True]):
            with self.assertRaises(ValueError): statistics_ms(values)

    def test_joint_mode_requires_explicit_selection(self):
        joint=copy.deepcopy(self.native)
        for row in joint:
            if row['mode']=='mtp-batch4':row['mode']='mtp-joint3'
        result=compare(joint,self.reference,b'A\n',mode_set='joint-base')
        self.assertEqual(result['mode_set'],'joint-base')
        self.assertEqual(result['fyodor']['mtp-joint3']['diagnostic_speed_percent_of_llama'],125)
        self.assertNotIn('mtp-batch4',result['fyodor'])
        with self.assertRaises(ValueError):compare(joint,self.reference,b'A\n')
        with self.assertRaises(ValueError):compare(self.native,self.reference,b'A\n',mode_set='joint-base')
        with self.assertRaises(ValueError):compare(joint,self.reference,b'A\n',mode_set='unknown')


if __name__ == '__main__': unittest.main()
