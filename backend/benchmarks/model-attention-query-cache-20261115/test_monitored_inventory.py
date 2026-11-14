import unittest
from run_monitored import reuse_inventory

PREFIX='CUDA cached query capture/direct calls:'

class InventoryTests(unittest.TestCase):
    def test_six_positive_records_required_when_enabled(self):
        good='\n'.join(PREFIX+' '+str(n) for n in (48,56,56,48,48,56))
        cases=[(good,True,True),(good,False,False),('',True,False),('',False,True),
               ('\n'.join(good.splitlines()[:-1]),True,False),
               (good+'\n'+PREFIX+' 48',True,False),
               (good.replace(' 48',' 0',1),True,False),
               (good.replace(' 48',' invalid',1),True,False),
               (good.replace(' 48',' 48 trailing',1),True,False)]
        for trace,enabled,expected in cases:
            with self.subTest(trace=trace,enabled=enabled):
                _,valid=reuse_inventory(trace,enabled,PREFIX,expected_count=6)
                self.assertEqual(valid,expected)

if __name__=='__main__':
    unittest.main(verbosity=2)
