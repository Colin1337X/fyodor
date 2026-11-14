import struct,unittest
from correlate_decode import correlate
L='luid_0x00000000_0x00012670'
def sample(i,t,pid=4,value=1,status=0):
 return dict(index=i,qpc_start=t,qpc_end=t+.01,collect_status=0,formatted=dict(rows=[dict(instance=f'pid_{pid}_{L}_phys_0_eng_13_engtype_copy',status=status,value_bits=struct.pack('<d',value).hex())]))
class Tests(unittest.TestCase):
 def test_boundaries(self):
  r=correlate([sample(0,1),sample(1,2),sample(2,3)],[dict(start=2.01,end=2.9)],L,42)[0]
  self.assertEqual(r['sample_indices'],[2]);self.assertEqual(r['events'][0]['kind'],'system_positive');self.assertTrue(r['bracketed'])
 def test_owned_above100_remains(self):
  r=correlate([sample(0,1),sample(1,2,42,125)],[dict(start=1.2,end=1.8)],L,42)[0]
  self.assertEqual(r['events'][0]['kind'],'above_100')
 def test_invalid_zero_not_idle(self):
  r=correlate([sample(0,1),sample(1,2,42,0,0xc0000bba)],[dict(start=1.2,end=1.8)],L,42)[0]
  self.assertEqual(r['events'][0]['kind'],'invalid')
 def test_gap_and_incomplete(self):
  r=correlate([sample(0,1),sample(1,4)],[dict(start=.5,end=5)],L,42)[0]
  self.assertEqual(r['gap_indices'],[1]);self.assertFalse(r['bracketed'])
 def test_reordered(self):
  with self.assertRaises(ValueError):correlate([sample(1,1)],[],L,42)
if __name__=='__main__':unittest.main()
