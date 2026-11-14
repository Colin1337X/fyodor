import unittest
from audit_joint_one import audit_run
from test_audit_generation import fixture,LUID

def one():
 r,out,err,samples=fixture()
 return [r,out.replace('mtp-batch4','mtp-joint1'),err.replace('mtp-batch4','mtp-joint1'),samples]

class Inventory(unittest.TestCase):
 def audit(self,items):return audit_run(*items,luid=LUID,lifetimes={7:(.1,14.)},mode_set='joint-one')
 def test_inventory(self):
  r=self.audit(one());self.assertTrue(r['sampled_coverage_clear']);self.assertFalse(r['accepted_benchmark']);self.assertEqual(len(r['observations']),12)
 def test_other_labels(self):
  for label in ('mtp-batch4','mtp-joint3'):
   a=one();a[1]=a[1].replace('mtp-joint1',label);a[2]=a[2].replace('mtp-joint1',label)
   with self.assertRaises(ValueError):self.audit(a)
 def test_default_rejects_one(self):
  with self.assertRaises(ValueError):audit_run(*one(),luid=LUID,lifetimes={7:(.1,14.)})
 def test_incomplete(self):
  a=one();a[1]='\n'.join(a[1].splitlines()[:-1])
  with self.assertRaises(ValueError):self.audit(a)
 def test_system_activity(self):
  a=one();c=a[3][3]['target_counters'][0];c.update(process_id=4,instance='pid_4_'+LUID+'_engtype_copy')
  self.assertFalse(self.audit(a)['sampled_coverage_clear'])
 def test_invalid_counter(self):
  a=one();a[3][3]['target_counters'][0]['status']=0xc0000bba
  self.assertFalse(self.audit(a)['sampled_coverage_clear'])
 def test_missing_end(self):
  a=one();a[3].pop();self.assertFalse(self.audit(a)['sampled_coverage_clear'])
if __name__=='__main__':unittest.main()
