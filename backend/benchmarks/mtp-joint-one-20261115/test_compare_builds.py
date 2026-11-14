import unittest
from compare_builds import validate,OLD,NEW,BASE,CANDIDATE,BASE_WORK,CANDIDATE_WORK
class Checks(unittest.TestCase):
 def args(self):return [OLD,NEW,'mode=mtp-joint3','mode=mtp-joint1',[[BASE_WORK,BASE,'-O3']],[[CANDIDATE_WORK,CANDIDATE,'-O3']],'archive','archive']
 def test_expected_pair(self):validate(*self.args())
 def test_extra_core_change(self):
  a=self.args();a[1]+=' extra'
  with self.assertRaises(ValueError):validate(*a)
 def test_duplicate_assignment(self):
  a=self.args();a[0]+=OLD
  with self.assertRaises(ValueError):validate(*a)
 def test_workload_change(self):
  a=self.args();a[3]+=' changed prompt'
  with self.assertRaises(ValueError):validate(*a)
 def test_compiler_flags(self):
  a=self.args();a[5][0][-1]='-O0'
  with self.assertRaises(ValueError):validate(*a)
 def test_archive(self):
  a=self.args();a[-1]='other archive'
  with self.assertRaises(ValueError):validate(*a)
if __name__=='__main__':unittest.main()
