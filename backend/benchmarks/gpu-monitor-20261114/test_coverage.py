"""Offline failure cases for sampled coverage; no GPU or acceptance claims."""
import unittest
from coverage import audit

LUID='luid_0x00000000_0x00013020'
def fixture():
    return [{'kind':'header','provider':'PDH wildcard array','adapter_luid':LUID}]+[
        {'kind':'sample','qpc_observed_seconds':t,'target_counters':[
            {'instance':'pid_7_'+LUID+'_engtype_compute','process_id':7,'status':0,'engine_percent':20}]} for t in (0.,1.,2.,3.)
    ]+[{'kind':'end','failed':False,'requested_stop':True}]

class Coverage(unittest.TestCase):
    def check_rows(self,rows,clear=False,**options):
        result=audit(rows,luid=LUID,start=.5,end=2.5,lifetimes={7:(.2,2.8)},**options)
        self.assertEqual(result['sampled_coverage_clear'],clear)
        self.assertFalse(result['accepted_benchmark'])
    def test_complete(self):self.check_rows(fixture(),True)
    def test_missing_end(self):self.check_rows(fixture()[:-1])
    def test_failed_end(self):
        rows=fixture();rows[-1]['failed']=True;self.check_rows(rows)
    def test_missing_header(self):self.check_rows(fixture()[1:])
    def test_missing_bracket(self):self.check_rows(fixture()[:1]+fixture()[2:])
    def test_gap(self):self.check_rows(fixture(),max_gap=.9)
    def test_invalid(self):
        for value in (None,float('nan'),float('inf'),-1,100.000001,133.31,True):
            rows=fixture();rows[2]['target_counters'][0]['engine_percent']=value
            self.check_rows(rows)
    def test_percent_boundaries(self):
        for value in (0,100):
            rows=fixture();rows[2]['target_counters'][0]['engine_percent']=value
            self.check_rows(rows,True)
    def test_status(self):
        rows=fixture();rows[2]['target_counters'][0]['status']=0x800007d5;self.check_rows(rows)
    def test_competitor(self):
        rows=fixture();counter=rows[2]['target_counters'][0]
        counter['process_id']=4;counter['instance']=counter['instance'].replace('pid_7_','pid_4_')
        self.check_rows(rows)
    def test_empty(self):
        rows=fixture();rows[2]['target_counters']=[];self.check_rows(rows)
    def test_wrong_adapter(self):
        rows=fixture();rows[2]['target_counters'][0]['instance']='other';self.check_rows(rows)
    def test_nonmonotonic(self):
        rows=fixture();rows[3]['qpc_observed_seconds']=1.;self.check_rows(rows)
    def test_lifetime(self):
        result=audit(fixture(),luid=LUID,start=.5,end=2.5,lifetimes={7:(4.,5.)})
        self.assertIn('unattributed GPU activity',result['errors'])
    def test_collector_error(self):
        rows=fixture();rows.insert(-1,{'kind':'error'});self.check_rows(rows)

if __name__=='__main__':unittest.main()
