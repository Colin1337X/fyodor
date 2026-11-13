import unittest
from summarize import PHASES,summarize

def fixture():
    phases=[dict(phase=name,start=1+5*i,end=6+5*i) for i,name in enumerate(PHASES)]
    rows=[dict(kind='header')]+[dict(kind='sample',qpc_observed_seconds=i+.5,target_counters=[dict(status=0,engine_percent=3.,process_id=4,instance='system-copy')]) for i in range(37)]
    rows.append(dict(kind='end',requested_stop=True,failed=False))
    return rows,phases
class Controls(unittest.TestCase):
    def test_boundaries_and_unattributed_system(self):
        rows,phases=fixture();result=summarize(rows,phases,100)
        self.assertFalse(result['accepted_benchmark']);self.assertFalse(result['system_attribution_proven'])
        for p in result['phases']:
            self.assertEqual(len(p['intervals']),4);self.assertFalse(p['errors'])
            self.assertTrue(all(e['category']=='System-unattributed' for e in p['positive_engines']))
    def test_invalid_counter_never_hidden_as_clear(self):
        for value in (True,101.,-1.,float('nan'),None):
            rows,phases=fixture();rows[3]['target_counters'][0]['engine_percent']=value
            self.assertIn('invalid target counter',summarize(rows,phases,100)['phases'][0]['errors'])
    def test_dead_collector_rejected(self):
        rows,phases=fixture();rows[-1]['failed']=True
        with self.assertRaises(ValueError):summarize(rows,phases,100)
    def test_bad_bounds_and_clocks(self):
        rows,phases=fixture();phases[1]['start']=0
        with self.assertRaises(ValueError):summarize(rows,phases,100)
        rows,phases=fixture();rows[3]['qpc_observed_seconds']=0
        with self.assertRaises(ValueError):summarize(rows,phases,100)
if __name__=='__main__':unittest.main()
