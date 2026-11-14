import unittest
from classify_invalid import classify
LUID='luid_0x00000000_0x00012670'
NAME='pid_42_'+LUID+'_phys_0_eng_10_engtype_cuda'

def row(index, names, invalid=()):
    return dict(index=index,collect_status=0,
                raw=dict(rows=[dict(instance=n,status=0) for n in names]),
                formatted=dict(rows=[dict(instance=n,status=0xc0000bba if n in invalid else 0) for n in names]))

class HistoryTests(unittest.TestCase):
    def test_new_then_persistent(self):
        result=classify([row(0,[]),row(1,[NAME],[NAME]),row(2,[NAME],[NAME])],LUID)
        self.assertEqual(result['history_counts'],dict(new=1,present=1,unknown=0))
        self.assertFalse(result['accepted_benchmark'])
    def test_first_sample_unknown(self):
        self.assertEqual(classify([row(0,[NAME],[NAME])],LUID)['history_counts']['unknown'],1)
    def test_failed_collection_resets_history(self):
        rows=[row(0,[NAME]),dict(index=1,collect_status=1),row(2,[NAME],[NAME])]
        result=classify(rows,LUID)
        self.assertEqual(result['api_failure_indices'],[1])
        self.assertEqual(result['history_counts']['unknown'],1)
    def test_missing_raw_not_new(self):
        rows=[row(0,[NAME]),dict(index=1,formatted=dict(rows=[])),row(2,[NAME],[NAME])]
        self.assertEqual(classify(rows,LUID)['history_counts']['unknown'],1)
    def test_wrong_adapter_excluded(self):
        other=NAME.replace('00012670','00012671')
        self.assertEqual(classify([row(0,[other],[other])],LUID)['invalid'],[])
    def test_reordered_rejected(self):
        with self.assertRaises(ValueError):classify([row(1,[])],LUID)

if __name__=='__main__':unittest.main()
