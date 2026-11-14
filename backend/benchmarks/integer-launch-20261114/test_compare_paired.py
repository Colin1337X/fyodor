import copy
import unittest
from compare_paired import aggregate

MODES=('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')

def fixture():
    cells=[]
    for i,engine in enumerate(('fyodor','llama','llama','fyodor')):
        if engine=='fyodor':
            rows=[dict(round=r,mode=m,interval=dict(prompt=25,generated=32,milliseconds=10.0))
                  for r in range(3) for m in (reversed(MODES) if r==1 else MODES)]
        else:
            rows=[dict(round=r,prompt=25,generated=32,eos=False,text_hex=b'output'.hex(),decode_ms=20.0) for r in range(3)]
        cells.append(dict(engine=engine,identity={'model':'same'},build={'engine':engine},text=b'output' if engine=='fyodor' else None,
            audit=dict(runs=[dict(context='short',observations=rows)]),
            timing=dict(runs=[dict(process_id=i+1,exit_code=0,observer_clock='qpc',
                spawn_before_seconds=float(i*2),spawn_after_seconds=float(i*2)+.1,exit_observed_seconds=float(i*2+1))])))
    return cells

class PairedEvidence(unittest.TestCase):
    def test_four_warmed_observations(self):
        result=aggregate(fixture())
        self.assertEqual(result['llama']['observations_ms'],[20.0]*4)
        for mode in MODES:
            self.assertEqual(result['fyodor'][mode]['diagnostic_speed_percent_of_llama'],200.0)
            self.assertEqual(result['fyodor'][mode]['sample_stddev_ms'],0)

    def test_reject_order_overlap_and_failed_process(self):
        for mutation in ('order','overlap','exit'):
            cells=fixture()
            if mutation=='order':cells.reverse()
            elif mutation=='overlap':cells[1]['timing']['runs'][0]['spawn_before_seconds']=0.5
            else:cells[0]['timing']['runs'][0]['exit_code']=1
            with self.assertRaises(ValueError):aggregate(cells)

    def test_reject_changed_model_build_context_and_output(self):
        original=fixture()
        for key,value in [('identity',{'model':'other'}),('build',{'engine':'different'}),('text',b'changed')]:
            cells=copy.deepcopy(original);cells[3][key]=value
            with self.assertRaises(ValueError):aggregate(cells)
        cells=fixture();cells[3]['audit']['runs'][0]['context']='long'
        with self.assertRaises(ValueError):aggregate(cells)

    def test_reject_mismatched_outputs_and_workloads(self):
        for key,value in [('text_hex',b'wrong'.hex()),('generated',31),('prompt',8205),('eos',True)]:
            cells=fixture();cells[2]['audit']['runs'][0]['observations'][1][key]=value
            with self.assertRaises(ValueError):aggregate(cells)
        cells=fixture();cells[0]['audit']['runs'][0]['observations'].pop()
        with self.assertRaises(ValueError):aggregate(cells)

    def test_reject_nonfinite_duration(self):
        cells=fixture();cells[0]['audit']['runs'][0]['observations'][5]['interval']['milliseconds']=float('nan')
        with self.assertRaises(ValueError):aggregate(cells)

if __name__=='__main__':unittest.main()
