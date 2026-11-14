"""Test the exact four-row marker admission block without launching a GPU job."""
from pathlib import Path
import re,textwrap,unittest
source=Path(__file__).with_name('run_generation.py').read_text()
a=source.index(' quad_lines=');b=source.index(' for p in inputs:',a)
code=compile(textwrap.dedent(source[a:b]),'generation-marker-block','exec')
class Markers(unittest.TestCase):
    def test_inventory(self):
        valid='CUDA Q4 four rows capture/direct calls: 96\n'*3
        cases=[(valid,True),('',False),(valid.splitlines()[0],False),(valid+valid,False),
               (valid.replace('96','0'),False),(valid.replace('96','bad'),False),
               (valid.replace('96','-1'),False),(valid.replace('96','96 extra'),False)]
        for trace,expected in cases:
            with self.subTest(trace=trace):
                if expected:exec(code,{'re':re,'trace':trace})
                else:
                    with self.assertRaises(AssertionError):exec(code,{'re':re,'trace':trace})
if __name__=='__main__':unittest.main()
