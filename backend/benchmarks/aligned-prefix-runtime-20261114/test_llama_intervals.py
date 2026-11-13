import copy
import json
import unittest
from llama_intervals import parse


class Intervals(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(round=i, prompt=2, generated=32, eos=False, start=2+i*2,
                          end=3+i*2, decode_ms=1000, ids=[42]*32, text_hex='41') for i in range(3)]

    def text(self, rows=None):
        return 'PROMPT_IDS 1 2\n' + '\n'.join(json.dumps(row) for row in (self.rows if rows is None else rows))

    def test_complete(self):
        self.assertEqual(parse(self.text(), [1, 2], (1, 8)), self.rows)

    def test_prompt_mismatch(self):
        with self.assertRaises(ValueError): parse(self.text(), [1, 3], (1, 8))
        with self.assertRaises(ValueError): parse('PROMPT_IDS 1 2\n'+self.text(), [1, 2], (1, 8))

    def test_incomplete(self):
        with self.assertRaises(ValueError): parse(self.text(self.rows[:2]), [1, 2], (1, 8))

    def test_contract(self):
        for field, value in [('round', True), ('prompt', 3), ('generated', 31), ('eos', True),
                             ('ids', [42]*31), ('ids', [True]*32), ('text_hex', 'ff')]:
            rows=copy.deepcopy(self.rows); rows[0][field]=value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                parse(self.text(rows), [1, 2], (1, 8))

    def test_interval(self):
        for field, value in [('start', 0), ('start', float('nan')), ('end', 9),
                             ('end', 2), ('decode_ms', 999), ('decode_ms', True)]:
            rows=copy.deepcopy(self.rows); rows[0][field]=value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                parse(self.text(rows), [1, 2], (1, 8))
        rows=copy.deepcopy(self.rows); rows[1].update(start=2,end=3)
        with self.assertRaises(ValueError): parse(self.text(rows), [1, 2], (1, 8))

    def test_nondeterminism(self):
        for field, value in [('text_hex', '42'), ('ids', [43]*32)]:
            rows=copy.deepcopy(self.rows); rows[2][field]=value
            with self.assertRaises(ValueError): parse(self.text(rows), [1, 2], (1, 8))


if __name__ == '__main__': unittest.main()
