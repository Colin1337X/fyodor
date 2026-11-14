import unittest
from run_sanitizer import validate, validate_oracle


def fixture():
    shapes = [(256, r) for r in (1, 7, 9, 15, 16, 17, 63, 64, 65)]
    shapes += [(k, 17) for k in (512, 768, 1024, 1280, 1792)]
    shapes += [(3840, r) for r in (512, 2048, 4096, 8192, 15360)]
    shapes += [(4096, 3840), (8192, 3840), (15360, 3840)]
    cases = [dict(columns=k, rows=r, half_bits=None, pattern='random') for k, r in shapes]
    bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
            0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x27ae, 0xa75a]
    cases += [dict(columns=256, rows=17, half_bits=[d, m], pattern='random') for d in bits for m in bits]
    cases += [dict(columns=256, rows=17, half_bits=None, pattern=p)
              for p in ('zero', 'one', 'alternating', 'subnormal', 'large', 'impulse')]
    for case in cases:
        case.update(guards_clear=True, maximum_fraction_of_float64_bound={
            'nya_small_batch_12': .1, 'nya_q4_tf32_split': .2})
    return dict(passed=True, cases=cases, rejection_cases=[dict(columns=k, batch=b, output_unchanged=True)
        for k, b in ((0, 4), (1, 4), (255, 4), (257, 4), (256, 0), (256, 1), (256, 2), (256, 8))])


class Evidence(unittest.TestCase):
    def test_all_tools(self):
        for tool in ('memcheck', 'synccheck'):
            validate(0, '========= ERROR SUMMARY: 0 errors\n', '', fixture(), tool)
        validate(0, '', '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n', fixture(), 'racecheck')

    def test_inventory(self):
        for mode in ('missing', 'reordered', 'shape'):
            proof = fixture()
            if mode == 'missing': proof['cases'].pop()
            elif mode == 'reordered': proof['cases'].reverse()
            else: proof['cases'][0]['rows'] = 2
            with self.assertRaises(ValueError): validate_oracle(proof)

    def test_numerical_failures(self):
        for value in (1.01, -1, float('nan'), float('inf'), True):
            proof = fixture()
            proof['cases'][0]['maximum_fraction_of_float64_bound']['nya_q4_tf32_split'] = value
            with self.assertRaises(ValueError): validate_oracle(proof)

    def test_guards_and_comparisons(self):
        proof = fixture(); proof['cases'][0]['guards_clear'] = False
        with self.assertRaises(ValueError): validate_oracle(proof)
        proof = fixture(); del proof['cases'][0]['maximum_fraction_of_float64_bound']['nya_small_batch_12']
        with self.assertRaises(ValueError): validate_oracle(proof)

    def test_rejections(self):
        proof = fixture(); proof['rejection_cases'].pop()
        with self.assertRaises(ValueError): validate_oracle(proof)
        proof = fixture(); proof['rejection_cases'][0]['output_unchanged'] = False
        with self.assertRaises(ValueError): validate_oracle(proof)

    def test_failed_exit_or_summary(self):
        zero = '========= ERROR SUMMARY: 0 errors\n'
        for code in (False, None, 1, 99):
            with self.assertRaises(ValueError): validate(code, zero, '', fixture(), 'memcheck')
        for summary in ('', zero * 2, zero.replace('0 errors', '1 errors')):
            with self.assertRaises(ValueError): validate(0, summary, '', fixture(), 'memcheck')
        with self.assertRaises(ValueError):
            validate(0, '', '========= RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)\n', fixture(), 'racecheck')


if __name__ == '__main__':
    unittest.main()
