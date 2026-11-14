import unittest
from validate_sanitizers import summary_valid

class SummaryTests(unittest.TestCase):
    def test_tool_specific_summaries(self):
        normal = '========= ERROR SUMMARY: 0 errors'
        race = '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
        for tool in ('memcheck', 'racecheck', 'synccheck'):
            good = race if tool == 'racecheck' else normal
            bad = normal if tool == 'racecheck' else race
            for text, expected in ((good, True), (good+'\n', True), ('', False),
                                   (good+'\n'+good, False), (bad, False),
                                   (good.replace('0 errors','1 errors'), False),
                                   (good+'\n'+bad, False)):
                with self.subTest(tool=tool, text=text):
                    self.assertEqual(summary_valid(tool,text),expected)
        self.assertFalse(summary_valid('racecheck',race.replace('0 warnings','1 warnings')))
        self.assertFalse(summary_valid('racecheck',race.replace('0 hazards','1 hazards')))

if __name__ == '__main__':
    unittest.main(verbosity=2)
