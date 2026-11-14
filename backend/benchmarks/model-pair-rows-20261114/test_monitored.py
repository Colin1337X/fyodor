import unittest
from run_monitored import reuse_inventory as inventory
def reuse_inventory(trace, enabled):
    return inventory(trace,enabled,'CUDA paired rows capture/direct calls:')

MARKER = 'CUDA paired rows capture/direct calls: 128\n'

class ReuseEvidence(unittest.TestCase):
    def test_enabled_matches_validated_inventory(self):
        self.assertEqual(reuse_inventory(MARKER * 3, True), ([128, 128, 128], True))

    def test_missing_or_extra_markers_fail(self):
        for count in (0, 1, 2, 4):
            self.assertFalse(reuse_inventory(MARKER * count, True)[1])

    def test_zero_hit_fails(self):
        self.assertFalse(reuse_inventory(MARKER * 2 + MARKER.replace('128', '0'), True)[1])

    def test_malformed_markers_fail(self):
        for trace in (MARKER.replace('128', '-1'), MARKER.replace('128', ''), MARKER.rstrip() + ' trailing'):
            for enabled in (False, True):
                self.assertFalse(reuse_inventory(trace, enabled)[1])

    def test_disabled_requires_no_reuse(self):
        self.assertEqual(reuse_inventory('unrelated trace\n', False), ([], True))
        for trace in (MARKER, MARKER * 3, MARKER.replace('128', '0')):
            self.assertFalse(reuse_inventory(trace, False)[1])

class BothPaths(unittest.TestCase):
    def test_qkv_stays_required(self):
        qkv='CUDA QKV input reuse capture/direct hits: 88\n'*3
        for paired in ('',MARKER*3):
            self.assertTrue(inventory(qkv+paired,True)[1])
            self.assertFalse(inventory(paired,True)[1])

if __name__ == '__main__':
    unittest.main()
