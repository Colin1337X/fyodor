import hashlib
from pathlib import Path
import tempfile
import unittest
from model_identity import verify


class Identity(unittest.TestCase):
    def test_replaced_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            result = verify({path: expected})[str(path)]
            self.assertEqual(result['sha256'], expected)
            self.assertEqual(result['size'], 8)
            path.write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                verify({path: expected})

    def test_missing_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                verify({Path(directory) / 'absent.gguf': '0' * 64})


if __name__ == '__main__':
    unittest.main()
