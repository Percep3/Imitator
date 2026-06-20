import unittest

from convert_papers import needs_convert


class TestNeedsConvert(unittest.TestCase):
    def test_force_always_true(self):
        entry = {"source_hash": "sha256:abc"}
        self.assertTrue(needs_convert(entry, "sha256:abc", force=True))

    def test_new_paper_no_entry(self):
        self.assertTrue(needs_convert(None, "sha256:abc", force=False))

    def test_unchanged_hash_skips(self):
        entry = {"source_hash": "sha256:abc"}
        self.assertFalse(needs_convert(entry, "sha256:abc", force=False))

    def test_changed_hash_reconverts(self):
        entry = {"source_hash": "sha256:old"}
        self.assertTrue(needs_convert(entry, "sha256:new", force=False))


if __name__ == "__main__":
    unittest.main()
