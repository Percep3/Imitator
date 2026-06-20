import unittest

from build_kb import mark_kb_built


class TestMarkKbBuilt(unittest.TestCase):
    def test_sets_timestamp(self):
        status = {"papers": {}, "kb_built_at": None}
        result = mark_kb_built(status, "2026-06-20T18:30:00+00:00")
        self.assertEqual(result["kb_built_at"], "2026-06-20T18:30:00+00:00")

    def test_preserves_other_keys(self):
        status = {"papers": {"foo": {"extracted": True}}, "kb_built_at": None}
        result = mark_kb_built(status, "2026-06-20T18:30:00+00:00")
        self.assertEqual(result["papers"], {"foo": {"extracted": True}})


if __name__ == "__main__":
    unittest.main()
