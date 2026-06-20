import json
import tempfile
import unittest
from pathlib import Path

from pipeline_status import bootstrap_if_missing, load_status, save_status, sha256_file


class TestSha256File(unittest.TestCase):
    def test_returns_prefixed_hex_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.pdf"
            p.write_bytes(b"hello")
            result = sha256_file(p)
            self.assertTrue(result.startswith("sha256:"))
            self.assertEqual(len(result), len("sha256:") + 64)

    def test_same_content_same_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.pdf"
            p2 = Path(tmp) / "b.pdf"
            p1.write_bytes(b"same content")
            p2.write_bytes(b"same content")
            self.assertEqual(sha256_file(p1), sha256_file(p2))

    def test_different_content_different_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.pdf"
            p2 = Path(tmp) / "b.pdf"
            p1.write_bytes(b"content one")
            p2.write_bytes(b"content two")
            self.assertNotEqual(sha256_file(p1), sha256_file(p2))


class TestLoadSaveStatus(unittest.TestCase):
    def test_load_missing_file_returns_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline_status.json"
            status = load_status(path)
            self.assertEqual(status, {"papers": {}, "kb_built_at": None})

    def test_save_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline_status.json"
            status = {
                "papers": {
                    "foo": {"source_hash": "sha256:abc", "converted": True, "extracted": False}
                },
                "kb_built_at": None,
            }
            save_status(status, path)
            loaded = load_status(path)
            self.assertEqual(loaded, status)


class TestBootstrapIfMissing(unittest.TestCase):
    def _make_dirs(self, tmp):
        papers_dir = Path(tmp) / "papers"
        markdown_dir = Path(tmp) / "markdown"
        papers_dir.mkdir()
        markdown_dir.mkdir()
        return papers_dir, markdown_dir

    def test_adds_entry_for_converted_and_extracted_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_dir, markdown_dir = self._make_dirs(tmp)
            (papers_dir / "foo.pdf").write_bytes(b"pdf content")
            (markdown_dir / "foo.md").write_text("# foo")
            graph_path = Path(tmp) / "graph.json"
            graph_path.write_text(json.dumps({
                "nodes": [{"id": "paper_foo", "source_file": "foo.md"}],
                "edges": [],
            }))

            status = {"papers": {}}
            bootstrap_if_missing(status, papers_dir, markdown_dir, graph_path)

            self.assertTrue(status["papers"]["foo"]["converted"])
            self.assertTrue(status["papers"]["foo"]["extracted"])
            self.assertTrue(status["papers"]["foo"]["source_hash"].startswith("sha256:"))

    def test_adds_entry_for_converted_but_not_extracted_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_dir, markdown_dir = self._make_dirs(tmp)
            (papers_dir / "bar.pdf").write_bytes(b"pdf content")
            (markdown_dir / "bar.md").write_text("# bar")
            graph_path = Path(tmp) / "graph.json"
            graph_path.write_text(json.dumps({"nodes": [], "edges": []}))

            status = {"papers": {}}
            bootstrap_if_missing(status, papers_dir, markdown_dir, graph_path)

            self.assertTrue(status["papers"]["bar"]["converted"])
            self.assertFalse(status["papers"]["bar"]["extracted"])

    def test_skips_paper_without_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_dir, markdown_dir = self._make_dirs(tmp)
            (papers_dir / "baz.pdf").write_bytes(b"pdf content")
            graph_path = Path(tmp) / "graph.json"
            graph_path.write_text(json.dumps({"nodes": [], "edges": []}))

            status = {"papers": {}}
            bootstrap_if_missing(status, papers_dir, markdown_dir, graph_path)

            self.assertNotIn("baz", status["papers"])

    def test_does_not_overwrite_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_dir, markdown_dir = self._make_dirs(tmp)
            (papers_dir / "foo.pdf").write_bytes(b"pdf content")
            (markdown_dir / "foo.md").write_text("# foo")
            graph_path = Path(tmp) / "graph.json"
            graph_path.write_text(json.dumps({"nodes": [], "edges": []}))

            status = {
                "papers": {
                    "foo": {"source_hash": "sha256:keep", "converted": True, "extracted": False}
                }
            }
            bootstrap_if_missing(status, papers_dir, markdown_dir, graph_path)

            self.assertEqual(status["papers"]["foo"]["source_hash"], "sha256:keep")

    def test_missing_graph_file_treats_nothing_as_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_dir, markdown_dir = self._make_dirs(tmp)
            (papers_dir / "foo.pdf").write_bytes(b"pdf content")
            (markdown_dir / "foo.md").write_text("# foo")
            graph_path = Path(tmp) / "graph.json"  # no existe

            status = {"papers": {}}
            bootstrap_if_missing(status, papers_dir, markdown_dir, graph_path)

            self.assertFalse(status["papers"]["foo"]["extracted"])


if __name__ == "__main__":
    unittest.main()
