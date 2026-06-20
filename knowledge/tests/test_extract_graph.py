import unittest

from extract_graph import merge_graph, needs_extract


class TestNeedsExtract(unittest.TestCase):
    def test_force_always_true(self):
        self.assertTrue(needs_extract({"extracted": True}, force=True))

    def test_new_paper_no_entry(self):
        self.assertTrue(needs_extract(None, force=False))

    def test_already_extracted_skips(self):
        self.assertFalse(needs_extract({"extracted": True}, force=False))

    def test_not_yet_extracted_runs(self):
        self.assertTrue(needs_extract({"extracted": False}, force=False))


class TestMergeGraph(unittest.TestCase):
    def test_merge_into_empty_graph(self):
        existing = {"nodes": [], "edges": []}
        new_nodes = [{"id": "paper_a", "label": "A"}]
        new_edges = [{"source": "paper_a", "target": "dataset_x", "relation": "evaluated_on"}]

        result = merge_graph(existing, new_nodes, new_edges)

        self.assertEqual(result["nodes"], new_nodes)
        self.assertEqual(result["edges"], new_edges)

    def test_merge_appends_edges_and_overwrites_nodes_by_id(self):
        existing = {
            "nodes": [{"id": "dataset_x", "label": "old label"}],
            "edges": [{"source": "paper_a", "target": "dataset_x", "relation": "evaluated_on"}],
        }
        new_nodes = [{"id": "dataset_x", "label": "new label"}, {"id": "paper_b", "label": "B"}]
        new_edges = [{"source": "paper_b", "target": "dataset_x", "relation": "evaluated_on"}]

        result = merge_graph(existing, new_nodes, new_edges)

        ids = {n["id"]: n for n in result["nodes"]}
        self.assertEqual(ids["dataset_x"]["label"], "new label")
        self.assertIn("paper_b", ids)
        self.assertEqual(len(result["edges"]), 2)


if __name__ == "__main__":
    unittest.main()
