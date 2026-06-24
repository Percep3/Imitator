# Pipeline Status Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `knowledge/pipeline_status.json` that tracks per-paper conversion/extraction state so `convert_papers.py` and `extract_graph.py` only do work (especially LLM calls) on papers that are new or changed, instead of all-or-nothing reprocessing.

**Architecture:** A new module `knowledge/pipeline_status.py` provides `sha256_file`, `load_status`, `save_status`, and `bootstrap_if_missing` (one-time migration that reconstructs status from existing `markdown/` and `graph.json` so already-done papers aren't redone). `convert_papers.py` and `extract_graph.py` each gain a small pure decision function (`needs_convert`, `needs_extract`) plus a merge function (`merge_graph`) and call into the shared module to read/update status as they iterate papers. `build_kb.py` gets a one-line status update (`kb_built_at`) since it has no LLM cost and doesn't need skip logic.

**Tech Stack:** Python 3, stdlib only for the new code (`hashlib`, `json`, `pathlib`, `datetime`). Tests use stdlib `unittest` (no pytest in `knowledge/.venv`) via `knowledge/.venv/bin/python -m unittest discover -s tests -v`, run with `knowledge/` as cwd.

## Global Constraints

- Full design spec: `docs/specs/2026-06-20-pipeline-status-tracking-design.md` — read it before starting if anything here is ambiguous.
- `knowledge/pipeline_status.json` is versioned in git (NOT added to `.gitignore`) — confirmed decision, unlike `papers/`, `markdown/`, `kb/` which are ignored.
- **Never call the real DeepSeek API during this plan's execution.** `extract_graph.py`'s real run (processing the papers that don't have `.md`→graph entries yet) costs real tokens/money and is left for the user to run manually after this plan is merged. Only run the test suite (which doesn't touch the network) and, where explicitly stated in Task 6, the real (free, local) `convert_papers.py`.
- No new third-party dependencies. Use only what's already installed in `knowledge/.venv` (verified: `openai`, `python-dotenv`, `markitdown` present; `pytest` is NOT present, hence stdlib `unittest`).
- Modified files keep their existing Spanish-language docstrings/comments style and CLI usage docstring at the top of each script.

---

### Task 1: `pipeline_status.py` core helpers (hash, load, save)

**Files:**
- Create: `knowledge/pipeline_status.py`
- Test: `knowledge/tests/test_pipeline_status.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str` (returns `"sha256:<hex>"`), `load_status(path: Path = STATUS_PATH) -> dict` (returns `{"papers": {}, "kb_built_at": None}` if file missing), `save_status(status: dict, path: Path = STATUS_PATH) -> None`, module constant `STATUS_PATH`.

- [ ] **Step 1: Create the tests directory and write the failing tests**

Create `knowledge/tests/test_pipeline_status.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_status import load_status, save_status, sha256_file


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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'pipeline_status'`

- [ ] **Step 3: Write the implementation**

Create `knowledge/pipeline_status.py`:

```python
"""Tracking de estado del pipeline de papers (papers/ -> markdown/ -> graph.json -> kb/).

Permite que convert_papers.py y extract_graph.py sepan qué papers ya fueron
procesados, para no reconvertir PDFs sin cambios ni volver a llamar al LLM
sobre papers ya extraídos.
"""
import hashlib
import json
from pathlib import Path

STATUS_PATH = Path(__file__).parent / "pipeline_status.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def load_status(path: Path = STATUS_PATH) -> dict:
    if not path.exists():
        return {"papers": {}, "kb_built_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(status: dict, path: Path = STATUS_PATH) -> None:
    path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: `OK` (5 tests: 3 in `TestSha256File`, 2 in `TestLoadSaveStatus`)

- [ ] **Step 5: Commit**

```bash
git add knowledge/pipeline_status.py knowledge/tests/test_pipeline_status.py
git commit -m "feat: add pipeline_status.py core helpers for sha256/load/save"
```

---

### Task 2: `bootstrap_if_missing` (migration from existing markdown/graph.json)

**Files:**
- Modify: `knowledge/pipeline_status.py`
- Test: `knowledge/tests/test_pipeline_status.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `bootstrap_if_missing(status: dict, papers_dir: Path, markdown_dir: Path, graph_path: Path) -> dict`. For every PDF in `papers_dir` whose stem has **no** entry yet in `status["papers"]`: if a matching `.md` exists in `markdown_dir`, add an entry `{"source_hash": ..., "converted": True, "extracted": <True if a node in graph_path has source_file == "<stem>.md", else False>}`. If no matching `.md` exists, add nothing (never converted, nothing to migrate). Never overwrites an existing entry. This is what makes already-converted/already-extracted papers (from before this feature existed) stay skipped instead of being reprocessed on the first run.

- [ ] **Step 1: Write the failing tests**

Append to `knowledge/tests/test_pipeline_status.py` (add import and new test class):

```python
from pipeline_status import bootstrap_if_missing, load_status, save_status, sha256_file
```

(replace the existing `from pipeline_status import load_status, save_status, sha256_file` import line with the line above)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'bootstrap_if_missing'`

- [ ] **Step 3: Write the implementation**

Append to `knowledge/pipeline_status.py`:

```python
def bootstrap_if_missing(
    status: dict, papers_dir: Path, markdown_dir: Path, graph_path: Path
) -> dict:
    """Completa entradas faltantes en status['papers'] a partir del estado en
    disco (markdown/ y graph.json), para no reprocesar papers que ya estaban
    convertidos/extraídos antes de que este tracking existiera. No toca
    entradas que ya estén registradas.
    """
    papers_status = status.setdefault("papers", {})

    graph_nodes = []
    if graph_path.exists():
        graph_nodes = json.loads(graph_path.read_text(encoding="utf-8")).get("nodes", [])
    extracted_source_files = {n.get("source_file") for n in graph_nodes}

    if not papers_dir.exists():
        return status

    for pdf in papers_dir.iterdir():
        if not pdf.is_file():
            continue
        stem = pdf.stem
        if stem in papers_status:
            continue
        md_path = markdown_dir / f"{stem}.md"
        if not md_path.exists():
            continue
        papers_status[stem] = {
            "source_hash": sha256_file(pdf),
            "converted": True,
            "extracted": md_path.name in extracted_source_files,
        }
    return status
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: `OK` (10 tests total)

- [ ] **Step 5: Commit**

```bash
git add knowledge/pipeline_status.py knowledge/tests/test_pipeline_status.py
git commit -m "feat: add bootstrap_if_missing to migrate pre-existing markdown/graph state"
```

---

### Task 3: Wire status tracking into `convert_papers.py`

**Files:**
- Modify: `knowledge/convert_papers.py`
- Test: `knowledge/tests/test_convert_papers.py`

**Interfaces:**
- Consumes: `pipeline_status.{load_status, save_status, sha256_file, bootstrap_if_missing}` (Tasks 1-2).
- Produces: `needs_convert(entry: dict | None, current_hash: str, force: bool) -> bool` (pure function, importable for Task 4/5 reference if needed, though they don't use it directly).

- [ ] **Step 1: Write the failing test**

Create `knowledge/tests/test_convert_papers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd knowledge && .venv/bin/python -m unittest tests.test_convert_papers -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'needs_convert'`

- [ ] **Step 3: Modify `knowledge/convert_papers.py`**

Replace the top imports and constants (currently lines 10-15):

```python
import argparse
import os
from pathlib import Path

PAPERS_DIR = Path(__file__).parent / "papers"
MARKDOWN_DIR = Path(__file__).parent / "markdown"
```

with:

```python
import argparse
import os
from pathlib import Path

from pipeline_status import bootstrap_if_missing, load_status, save_status, sha256_file

PAPERS_DIR = Path(__file__).parent / "papers"
MARKDOWN_DIR = Path(__file__).parent / "markdown"
GRAPH_PATH = Path(__file__).parent / "graph.json"


def needs_convert(entry: dict | None, current_hash: str, force: bool) -> bool:
    if force:
        return True
    if entry is None:
        return True
    return entry.get("source_hash") != current_hash
```

Then replace the whole `convert_all` function body:

```python
def convert_all(force: bool, parser_name: str, tier: str) -> None:
    MARKDOWN_DIR.mkdir(exist_ok=True)

    sources = sorted(p for p in PAPERS_DIR.iterdir() if p.is_file())
    if not sources:
        print(f"No se encontraron archivos en {PAPERS_DIR}")
        return

    status = load_status()
    bootstrap_if_missing(status, PAPERS_DIR, MARKDOWN_DIR, GRAPH_PATH)
    papers_status = status["papers"]

    for src in sources:
        stem = src.stem
        dst = MARKDOWN_DIR / f"{stem}.md"
        current_hash = sha256_file(src)
        entry = papers_status.get(stem)

        if not needs_convert(entry, current_hash, force):
            print(f"skip  {dst.name} (hash sin cambios)")
            continue
        try:
            if parser_name == "llamaparse":
                text = convert_with_llamaparse(src, tier)
            else:
                text = convert_with_markitdown(src)
        except Exception as e:
            print(f"error {src.name}: {e}")
            continue
        dst.write_text(text, encoding="utf-8")

        hash_changed = entry is None or entry.get("source_hash") != current_hash
        papers_status[stem] = {
            "source_hash": current_hash,
            "converted": True,
            "extracted": False if hash_changed else (entry or {}).get("extracted", False),
        }
        save_status(status)
        print(f"ok    {src.name} -> {dst.name} ({parser_name})")
```

(The `convert_with_markitdown` and `convert_with_llamaparse` functions, and the `if __name__ == "__main__":` block, are unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: `OK` (14 tests total)

- [ ] **Step 5: Commit**

```bash
git add knowledge/convert_papers.py knowledge/tests/test_convert_papers.py
git commit -m "feat: skip unchanged papers in convert_papers.py via pipeline_status"
```

---

### Task 4: Wire status tracking into `extract_graph.py`

**Files:**
- Modify: `knowledge/extract_graph.py`
- Test: `knowledge/tests/test_extract_graph.py`

**Interfaces:**
- Consumes: `pipeline_status.{load_status, save_status, bootstrap_if_missing}` (Tasks 1-2).
- Produces: `needs_extract(entry: dict | None, force: bool) -> bool`, `merge_graph(existing: dict, new_nodes: list[dict], new_edges: list[dict]) -> dict` (both pure functions).

- [ ] **Step 1: Write the failing tests**

Create `knowledge/tests/test_extract_graph.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd knowledge && .venv/bin/python -m unittest tests.test_extract_graph -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'merge_graph'`

- [ ] **Step 3: Modify `knowledge/extract_graph.py`**

Add the import and a `PAPERS_DIR` constant near the top (after the existing `load_dotenv(...)` line and before `MARKDOWN_DIR = ...`):

```python
from pipeline_status import bootstrap_if_missing, load_status, save_status

MARKDOWN_DIR = Path(__file__).parent / "markdown"
PAPERS_DIR = Path(__file__).parent / "papers"
OUTPUT_PATH = Path(__file__).parent / "graph.json"
MODEL = "deepseek-chat"
```

Add the two pure functions right after `extract_paper(...)` and before `def main(...)`:

```python
def needs_extract(entry: dict | None, force: bool) -> bool:
    if force:
        return True
    if entry is None:
        return True
    return not entry.get("extracted", False)


def merge_graph(existing: dict, new_nodes: list[dict], new_edges: list[dict]) -> dict:
    nodes_by_id = {n["id"]: n for n in existing.get("nodes", [])}
    for n in new_nodes:
        nodes_by_id[n["id"]] = n
    edges = list(existing.get("edges", [])) + list(new_edges)
    return {"nodes": list(nodes_by_id.values()), "edges": edges}
```

Replace the entire `main` function:

```python
def main(force: bool) -> None:
    status = load_status()
    bootstrap_if_missing(status, PAPERS_DIR, MARKDOWN_DIR, OUTPUT_PATH)
    papers_status = status["papers"]

    if force or not OUTPUT_PATH.exists():
        graph = {"nodes": [], "edges": []}
    else:
        graph = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url="https://api.deepseek.com")

    processed = 0
    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        stem = md_path.stem
        entry = papers_status.get(stem)
        if not needs_extract(entry, force):
            print(f"skip  {md_path.name} (ya extraido)")
            continue

        print(f"extrayendo {md_path.name}...")
        result = extract_paper(client, md_path)
        graph = merge_graph(graph, result["nodes"], result["edges"])
        papers_status[stem] = {**(entry or {}), "extracted": True}
        processed += 1

        OUTPUT_PATH.write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        save_status(status)
        print(f"  -> {len(result['nodes'])} nodes, {len(result['edges'])} edges")

    if processed == 0:
        print("Nada para extraer, todos los papers ya estaban procesados (usa --force para regenerar todo)")
    else:
        print(f"Grafo guardado en {OUTPUT_PATH} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")
```

(The `extract_paper` function and the `if __name__ == "__main__":` argparse block are unchanged. Remove the old early-return gate `if OUTPUT_PATH.exists() and not force: ...` — it's fully replaced by the per-paper logic above.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: `OK` (20 tests total)

- [ ] **Step 5: Commit**

```bash
git add knowledge/extract_graph.py knowledge/tests/test_extract_graph.py
git commit -m "feat: make extract_graph.py incremental via pipeline_status"
```

---

### Task 5: `kb_built_at` timestamp in `build_kb.py`

**Files:**
- Modify: `knowledge/build_kb.py`
- Test: `knowledge/tests/test_build_kb.py`

**Interfaces:**
- Consumes: `pipeline_status.{load_status, save_status}` (Task 1).
- Produces: `mark_kb_built(status: dict, built_at: str) -> dict` (pure function).

- [ ] **Step 1: Write the failing test**

Create `knowledge/tests/test_build_kb.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd knowledge && .venv/bin/python -m unittest tests.test_build_kb -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'mark_kb_built'`

- [ ] **Step 3: Modify `knowledge/build_kb.py`**

Replace the top imports (currently lines 13-17):

```python
import json
from pathlib import Path

GRAPH_PATH = Path(__file__).parent / "graph.json"
KB_DIR = Path(__file__).parent / "kb"
```

with:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline_status import load_status, save_status

GRAPH_PATH = Path(__file__).parent / "graph.json"
KB_DIR = Path(__file__).parent / "kb"


def mark_kb_built(status: dict, built_at: str) -> dict:
    status["kb_built_at"] = built_at
    return status
```

Replace the end of `main()` (currently the last two statements before `if __name__ == "__main__":`):

```python
    (KB_DIR / "research_gaps.md").write_text(
        build_research_gaps_md(nodes_by_id, edges), encoding="utf-8"
    )
    print(f"KB escrito en {KB_DIR}/: papers_metadata.json, extraction_table.json, research_gaps.md")
```

with:

```python
    (KB_DIR / "research_gaps.md").write_text(
        build_research_gaps_md(nodes_by_id, edges), encoding="utf-8"
    )

    status = load_status()
    save_status(mark_kb_built(status, datetime.now(timezone.utc).isoformat()))

    print(f"KB escrito en {KB_DIR}/: papers_metadata.json, extraction_table.json, research_gaps.md")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: `OK` (22 tests total)

- [ ] **Step 5: Commit**

```bash
git add knowledge/build_kb.py knowledge/tests/test_build_kb.py
git commit -m "feat: record kb_built_at timestamp in pipeline_status.json"
```

---

### Task 6: Real bootstrap run and verification (no API calls)

**Files:**
- Modify: none (verification only)
- Create (generated by running the real script, then committed): `knowledge/pipeline_status.json`

**Interfaces:** None — this task runs the already-implemented code against the real `knowledge/papers/`, `knowledge/markdown/`, `knowledge/graph.json` to produce the real status file.

- [ ] **Step 1: Run the full test suite one more time as a final regression check**

Run: `cd knowledge && .venv/bin/python -m unittest discover -s tests -v`
Expected: `OK` (22 tests total)

- [ ] **Step 2: Run convert_papers.py for real (free, local, no API calls)**

Run: `cd knowledge && .venv/bin/python convert_papers.py`

Expected output: for the 2 papers that already have a `.md` (`Continuous_Sign_Language_Recognition_with_Multi-Sc`, `Nam_LiftSign_at_SignEval_2026_...`), a `skip ... (hash sin cambios)` line each (bootstrapped on this first run, then skipped within the same run since `convert_all` reads `papers_status` once per run — note: on this very first run these two will show as bootstrapped silently with no print, since `bootstrap_if_missing` only fills the dict, it doesn't print; the *first* run will actually print `skip` for them because `needs_convert` sees the freshly-bootstrapped entry with matching hash). For the other 5 PDFs without a `.md` yet, expect `ok    <name>.pdf -> <name>.md (markitdown)` lines, since `markitdown` is the default parser and is local/free.

- [ ] **Step 3: Inspect the generated status file**

Run: `cat knowledge/pipeline_status.json`
Expected: a `"papers"` object with 7 entries (one per PDF in `knowledge/papers/`). The 2 originally-already-converted papers should show `"extracted": true` (because their nodes already exist in `graph.json` with matching `source_file`). The 5 newly-converted papers should show `"converted": true, "extracted": false` (correctly queued for the user to run `extract_graph.py` for real, later, on their own).

- [ ] **Step 4: Confirm `pipeline_status.json` is not gitignored**

Run: `git check-ignore -v knowledge/pipeline_status.json`
Expected: no output (exit code 1) — confirms the file is NOT ignored and can be committed.

- [ ] **Step 5: Commit the generated status file**

```bash
git add knowledge/pipeline_status.json
git commit -m "chore: bootstrap pipeline_status.json from existing markdown/graph state"
```

- [ ] **Step 6: Tell the user what's left**

Report to the user (do not run it yourself): `knowledge/.venv/bin/python knowledge/extract_graph.py` will now only call the DeepSeek API for the 5 papers that show `"extracted": false` in `pipeline_status.json` — running it again after that will skip all 7 and cost nothing, and adding an 8th paper later will only extract that one.
