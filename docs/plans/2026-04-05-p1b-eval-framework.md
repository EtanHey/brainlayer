# P1b Eval Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Ranx-based evaluation framework, checked-in 25-query qrels suite, and a baseline benchmark harness for BrainLayer search quality.

**Architecture:** Add a small `brainlayer.eval` package that loads qrels, runs arbitrary retrieval pipelines into Ranx `Run` objects, and evaluates or compares them against graded judgments. Keep benchmark pipelines and live-DB utilities in scripts so production search behavior stays unchanged while the eval harness can exercise the current FTS5 baseline and future hybrid variants.

**Tech Stack:** Python 3.11, Ranx, APSW/sqlite FTS5, pytest, Ruff

---

### Task 1: Add failing framework tests

**Files:**
- Create: `tests/test_eval_framework.py`
- Create: `tests/eval_qrels.json`
- Test: `tests/test_eval_framework.py`

**Step 1: Write the failing test**

Add tests for qrels loading, pipeline execution, metric evaluation, pipeline comparison, qrels file structure, and a temp-DB FTS pipeline.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_framework.py -q`
Expected: FAIL because `brainlayer.eval` and benchmark helpers do not exist yet, and/or `ranx` is missing.

**Step 3: Write minimal implementation**

Create the eval package and script-exposed FTS pipeline helper needed by the tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_framework.py -q`
Expected: PASS

### Task 2: Add Ranx benchmark module

**Files:**
- Create: `src/brainlayer/eval/__init__.py`
- Create: `src/brainlayer/eval/benchmark.py`
- Modify: `pyproject.toml`
- Test: `tests/test_eval_framework.py`

**Step 1: Write the failing test**

Cover JSON qrels loading, Ranx `Run` creation from pipeline output, metric evaluation, and multi-run comparison behavior.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_framework.py::test_benchmark_loads_qrels -q`
Expected: FAIL due to missing module or dependency.

**Step 3: Write minimal implementation**

Add `ranx` dependency and implement `SearchBenchmark` with JSON loading, metric defaults, and deterministic run building.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_framework.py -q`
Expected: PASS

### Task 3: Add qrels scaffold and build script

**Files:**
- Create: `scripts/build_qrels.py`
- Modify: `tests/eval_qrels.json`
- Test: `tests/test_eval_framework.py`

**Step 1: Write the failing test**

Validate that the checked-in qrels file has exactly 25 queries and every graded relevance value is in `0..3`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_framework.py::test_qrels_format_valid -q`
Expected: FAIL until the fixture is present and valid.

**Step 3: Write minimal implementation**

Create a live-DB helper script that fetches top FTS5 candidates, applies heuristic grades, and saves a Ranx-compatible JSON file. Check in an initial 25-query qrels fixture.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_framework.py::test_qrels_format_valid -q`
Expected: PASS

### Task 4: Add benchmark harness and baseline capture

**Files:**
- Create: `scripts/run_benchmark.py`
- Create: `tests/eval_results/`
- Test: `tests/test_eval_framework.py`

**Step 1: Write the failing test**

Cover the FTS5 pipeline against a temp DB and assert it returns ranked `(chunk_id, score)` results.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_framework.py::test_pipeline_fts5_returns_results -q`
Expected: FAIL until the pipeline helper exists.

**Step 3: Write minimal implementation**

Add FTS5-only, hybrid placeholder, and future entity placeholder pipelines plus a CLI benchmark runner that saves timestamped JSON metrics.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_framework.py -q`
Expected: PASS

### Task 5: Verify, format, and collect baseline

**Files:**
- Modify: new eval files only

**Step 1: Run targeted tests**

Run: `pytest tests/test_eval_framework.py -q`

**Step 2: Run repo-required quality checks**

Run: `ruff check src/`
Run: `ruff format src/ scripts/ tests/test_eval_framework.py`

**Step 3: Run live benchmark**

Run: `python scripts/run_benchmark.py`
Expected: baseline JSON written under `tests/eval_results/`

**Step 4: Review outputs**

Confirm the saved baseline includes at least `ndcg@10`, `recall@20`, `map@10`, and `mrr`.
