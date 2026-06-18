import json
from pathlib import Path

import pytest

from scripts import braintrust_ingest


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    labeled_path = tmp_path / "labeled_eval_candidates.json"
    labeled_path.write_text(
        json.dumps(
            [
                {
                    "query": "personal query text",
                    "returned_chunks": [
                        {
                            "id": "chunk-relevant",
                            "snippet": "personal chunk text",
                            "machine_label": "relevant",
                            "why": "personal rationale",
                        },
                        {
                            "id": "chunk-not",
                            "snippet": "other personal chunk text",
                            "machine_label": "not",
                            "why": "personal rationale",
                        },
                    ],
                    "suspected_miss": None,
                    "suspected_misorder": None,
                    "notes": "personal label notes",
                    "label_source": "machine",
                }
            ]
        )
    )
    retrieval_path = tmp_path / "retrieval_benchmark_results.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "meta": {"n_chunks": 893, "method": "fixture", "source": "local", "generated": "2026-06-02"},
                "per_variant": {
                    "A": {
                        "ok_chunks": 893,
                        "scored_queries": 51,
                        "indexed_coverage": 0.9,
                        "recall@1": 0.1,
                        "recall@5": 0.5,
                        "recall@10": 0.8,
                        "mrr": 0.4,
                        "ndcg@10": 0.7,
                    }
                },
            }
        )
    )
    raw_path = tmp_path / "abcde_enrich_nebius.jsonl"
    raw_path.write_text('{"personal":"row"}\n{"personal":"row2"}\n')
    return labeled_path, retrieval_path, raw_path


def test_payload_redacts_personal_content_and_keeps_ids(tmp_path: Path):
    labeled_path, retrieval_path, raw_path = _write_fixture_files(tmp_path)

    payload = braintrust_ingest.build_ingestion_payload(labeled_path, retrieval_path, raw_path)

    assert payload.counts == {
        "labeled_queries": 1,
        "dataset_records": 1,
        "raw_enriched_rows_local_only": 2,
        "retrieval_variants": 1,
    }
    record = payload.dataset_records[0]
    assert record["id"] == "query-0001"
    relevant_hash = braintrust_ingest.stable_hash("chunk-relevant")
    not_hash = braintrust_ingest.stable_hash("chunk-not")
    assert record["input"] == {
        "query_id": "query-0001",
        "query_sha256": braintrust_ingest.stable_hash("personal query text"),
        "query_text_redacted": True,
        "returned_chunk_id_hashes": [relevant_hash, not_hash],
    }
    assert record["expected"] == {"relevant_chunk_id_hashes": [relevant_hash]}
    assert record["tags"] == ["brainlayer", "retrieval-grading", "machine"]
    assert record["metadata"]["label_counts"] == {"not": 1, "relevant": 1}
    assert record["metadata"]["notes_redacted"] is True

    serialized = json.dumps(payload.as_dry_run_dict())
    assert "personal query text" not in serialized
    assert "personal chunk text" not in serialized
    assert "personal label notes" not in serialized


def test_recall_scorer_uses_expected_relevant_ids():
    scorer = braintrust_ingest.make_recall_at_k_scorer(2)

    score = scorer(
        input={"returned_chunk_id_hashes": ["chunk-a", "chunk-b", "chunk-c"]},
        output={"retrieved_chunk_id_hashes": ["chunk-a", "chunk-b", "chunk-c"]},
        expected={"relevant_chunk_id_hashes": ["chunk-b", "chunk-z"]},
    )

    assert score == pytest.approx(0.5)
    assert scorer.__name__ == "recall_at_2"


def test_boolean_false_suspected_flags_are_not_marked_present():
    record = braintrust_ingest._build_dataset_record(
        {
            "query": "search query",
            "returned_chunks": [],
            "suspected_miss": False,
            "suspected_misorder": False,
            "label_source": "machine",
        },
        index=1,
    )

    assert record["metadata"]["suspected_miss_present"] is False
    assert record["metadata"]["suspected_misorder_present"] is False


def test_default_paths_do_not_point_at_user_checkout():
    assert "/Users/" not in str(braintrust_ingest.DEFAULT_RAW_ENRICHED_PATH)
    assert str(braintrust_ingest.DEFAULT_RAW_ENRICHED_PATH).startswith("/tmp/")


def test_send_payload_uses_braintrust_sdk_without_leaking_raw_text(tmp_path: Path):
    labeled_path, retrieval_path, raw_path = _write_fixture_files(tmp_path)
    payload = braintrust_ingest.build_ingestion_payload(labeled_path, retrieval_path, raw_path)

    class FakeDataset:
        def __init__(self):
            self.inserted = []
            self.flushed = False

        def insert(self, **kwargs):
            self.inserted.append(kwargs)
            return kwargs["id"]

        def flush(self):
            self.flushed = True

    class FakeBraintrust:
        def __init__(self):
            self.dataset = FakeDataset()
            self.init_dataset_calls = []
            self.eval_calls = []

        def init_dataset(self, **kwargs):
            self.init_dataset_calls.append(kwargs)
            return self.dataset

        def Eval(self, *args, **kwargs):
            self.eval_calls.append((args, kwargs))
            return {"ok": True}

    fake = FakeBraintrust()
    settings = braintrust_ingest.BraintrustSettings(
        project="BrainLayer Retrieval",
        dataset="Candidates",
        experiment="ABCD-E",
        api_key="env-key",
    )

    result = braintrust_ingest.send_payload(payload, settings, braintrust_module=fake)

    assert result == {"dataset_records_sent": 1, "eval_invoked": True}
    assert fake.init_dataset_calls == [
        {
            "project": "BrainLayer Retrieval",
            "name": "Candidates",
            "api_key": "env-key",
            "use_output": False,
            "metadata": payload.dataset_metadata,
        }
    ]
    assert fake.dataset.inserted == [
        {
            "id": "query-0001",
            "input": payload.dataset_records[0]["input"],
            "expected": payload.dataset_records[0]["expected"],
            "metadata": payload.dataset_records[0]["metadata"],
            "tags": payload.dataset_records[0]["tags"],
        }
    ]
    assert fake.dataset.flushed is True
    assert fake.eval_calls[0][0] == ("BrainLayer Retrieval",)
    assert fake.eval_calls[0][1]["experiment_name"] == "ABCD-E"
    assert fake.eval_calls[0][1]["metadata"]["privacy"] == "raw_query_and_chunk_text_redacted"
    assert fake.eval_calls[0][1]["no_send_logs"] is False

    serialized = json.dumps(fake.eval_calls, default=str)
    assert "personal query text" not in serialized
    assert "personal chunk text" not in serialized
