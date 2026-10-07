"""Historical pilot artifacts remain gradeable without a model client."""


def test_pilot_artifact_analysis_and_report_are_offline(tmp_path):
    from brainlayer.eval.pilot_artifact_report import analyze_results, write_report

    rows = [
        {
            "chunk_id": "synthetic-1",
            "source": "fixture",
            "old_tags": '["historical"]',
            "parsed": {
                "b_topics": ["synthetic-topic"],
                "c_activity": "act:testing",
                "d_domain": ["dom:python"],
                "e_confidence": 0.9,
            },
        },
        {"chunk_id": "synthetic-2", "source": "fixture", "old_tags": "", "parsed": None, "error": "synthetic failure"},
    ]
    stats = analyze_results(rows, ())
    assert stats["total"] == 2
    assert stats["valid_json"] == 1
    assert stats["invalid_json"] == 1
    assert stats["topic_freq"] == {"synthetic-topic": 1}
    assert stats["avg_confidence"] == 0.9
    report = tmp_path / "pilot-report.md"
    write_report(stats, report)
    text = report.read_text()
    assert "1/2 (50.0%)" in text
    assert "synthetic-topic" in text and "historical" in text
    assert "synthetic failure" in text


def test_pilot_artifact_malformed_tag_lists_do_not_invent_grades():
    from brainlayer.eval.pilot_artifact_report import analyze_results

    stats = analyze_results(
        [
            {
                "chunk_id": "synthetic",
                "source": "fixture",
                "old_tags": "",
                "parsed": {"b_topics": "invalid", "d_domain": "invalid"},
            }
        ],
        (),
    )
    assert stats["unique_topics"] == 0
    assert stats["topic_freq"] == {}
    assert stats["domain_freq"] == {}
    assert stats["avg_confidence"] == 0
