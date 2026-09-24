from scripts.check_tracked_privacy_exports import matching_tracked_exports


def test_matching_tracked_exports_finds_only_private_export_patterns():
    paths = [
        "eval_results/kg-export.json",
        "eval_results/result.json",
        "eval_results/kg-export.JSON",
        "artifacts/run.jsonl",
        "artifacts/2026/run/raw.jsonl",
        "artifacts/2026/run/summary.json",
        "src/artifacts/raw.jsonl",
    ]

    assert matching_tracked_exports(paths) == [
        "artifacts/2026/run/raw.jsonl",
        "artifacts/run.jsonl",
        "eval_results/kg-export.json",
    ]
