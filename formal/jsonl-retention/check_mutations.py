"""Run the real randomized oracle against five in-memory production-function mutations."""

import inspect
import subprocess
import sys

from brainlayer import jsonl_backup

if len(sys.argv) > 1:
    source = inspect.getsource(jsonl_backup.prune_local_archives)
    replacements = {
        "verified": [('receipt.get("verified") is not True', "False")],
        "age": [("age < days * 86400", "False"), ('now - receipt["verified_at"] < days * 86400', "False")],
        "live": [('surviving_archives.get(receipt.get("archive_id")) != receipt["archive_md5"]', "False")],
        "md5": [('_md5_file(path) != receipt["archive_md5"]', "False")],
        "newest": [("path == newest", "False")],
    }
    for old, new in replacements[sys.argv[1]]:
        assert old in source, old
        source = source.replace(old, new)
    exec(compile(source, "<mutated-retention>", "exec"), jsonl_backup.__dict__)
    import pytest

    sys.exit(pytest.main(["-q", "tests/test_jsonl_local_retention.py::test_random_retention_and_crashes"]))

for mutation in ("verified", "age", "live", "md5", "newest"):
    result = subprocess.run([sys.executable, __file__, mutation], capture_output=True, text=True)
    expected = {"age": "deleted younger than 14 days", "newest": "deleted newest"}.get(
        mutation, "deleted without verified exact copy"
    )
    if result.returncode != 1 or expected not in result.stdout:
        print(result.stdout, result.stderr)
        raise SystemExit(f"{mutation} mutation did not fail the invariant oracle")
    print(f"PASS: {mutation} mutation rejected by real retention randomized oracle")
