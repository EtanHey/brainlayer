"""Run the real randomized oracle against two in-memory production-function mutations."""

import inspect
import subprocess
import sys

from brainlayer import jsonl_backup

if len(sys.argv) > 1:
    source = inspect.getsource(jsonl_backup.prune_local_archives)
    if sys.argv[1] == "verified":
        source = source.replace('receipt.get("verified") is not True', "False")
    else:
        source = source.replace("age < days * 86400", "False")
    exec(compile(source, "<mutated-retention>", "exec"), jsonl_backup.__dict__)
    import pytest

    sys.exit(pytest.main(["-q", "tests/test_jsonl_local_retention.py::test_random_retention_and_crashes"]))

for mutation in ("verified", "age"):
    result = subprocess.run([sys.executable, __file__, mutation], capture_output=True, text=True)
    expected = "deleted without verified exact copy" if mutation == "verified" else "deleted younger than 14 days"
    if result.returncode != 1 or expected not in result.stdout:
        print(result.stdout, result.stderr)
        raise SystemExit(f"{mutation} mutation did not fail the invariant oracle")
    print(f"PASS: {mutation} mutation rejected by real retention randomized oracle")
