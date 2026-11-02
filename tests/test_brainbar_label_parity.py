"""Pin Swift labels to CHECK_LABELS at #1595 (63af543b).

Merge after #1595. Its snapshot wording is not on main yet, so this frozen
oracle permits testing the stack now without an xfail or weakening the comparison.
"""

import ast
import re
from pathlib import Path

# Exact CHECK_LABELS from 63af543b:src/brainlayer/health_verdict.py.
EXPECTED_CHECK_LABELS = {
    "backup_local": "Database snapshot (verified)",
    "backup_offsite": "Offsite (Drive) database backup",
    "transcripts": "Transcript backup",
    "drive_auth": "Drive authorization",
    "integrity": "Database integrity",
    "ingest": "Transcript ingest",
    "embedding": "Embedding",
    "stores": "Memory stores",
    "search": "Memory search",
    "disk": "Disk space for backups",
    "monitor": "Health monitor",
}


def test_brainbar_check_labels_match_python_contract():
    source = (
        Path(__file__).resolve().parents[1] / "brain-bar/Sources/BrainBar/BrainBarStatusVerdict.swift"
    ).read_text()
    dictionaries = re.findall(r"static let labels\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert len(dictionaries) == 1, "Expected exactly one Swift check-label dictionary"
    # Both languages use quoted string keys and values; literal_eval also rejects
    # unparsed entries rather than silently dropping unfamiliar syntax.
    labels = ast.literal_eval("{" + dictionaries[0] + "}")
    assert labels == EXPECTED_CHECK_LABELS
