"""scripts/score_entity_health.py must write only to the database its --db flag names.

Its docstring advertised `[--db PATH] [--contracts PATH]`, but main() ignored argv and always
upserted entity_health into the resolved default database.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "score_entity_health.py"


def test_db_flag_selects_the_scored_database(tmp_path):
    default_db = tmp_path / "default" / "brainlayer.db"
    chosen_db = tmp_path / "chosen" / "brainlayer.db"
    default_db.parent.mkdir()
    chosen_db.parent.mkdir()
    env = dict(os.environ, BRAINLAYER_DB=str(default_db), PYTHONPATH=str(REPO_ROOT / "src"))

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(chosen_db)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"Opening database at {chosen_db}" in completed.stdout
    assert chosen_db.exists()
    assert not default_db.exists(), "the --db run touched the default database"


def test_unknown_flag_is_rejected(tmp_path):
    env = dict(os.environ, BRAINLAYER_DB=str(tmp_path / "unused.db"), PYTHONPATH=str(REPO_ROOT / "src"))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--database", str(tmp_path / "x.db")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    assert not (tmp_path / "unused.db").exists()
