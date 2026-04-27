#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/brainlayer-fts5-determinism.XXXXXX")"
DB_PATH="$TMP_DIR/stale-index.db"
EXPECTED_PATH="$TMP_DIR/expected.json"
ACTUAL_RAW_PATH="$TMP_DIR/actual.raw.json"
ACTUAL_PATH="$TMP_DIR/actual.json"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cd "$ROOT_DIR"

uv run python3 - <<'PY' "$DB_PATH" "$EXPECTED_PATH"
from pathlib import Path
import sys

from tests.regression._stale_index_fixture import create_fixture_db, write_expected_ranking_json

db_path = Path(sys.argv[1])
expected_path = Path(sys.argv[2])
create_fixture_db(db_path)
write_expected_ranking_json(expected_path)
PY

QUERY_SQL="$(uv run python3 - <<'PY'
from tests.regression._stale_index_fixture import load_fixture

print(load_fixture()["sqlite_snapshot"]["query_sql"])
PY
)"

uvx --from sqlite-utils sqlite-utils query "$DB_PATH" "$QUERY_SQL" > "$ACTUAL_RAW_PATH"

uv run python3 - <<'PY' "$ACTUAL_RAW_PATH" "$ACTUAL_PATH"
import json
from pathlib import Path
import sys

raw_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.write_text(json.dumps(json.loads(raw_path.read_text()), indent=2, sort_keys=True) + "\n")
PY

diff -u "$EXPECTED_PATH" "$ACTUAL_PATH"
