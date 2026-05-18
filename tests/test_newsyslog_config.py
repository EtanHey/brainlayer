from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "launchd" / "newsyslog.d" / "brainlayer.conf"
INSTALLER = REPO_ROOT / "launchd" / "install-newsyslog.sh"


def _entries() -> list[list[str]]:
    return [
        line.split() for line in CONFIG.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def test_newsyslog_config_uses_user_writable_rotated_logs():
    entries = _entries()
    assert entries
    for fields in entries:
        assert fields[0].startswith("/Users/etanheyman/Library/Logs/brainlayer/")
        assert fields[0].endswith(".log")
        assert fields[1] == "etanheyman:staff"
        assert fields[2] == "644"
        assert fields[6] == "JN"


def test_newsyslog_config_covers_launchd_log_pairs():
    names = {Path(fields[0]).name for fields in _entries()}
    for daemon in {
        "backup-daily",
        "decay",
        "wal-checkpoint",
        "index",
        "repair-fts",
    }:
        assert f"{daemon}.out.log" in names
        assert f"{daemon}.err.log" in names


def test_newsyslog_config_excludes_held_open_long_running_launchd_logs():
    names = {Path(fields[0]).name for fields in _entries()}
    for daemon in {"brainbar", "enrichment", "watch", "drain"}:
        assert f"{daemon}.out.log" not in names
        assert f"{daemon}.err.log" not in names

    config = CONFIG.read_text()
    assert "post-rotate hook or copy-truncate mode" in config


def test_newsyslog_installer_documents_root_owned_replacement_footgun():
    readme = (REPO_ROOT / "launchd" / "newsyslog.d" / "README.md").read_text()
    assert "root:admin" in readme
    assert "etanheyman:staff" in readme
    assert "Long-running jobs" in readme
    assert "post-rotate hook" in readme
    assert "sudo newsyslog -nv -f /etc/newsyslog.d/brainlayer.conf" in readme


def test_newsyslog_installer_repairs_root_owned_logs_for_invoking_user():
    script = INSTALLER.read_text()
    assert 'OWNER="${BRAINLAYER_LOG_OWNER:-${SUDO_USER:-$(id -un)}}"' in script
    assert 'sudo mkdir -p "$LOG_DIR"' in script
    assert '[ -L "$log" ] || [ ! -f "$log" ]' in script
    assert "Skipping non-regular log path: $log" in script
    assert 'sudo chown "$OWNER:$GROUP" "$log"' in script
    assert 'sudo chmod 0644 "$log"' in script


def test_newsyslog_installer_renders_runtime_owner_and_log_dir():
    script = INSTALLER.read_text()
    assert 'mktemp "${TMPDIR:-/tmp}/brainlayer-newsyslog.XXXXXX"' in script
    assert "trap 'rm -f \"$RENDERED_CONFIG\"' EXIT" in script
    assert 'LOG_DIR_ESCAPED="$(escape_sed_replacement "$LOG_DIR")"' in script
    assert 'OWNER_GROUP_ESCAPED="$(escape_sed_replacement "$OWNER:$GROUP")"' in script
    assert "s#/Users/etanheyman/Library/Logs/brainlayer#$LOG_DIR_ESCAPED#g" in script
    assert "s#etanheyman:staff#$OWNER_GROUP_ESCAPED#g" in script
    assert 'sudo newsyslog -nv -f "$RENDERED_CONFIG"' in script
    assert 'sudo install -o root -g wheel -m 0644 "$RENDERED_CONFIG" "$DST"' in script


def test_newsyslog_installer_handles_home_paths_with_spaces_explicitly():
    script = INSTALLER.read_text()
    assert "awk '{print $2}'" not in script
    assert 'if ! OWNER_HOME="$(dscl . -read "/Users/$OWNER" NFSHomeDirectory' in script
    assert "sed 's/^NFSHomeDirectory:[[:space:]]*//'" in script
    assert '[[ "$LOG_DIR" =~ [[:space:]] ]]' in script
    assert "newsyslog log paths cannot contain whitespace" in script
