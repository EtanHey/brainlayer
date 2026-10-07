"""Historical enrichment instructions are labeled; restore examples cannot revive jobs."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIRS = {"adr", "archive", "architecture", "operations", "plans", "research"}
HISTORY_PAGES = {"enrichment-provider-agnostic-design.md", "brew-layer-conformance.md", "data-locations.md"}
PRODUCER_REFERENCE = re.compile(r"\benrichment\b|enrich_realtime|enrich_limit|brainlayer\s+enrich\b", re.I)
HISTORICAL = [
    path
    for path in sorted((ROOT / "docs").rglob("*.md"))
    if (path.relative_to(ROOT / "docs").parts[0] in HISTORY_DIRS or path.name in HISTORY_PAGES)
    and PRODUCER_REFERENCE.search(path.read_text())
]


@pytest.mark.parametrize("path", HISTORICAL, ids=lambda p: str(p.relative_to(ROOT)))
def test_historical_enrichment_references_are_retired_before_instructions(path):
    preamble = path.read_text().split("\n## ", 1)[0]
    assert "LLM enrichment is retired" in preamble
    assert "History:" in preamble and "enrichment.md" in preamble


@pytest.mark.parametrize("retired", ["com.brainlayer.enrichment", "com.brainlayer.gemini-loopback"])
def test_documented_wave3_restore_skips_retired_job_even_without_pause(tmp_path, retired):
    text = (ROOT / "docs/operations/wave3-3a-two-host-live-runbook.md").read_text()
    section = text.split("## 6. Restart", 1)[1]
    script = section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "calls"
    calls.write_text("")
    # The runbook targets macOS; GNU tail on Linux runners has no -r option.
    tail = binaries / "tail"
    tail.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "-r" ] || exit 2\n'
        "awk '{ rows[NR] = $0 } END { for (i = NR; i > 0; i--) print rows[i] }' \"$2\"\n"
    )
    tail.chmod(0o700)
    launcher = binaries / "launchctl"
    launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    launcher.chmod(0o700)
    python_stub = binaries / "unpaused-python"
    python_stub.write_text("#!/bin/sh\nexit 1\n")
    python_stub.chmod(0o700)
    labels = [retired, "com.brainlayer.brainbar-daemon"]
    (tmp_path / "stopped-labels").write_text("\n".join(labels) + "\n")
    plists = tmp_path / "Library/LaunchAgents"
    plists.mkdir(parents=True)
    for label in labels:
        (plists / f"{label}.plist").write_text("synthetic")
    env = {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CALLS": str(calls),
        "WAVE3A_RUN_DIR": str(tmp_path),
        "WAVE3A_PYTHON": str(python_stub),
    }
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    recorded = calls.read_text()
    assert "brainbar-daemon.plist" in recorded
    assert retired not in recorded
