"""Release must omit the debug adapter and ignore a present unsafe clock file."""

import subprocess
import sys
import tempfile
from pathlib import Path

directory = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(subprocess.check_output(["swift", "build", "-c", "release", "--show-bin-path"], text=True).strip())
)
for name in ("BrainBar", "BrainBarDaemon"):
    binary = directory / name
    evidence = subprocess.check_output(["/usr/bin/strings", str(binary)], text=True)
    evidence += subprocess.check_output(["/usr/bin/nm", str(binary)], text=True)
    assert "BRAINBAR_RECOVERY_TEST_CLOCK" not in evidence, name
    assert "processFixture" not in evidence, name
scripts = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="bb-release-clock-") as root:
    library = Path(root) / "guard.dylib"
    subprocess.run(
        ["/usr/bin/clang", "-dynamiclib", "-lsqlite3", str(scripts / "daemon_shutdown_guard.c"), "-o", str(library)],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/python3",
            str(scripts / "daemon_reconnect_process.py"),
            str(directory / "BrainBarDaemon"),
            str(library),
            "RELEASE",
        ],
        check=True,
        timeout=20,
    )
print("Release clock adapter absent; real private release daemon ignores the unsafe clock file.")
