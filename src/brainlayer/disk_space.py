"""One volume-space measurement: Available includes macOS purgeable capacity."""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class DiskSpace:
    available_bytes: int
    free_now_bytes: int
    purgeable_bytes: int
    source: str
    measured_at: str

    def describe(self) -> str:
        label = "macOS Available" if self.source == "macos_important_usage" else "Available"
        return (
            f"{label} {self.available_bytes / 1e9:.1f} GB = "
            f"{self.free_now_bytes / 1e9:.1f} GB free now + {self.purgeable_bytes / 1e9:.1f} GB purgeable"
        )


def _important_usage_capacity_bytes(path: Path) -> int | None:
    """Ask macOS for capacity including purgeable space; None means use statfs."""
    if sys.platform != "darwin":
        return None
    script = (
        'function run(argv) { ObjC.import("Foundation"); '
        "var url = $.NSURL.fileURLWithPath(argv[0]); "
        "var key = $.NSURLVolumeAvailableCapacityForImportantUsageKey; "
        "var values = url.resourceValuesForKeysError($.NSArray.arrayWithObject(key), null); "
        'if (!values) return "unavailable"; '
        "var number = values.objectForKey(key); "
        'return number ? ObjC.unwrap(number).toString() : "unavailable"; }'
    )
    try:
        output = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", script, str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return int(output) if output.isdecimal() else None
    except TimeoutError:
        raise
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def measure_disk_space(path: Path, *, include_available: bool = True) -> DiskSpace:
    """Measure once; raw-only mode keeps per-chunk safety checks cheap.

    Non-macOS and explicitly labeled probe failures use Available = free now.
    Important-usage capacity is an estimate, never an allocation guarantee.
    """
    path = Path(path).expanduser().resolve()
    free = shutil.disk_usage(path).free
    important = None
    if include_available and sys.platform == "darwin":
        try:
            important = _important_usage_capacity_bytes(path)
        except TimeoutError:
            raise
        except (OSError, RuntimeError, ValueError):
            pass
    source = (
        "statfs_raw_only"
        if not include_available
        else "statfs_non_macos"
        if sys.platform != "darwin"
        else "statfs_macos_available_unavailable"
        if important is None
        else "macos_important_usage"
    )
    available = max(important, free) if important is not None else free
    return DiskSpace(available, free, max(0, available - free), source, datetime.now(UTC).isoformat())
