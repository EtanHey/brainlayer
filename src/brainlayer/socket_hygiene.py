"""Refuse production BrainBar connections when the unit-test guard is armed."""

import os
from pathlib import Path


def refuse_production_brainbar_socket(address: str | os.PathLike[str]) -> None:
    """Keep the guard effective when a child replaces its test PYTHONPATH."""
    if os.environ.get("BRAINLAYER_FORBID_BRAINBAR_SOCKET") != "1":
        return
    resolved = Path(os.path.realpath(os.fspath(address)))
    tmp_dir = Path(os.path.realpath("/tmp"))
    if resolved == tmp_dir / "brainbar.sock" or (
        resolved.parent == tmp_dir and resolved.name.startswith("brainbar-hybrid-") and resolved.name.endswith(".sock")
    ):
        raise RuntimeError(f"unit test refused production BrainBar socket: {address}")
