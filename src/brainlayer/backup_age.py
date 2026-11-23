"""Streaming age encryption; no private identity is passed on a process argv."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

KEYCHAIN_SERVICE = "brainlayer-backup-age"


def config_dir() -> Path:
    return Path.home() / ".config" / "brainlayer"


def enabled() -> bool:
    return (config_dir() / "backup-age-recovery-confirmed").is_file()


def executable(name: str = "age") -> Path:
    # Homebrew's two supported macOS prefixes. Never search PATH.
    for prefix in ("/opt/homebrew", "/usr/local"):
        path = Path(prefix) / "bin" / name
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(f"{name} missing from Homebrew prefix")


def encrypt(path: Path) -> tuple[Path, dict]:
    try:
        recipient = (config_dir() / "backup-age-recipient.txt").read_text().strip()
    except OSError:
        raise RuntimeError("no age recipient configured") from None
    if not recipient.startswith("age1") or any(c.isspace() for c in recipient):
        raise RuntimeError("invalid age recipient")
    tool = executable()
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {"plaintext_sha256": digest, "plaintext_size": path.stat().st_size}
    target = path.with_name(path.name + ".age")
    with tempfile.NamedTemporaryFile(prefix=".age-", dir=path.parent, delete=False) as output:
        partial = Path(output.name)
        try:
            result = subprocess.run(
                [str(tool), "--encrypt", "--recipient", recipient, str(path)],
                stdout=output,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            output.flush()
            os.fsync(output.fileno())
            if result.returncode or partial.stat().st_size <= 0:
                raise RuntimeError("age encryption failed")
            with partial.open("rb") as check:
                if check.read(22) != b"age-encryption.org/v1\n"[:22]:
                    raise RuntimeError("invalid age ciphertext")
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
    return target, metadata


def private_identity() -> bytes | None:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True,
        check=False,
        timeout=15,
    )
    # Not found is normal on the M1. Other failures must not hide a broken test.
    if result.returncode == 44:
        return None
    if result.returncode:
        raise RuntimeError("backup age keychain lookup failed")
    return result.stdout


def decrypted_sha256(path: Path, identity: bytes) -> str:
    with subprocess.Popen(
        [str(executable()), "--decrypt", "--identity", "/dev/stdin", str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as process:
        try:
            process.stdin.write(identity)
            process.stdin.close()
            digest = hashlib.file_digest(process.stdout, "sha256").hexdigest()
            if process.wait():
                raise RuntimeError("age decrypt test failed")
            return digest
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
