#!/usr/bin/env python3
"""One-time M4 setup. Recovery confirmation is deliberately a separate owner step."""

import argparse
import shlex
import subprocess

from brainlayer import backup_age


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-command", action="store_true")
    args = parser.parse_args()
    if args.recovery_command:
        template = (
            "import json,sys; key=sys.stdin.read(); assert key.startswith('AGE-SECRET-KEY-'); "
            "print(json.dumps({'title':'BrainLayer backup age recovery',"
            "'category':'SECURE_NOTE','fields':[{'id':'notesPlain','type':'STRING',"
            "'purpose':'NOTES','value':key}]}))"
        )
        print(
            "/usr/bin/security find-generic-password -s brainlayer-backup-age -w | "
            f"python3 -c {shlex.quote(template)} | op item create -"
        )
        return
    recipient_path = backup_age.config_dir() / "backup-age-recipient.txt"
    if recipient_path.exists() or backup_age.private_identity() is not None:
        raise RuntimeError("backup age key already exists; refusing to rotate")
    generated = subprocess.run([str(backup_age.executable("age-keygen"))], capture_output=True, check=True)
    identity = next(line for line in generated.stdout.decode().splitlines() if line.startswith("AGE-SECRET-KEY-"))
    public = (
        subprocess.run(
            [str(backup_age.executable("age-keygen")), "-y"],
            input=generated.stdout,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    # security's interactive command channel keeps the secret off process argv.
    command = "add-generic-password -a brainlayer -s brainlayer-backup-age -w " + shlex.quote(identity) + "\n"
    stored = subprocess.run(["/usr/bin/security", "-i"], input=command.encode(), capture_output=True, check=False)
    if stored.returncode or backup_age.private_identity().decode().strip() != identity:
        raise RuntimeError("keychain write verification failed")
    recipient_path.parent.mkdir(parents=True, exist_ok=True)
    with recipient_path.open("x") as handle:
        handle.write(public + "\n")
    print(public)


if __name__ == "__main__":
    main()
