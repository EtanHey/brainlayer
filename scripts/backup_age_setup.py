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
        # Capture both item output and diagnostics: neither is safe to forward.
        template = """import json,re,subprocess,sys
try:
    key = sys.stdin.read()
    if not key.startswith('AGE-SECRET-KEY-'):
        raise ValueError()
    item = {'title':'BrainLayer backup age recovery','category':'SECURE_NOTE',
            'fields':[{'id':'identity','type':'CONCEALED','label':'age identity','value':key}]}
    result = subprocess.run(['op','item','create','-','--format=json'],
                            input=json.dumps(item),text=True,capture_output=True,check=False)
    if result.returncode:
        print('ERROR')
        sys.exit(result.returncode)
    item_id = json.loads(result.stdout).get('id')
    if not isinstance(item_id, str) or not re.fullmatch('[a-z0-9]{26}', item_id):
        raise ValueError()
except (OSError, ValueError, AttributeError):
    print('ERROR')
    sys.exit(1)
print('STORED id=' + item_id)
"""
        print(
            f"/usr/bin/security find-generic-password -s brainlayer-backup-age -w | python3 -c {shlex.quote(template)}"
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
