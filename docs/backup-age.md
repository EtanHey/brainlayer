# Transcript backup encryption

JSONL backup encryption uses Homebrew's `age` CLI at `/opt/homebrew/bin/age`
or `/usr/local/bin/age`, never PATH. The release must add `depends_on "age"`
to the brainlayer formula in EtanHey/homebrew-layers before enabling it.
Both encryption and decrypt verification stream, including multi-GB bundles.

The recovery gate is `~/.config/brainlayer/backup-age-recovery-confirmed`.
Until that file exists, existing uploads continue and each run logs
`encryption: not_enabled` without a job alert. Enabled runs log `encryption: enabled`.
Any older setup alert is cleared. After confirmation, missing recipients,
missing age, or failed encryption block upload, retain the local tarball and
raise `JSONL backup not encrypted: <reason>`. iCloud and optional forever uploads
also receive ciphertext. Existing remote plaintext backups remain untouched.

The lead runs `scripts/backup_age_setup.py` once on the M4 after merge. It refuses
to rotate existing keys, generates an age identity in memory, sends the private
key through security's stdin into service `brainlayer-backup-age`, verifies the
keychain value and writes only the public recipient to
`~/.config/brainlayer/backup-age-recipient.txt`. Output contains only the public
key. Copy that public recipient to the M1. This script never creates the gate.

`scripts/backup_age_setup.py --recovery-command` prints the command for orc to
run with Etan present. It transfers the keychain value through stdin as a JSON
1Password item with the identity in a `CONCEALED` field. The command captures
both create stdout and stderr; it prints only `STORED id=<26-character item ID>`
or `ERROR`, preserving a failed create's exit status. Invalid JSON or item IDs
fail closed without forwarding item content or diagnostics. Only after
successful recovery-copy creation and verification
should the lead create the recovery-confirmation marker on each Mac.

Receipts retain ciphertext `name`, `size`, `sha256`, `archive_md5`, and Drive ID;
`plaintext_sha256` and `plaintext_size` describe the original tarball. A small
`.age.json` sidecar retains those plaintext measurements across interrupted
uploads for receipt reconciliation. Local pruning still requires 14 days plus
the exact live ciphertext ID, size and hashes. Plaintext is removed only after
the ciphertext and its metadata are durably written.

On a scheduled backup run, at most once per successful week, a Mac with the
keychain identity decrypts its newest local, verified uploaded ciphertext and
compares its digest to the receipt. Decrypt failures and digest mismatches remain
alerting and are retried on the next run. A missing identity or a locked/unavailable
keychain skips without an alert or advancing the successful-test timestamp, so
the next run retries. This does not download
or rewrite any existing Drive object. Setup/keychain integration and real Drive
operation require release validation; tests use throwaway keys and local fixtures.
