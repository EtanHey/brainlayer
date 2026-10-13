# Backup observation and consolidated logs

Before migration, the JSONL backup observer probes its legacy launchd label. When
launchctl confirms that service is absent, its plist is archived, and the
maintenance plist exists, it probes `com.brainlayer.maintenance` instead. Other
errors remain unmeasurable. The frozen schema-v1 `launchd.label` remains the
logical JSONL backup key; `bootstrapped` describes the selected launcher. The
actual command and both definition paths appear in `inputs`, including failures.
Upload freshness still requires backup receipts; a loaded launcher proves no upload.

After archival, run the log cap with explicit `--manifest <jobs.json>`,
`--calendar-logs <directory>`, and repeated `--backup-log <receipt-log>` options.
PR5 must supply these paths to the supervised log-cap component. Legacy plist
log discovery remains supported. Caps preserve open append inodes and reject
symlinks, non-owner files, and hardlinks. Backup tails retain whole JSONL records,
expanding to the final receipt when it fits the maximum size. If no complete
receipt fits that maximum, the cap refuses before truncation. An incomplete
final backup record also refuses before truncation; retry after the append finishes.
Both backup receipt writers share an exclusive inode lock with the cap; a busy
receipt log refuses capping. The cap also retains the latest real-provenance
daily receipt, or refuses if its suffix exceeds the maximum. Unmarked JSONL
archive receipts remain supported. These guarantees require the updated writers;
do not cap receipt logs still being written by an older installed version.

These options do not activate a launcher or migrate installed files.
