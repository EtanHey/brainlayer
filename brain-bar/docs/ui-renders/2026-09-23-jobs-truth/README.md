# Jobs truth render review

Source-built DEBUG render of Settings › Jobs at 760, 960, and 1280 pt wide, each 720 pt high and 2× scale:

- [Compact](unified-settings-jobs-compact.png)
- [Default](unified-settings-jobs-default.png)
- [Wide](unified-settings-jobs-wide.png)

The fixture shows an index exit code 1 with its dated run time, a scheduled nightly job with counters reset to zero, and date-plus-time next runs. The red reason and neutral “Awaiting next run” badge fit at all three widths. Images were visually inspected after rendering with `BRAINBAR_RENDER_ONLY`; they prove source presentation only, not installed app behavior. Real run receipts begin when the updated launchd wrapper is installed and each job next starts; older output file timestamps are never substituted as run dates.
