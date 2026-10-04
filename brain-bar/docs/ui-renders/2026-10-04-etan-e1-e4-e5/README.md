# Etan rows E1, E4, E5: Backups checks, status-dot alignment, one Config file row

Etan's feedback, 2026-10-01: the Backups page's bottom status list was "very ad-hoc-ish… very bad UI
or UX or data visualization" (E1); the green dot was not vertically centred with "Google Drive"
(E4); the `brainlayer.env` path under every page title was "repetitive… I don't know what it even
means" (E5).

DEBUG render-harness captures (`renderEtanRowsE1E4E5`), fixtures only. **Before** is the same
render pass built on this PR's base (#1062 head `6b7aa3bc`); **after** is this branch.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

## E1: Backups recovery checks
The "needs attention" fixture has the transcript job parked by the retention safety stop, a copy
over the 36 h limit, and an unverified upload whose archive ID is `1-QM42…`.

| | Before | After |
|---|---|---|
| Needs attention | [760](before-backups-attention-compact-dark.png) · [960](before-backups-attention-default-dark.png) · [1280](before-backups-attention-wide-dark.png) | [760](after-backups-attention-compact-dark.png) · [960](after-backups-attention-default-dark.png) · [1280](after-backups-attention-wide-dark.png) |
| Technical details open | n/a | [960](after-backups-attention-details-default-dark.png) |
| Healthy | n/a | [960](after-backups-healthy-default-dark.png) |

- Six checks with human labels and plain states ("Verified 1 hour ago", "Paused by a safety stop",
  "A copy is older than 36 hours"), relative times, and one verdict ("3 of 6 checks need attention").
- No Drive ID, file name or launchd label in the default view. They sit under **Technical
  details** (closed by default), each value selectable, each with Copy.
- The Backups badge's reason uses the same words ("Transcripts in Google Drive: Uploaded 1 day ago,
  not verified"). Each check keeps the tone of the status line it replaces, so the badge and the list
  cannot disagree (#1029 B1, pinned by `test_every_check_tone_matches_the_status_line_it_replaces`).
- A job alert (#1062) keeps its own card at the top and is not repeated in the checks.

## E4: status dot centred on its label
[Before](before-drive-card-default-dark.png) · after at [760](after-drive-card-compact-dark.png) ·
[960](after-drive-card-default-dark.png) · [1280](after-drive-card-wide-dark.png), drawn at 2× zoom.
Measured on these renders, dot centre minus the cap-height centre of "G": **before +1.00 pt
(low) at every width, after 0.00 pt.** `BrainBarStatusDot` centres the dot on an invisible
line of the label's own font. It also fixes the same bug on the backup status rows, the
Dashboard status strip (`.top`-aligned, so the dot rode high) and the attention items.
`BrainBarStatusDotTests` measures it in pixels at all five sizes the app uses (≤ 0.5 pt), and
shows the old bare `Circle` failing that bar.

## E5: no path under page titles; one Config file row
[Before](before-advanced-default-dark.png) · after at [760](after-advanced-compact-dark.png) ·
[960](after-advanced-default-dark.png). The path is gone from every page header (Jobs, Backups,
Advanced). Advanced ends with one **Config file** row: what the file is, the path (selectable), and
Copy path / Reveal in Finder. The Backups schedule's local-copy file names are now selectable too.
Each already had Copy path.

## Light and dark
Every capture was rendered under the `darkAqua` and `aqua` system appearances
([light example](after-backups-attention-default-light.png)). They match pixel for pixel except
the few digits of the Drive card's "renews by …" time, which comes from the wall clock at render
time. BrainBar pins its own dark scheme.

## Round 1 (lead UX)
- **Each failure shows once.** The Backups header keeps only its badge. Any reason that another card
  already states (a recovery check, the Google Drive card or the job-alert card) is not repeated
  under it. A reason that appears nowhere else, such as the launchd job's exit, still shows.
  Test: `test_the_backups_header_never_repeats_a_reason_the_page_already_shows`.
- **No monospace `LAST RUN` / `NEXT RUN` rows** on the Backups page: the Schedule section is the one
  source of last/next runs. The Jobs page keeps them.
- **Coherent fixture times:** every time in these renders comes from the fixture's single clock (the
  Schedule rows, the checks, Technical details, the local-copy names and the Drive renewal).
  A parked job reads "Not scheduled" with no next run. Light and dark Backups captures are now
  byte-identical (the wall-clock minute is gone).
- **Harness OCR (review N1):** the fixed-height Dashboard check now requires a label to be missing
  on three Vision passes before it fails. The full harness exits 0 on two consecutive runs.

The `after-*` images above are the round-1 renders. Healthy is shown at
[760](after-backups-healthy-compact-light.png) · [960](after-backups-healthy-default-light.png) ·
[1280](after-backups-healthy-wide-light.png) under the light appearance too.
