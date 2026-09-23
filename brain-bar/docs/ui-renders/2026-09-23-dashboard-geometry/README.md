# Dashboard geometry render proof

Generated from source commit `ad800672` on `e07f5dd3`, before this artifact commit:

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=/tmp/brainbar-l0-pr-final/brainlayer.db BRAINBAR_RENDER_ONLY=/tmp/brainbar-l0-pr-render .build/debug/BrainBar
```

The DEBUG renderer runs before app startup, uses fixture data, and prohibits app activation. These are source-build pixels, not installed-app or live-data proof.

| Width | Collapsed | Expanded | After collapse |
| --- | --- | --- | --- |
| Compact, 760 × 640 pt | [PNG](dashboard-fixed-compact-collapsed.png) | [PNG](dashboard-fixed-compact-expanded.png) | [PNG](dashboard-fixed-compact-after-collapse.png) |
| Default, 960 × 640 pt | [PNG](dashboard-fixed-default-collapsed.png) | [PNG](dashboard-fixed-default-expanded.png) | [PNG](dashboard-fixed-default-after-collapse.png) |
| Wide, 1280 × 640 pt | [PNG](dashboard-fixed-wide-collapsed.png) | [PNG](dashboard-fixed-wide-expanded.png) | [PNG](dashboard-fixed-wide-after-collapse.png) |

The renderer reopened all nine PNGs for bitmap and color checks. The compact expanded and wide after-collapse images were visually inspected. At each width, the collapsed and after-collapse PNGs are byte-identical (`cmp`). The expanded image shows Details at the bottom of the fixed viewport; the collapsed image returns the header to its prior position.
