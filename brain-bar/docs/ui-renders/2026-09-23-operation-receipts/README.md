# Operation receipt render proof

Generated from `a787b436` rebased onto `f5994eb5` before the artifact-only commit, using `DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar` and `BRAINBAR_RENDER_ONLY=/tmp/brainbar-l4-pr-render .build/debug/BrainBar`. The DEBUG harness runs before app startup with prohibited activation and fixture data; these are source-build pixels, not installed or real-client proof.

| State | Render |
| --- | --- |
| Compact (760 pt) | [populated](dashboard-cli-compact-readable.png) |
| Default (960 pt) | [populated](dashboard-cli-default-readable.png), [Details expanded](dashboard-cli-default-readable-details-expanded.png), [unavailable](dashboard-cli-default-receiptUnavailable.png), [failed and aged](dashboard-cli-default-receiptFailed.png) |
| Wide (1280 pt) | [populated](dashboard-cli-wide-readable.png) |

All six PNGs were reopened during rendering for bitmap/color checks. The compact, wide, expanded, and failed PNGs were visually inspected; the receipt line remains inside the Ingest header and the failed age is readable. The lead's real-client MCP check remains required before merge.
