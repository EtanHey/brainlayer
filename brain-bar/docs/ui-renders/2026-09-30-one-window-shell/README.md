# One window + status item: shell (#963 PR 1, vNext D5)

These are DEBUG render-harness captures (`BRAINBAR_RENDER_ONLY`, `renderMainWindowShell`) made from fixture data. They are not the installed app. The installed-app window and menu-bar capture belong to the 1g release QA, under the UI-TEST lock.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

## The window
The production controller builds the real `BrainBarMainWindow` from a fixture runtime. It is captured with its frame (traffic lights and title bar) at 760, 960 and 1280 × 640 pt:
- [760](window-dashboard-compact.png)
- [960](window-dashboard-default.png)
- [1280](window-dashboard-wide.png)

The content is unchanged in this PR: the top Dashboard/Settings tabs move into the sidebar in PR 2.

The "status unavailable" and "Database path unavailable" rows appear because the fixture runtime has no database path. They are not a regression.

## The status item
The icon is unchanged: the live sparkline, plus the attention badge. It is shown 8× on a dark strip: [normal](status-item-icon.png), [needs attention](status-item-icon-attention.png).

Every click opens this menu. Its titles are pinned by `BrainBarMainWindowTests.test_the_status_item_opens_a_small_menu_like_voicebar`:

```
Needs attention: <reason>    ← or "Nothing needs attention"; disabled status line
────────────
Open Dashboard
Settings…
────────────
Restart BrainBar
────────────
Quit BrainBar
```
