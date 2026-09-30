# Reconnect Google Drive (Backups page + Dashboard banner)

These are DEBUG render-harness captures from fixture states set through `BrainBarDriveAuthModel.setForPreview`. No CLI runs and no token is read. They are not the installed app.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

| State (`brainlayer backup auth --status --json`) | Backups page | Button |
|---|---|---|
| `valid` | [Connected: renews by <date>](backups-connected-default.png) | none |
| `expiring` (under 24 h left, day 6 of the 7-day Testing-mode consent) | "Drive access expires in 6 h. Reconnect", amber, at [760](backups-expiring-compact.png), [960](backups-expiring-default.png) and [1280](backups-expiring-wide.png) | Reconnect Google Drive |
| `missing` | ["Google Drive is not connected. Backups can't upload."](backups-missing-default.png) plus the CLI's reason | Reconnect Google Drive |
| `invalid`, after a cancelled reconnect | ["Drive access expired or was revoked…"](backups-invalid-cancelled-default.png) plus "Reconnect was cancelled: <reason>" | still shown |
| reconnect running | ["Waiting for Google consent in your browser…"](backups-reconnecting-default.png) | disabled, with a spinner |

**Dashboard banner:** [expiring, in the real window](dashboard-banner-expiring-default.png). It sits under the status strip and shows only when Drive needs a click. The button is grey in this capture only because an off-screen window is never the key window.
