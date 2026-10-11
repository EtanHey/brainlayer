# Settings footer and model residency renders

Source-built DEBUG-only render matrix from the rebased L6 code at `bec62cb4` (before this artifact commit). The [L5 Settings sidebar renders](../2026-09-23-settings-sidebar/README.md) are the before state. These 15 images show General, Jobs, Backups, Advanced, and a save receipt at 760, 960, and 1280 pt wide; each is 720 pt high with 2× backing pixels.

The footer stays on one row and reports `Memory on this Mac · Enrichment → Gemini · Backups → Drive` for the fixture. Its cloud icon reflects the configured enrichment and Drive jobs. The Advanced rows explicitly mark embedding-model name, residency, and resident memory unavailable. The save receipt is a fixture acknowledgment, not evidence of a live config write.

| State | Width | Image | SHA-256 |
|---|---|---|---|
| Advanced | 760 pt / 1520 px | [unified-settings-advanced-compact](unified-settings-advanced-compact.png) | `272c378b38c2d54eaff113ef84c4df28c81a46abd3eed2290409369795ef8c20` |
| Advanced | 960 pt / 1920 px | [unified-settings-advanced-default](unified-settings-advanced-default.png) | `c619a2fffc53e748ecb53067a505d006a7fea2090122a98fd5f1d7dd0887f5ab` |
| Advanced | 1280 pt / 2560 px | [unified-settings-advanced-wide](unified-settings-advanced-wide.png) | `10611124bba6ef2f1192c8882fc81730a7ba0909a622ed57282520bf9da3d31a` |
| Backups | 760 pt / 1520 px | [unified-settings-backups-compact](unified-settings-backups-compact.png) | `09d707fca237512a1fbbc190806b83d7fa72a71a696a70f7740a7c2938d37c77` |
| Backups | 960 pt / 1920 px | [unified-settings-backups-default](unified-settings-backups-default.png) | `830e075755c4cffb2303b7d7989ae4e8b58cbb9711fe7038646ccf5cd4bbda35` |
| Backups | 1280 pt / 2560 px | [unified-settings-backups-wide](unified-settings-backups-wide.png) | `2f45698a3272be0c334c4ce60c3f8d27612cfe548d539bf946eca964a8d568bf` |
| General | 760 pt / 1520 px | [unified-settings-general-compact](unified-settings-general-compact.png) | `b44a177830fc3b60e75e947f8ab417a95c139e03922ad7d6e1e6a2cd4d65c12d` |
| General | 960 pt / 1920 px | [unified-settings-general-default](unified-settings-general-default.png) | `22ac96e6ed1a3599ba8bba3fca6a07c97389ddd3a4815f63c48b65fb4c462a1d` |
| General | 1280 pt / 2560 px | [unified-settings-general-wide](unified-settings-general-wide.png) | `f0985a6d30cba6b83c1e07e0feafc7237a34bf397e88861f650e0fe08a2a9532` |
| Jobs | 760 pt / 1520 px | [unified-settings-jobs-compact](unified-settings-jobs-compact.png) | `909c4390dcb5fa9674c727b7dcf6314e3fda84c43e2b493b568e492e466dfd6d` |
| Jobs | 960 pt / 1920 px | [unified-settings-jobs-default](unified-settings-jobs-default.png) | `a14dcda4e0ebd9f4841d3d493709cae604358ada395af023ef22da2298032ce2` |
| Jobs | 1280 pt / 2560 px | [unified-settings-jobs-wide](unified-settings-jobs-wide.png) | `5ccf31de7a5c511fb0d4f8a0d61638278e67b6c7ec4e7074a55a1d0d57f58ade` |
| Receipt | 760 pt / 1520 px | [unified-settings-receipt-compact](unified-settings-receipt-compact.png) | `5b506c19453c770705e1c5cba57654fbed81749ef25aca02dd4decdec215bfca` |
| Receipt | 960 pt / 1920 px | [unified-settings-receipt-default](unified-settings-receipt-default.png) | `f50597f802dfe62c212af6622ac938fbbd348a2522c60f3e2790378ad4d92ba7` |
| Receipt | 1280 pt / 2560 px | [unified-settings-receipt-wide](unified-settings-receipt-wide.png) | `5d43a296f16d1da5d0752db0e64debb95d121c1b1402f211d67d71f382f34a7d` |

Renderer: `BRAINBAR_RENDER_ONLY=/tmp/brainbar-l6-pr-render swift run BrainBar` with Xcode 26 developer tools. It runs before `App.main()` with activation prohibited and a temporary fixture config. It did not open the canonical database, live socket, installed app, or production GUI. Visual inspection covered compact General and wide Advanced on the rebased code; the prior accepted L6 review inspected the full section/width matrix.
