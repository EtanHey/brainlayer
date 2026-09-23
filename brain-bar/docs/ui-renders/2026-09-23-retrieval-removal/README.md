# BrainBar retrieval UI removal: Settings and header

Source-built DEBUG render of the unified Settings window after removal. The header shows only Dashboard and Settings; the General section no longer offers the retrieval toggle. Rendered with `BRAINBAR_RENDER_ONLY=/tmp/brainbar-vnext-phase3-render .build/out/Products/Debug/BrainBar` after building with `DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer`. The full render harness exited 0. These are fixture renders, not an installed-app claim.

| Width | Render | SHA-256 |
| --- | --- | --- |
| 760 pt | [compact](unified-settings-general-compact.png) | `865e237822c3597055556466abb5a7ba0f880c44d7ba5a3f5ad7ab5040c6b157` |
| 960 pt | [default](unified-settings-general-default.png) | `ea13cde4782472307202f704348f6731ae4b0c96df1c1fb448c0020845b6bf09` |
| 1280 pt | [wide](unified-settings-general-wide.png) | `e2fb765f533627bfcb3af3a583a6fb152dc5f4fbe10544bc0d547d13144f4e78` |

I inspected the compact and wide PNGs: the two destinations fit in one header row; the General page has no retrieval control, graph destination, or command bar.
