# Settings sidebar visual evidence

Source-built, headless fixture renders for the L5 Settings sidebar. The three before images show the unified Settings view at the Settings base (`e6f79b88`) before the sidebar. The 15 after images show accepted L5 polish (`504678e6`) across General, Jobs, Backups, Advanced, and a save-receipt state.

Widths are 760 pt (compact), 960 pt (default), and 1280 pt (wide), all at 720 pt height. The PNGs use 2× backing resolution. The save-receipt fixture shows a successful save acknowledgment; it does not prove a live configuration write.

| State | Image | Pixels | SHA-256 |
|---|---|---:|---|
| Before | [unified-settings-compact](before/unified-settings-compact.png) | 1520×1440 | `b25570011e94847bb8d428ad068c547742ce5000b06e31ee1454d2ba9509a9ed` |
| Before | [unified-settings-default](before/unified-settings-default.png) | 1920×1440 | `9c6efb3bf2cf43e81f5808b55f7b6bda8f9f03493a9f9335a18c1062707b3900` |
| Before | [unified-settings-wide](before/unified-settings-wide.png) | 2560×1440 | `bed390e9a1462d38802fd2706abedc164d8c80007f5fdeb8197e84a780d37d79` |
| After | [unified-settings-advanced-compact](after/unified-settings-advanced-compact.png) | 1520×1440 | `404ef275db7d1def43c0725adbf1931089102eed0cb9841f7d7c0aac1a92b877` |
| After | [unified-settings-advanced-default](after/unified-settings-advanced-default.png) | 1920×1440 | `e53144e662207bca4801bc9c235acf58cb39e9d8492a9dc4b61199b53b664f53` |
| After | [unified-settings-advanced-wide](after/unified-settings-advanced-wide.png) | 2560×1440 | `f7e6b6c916e906f7897ae367658a6054066d11532e7d877f4a77ce2bef2b3422` |
| After | [unified-settings-backups-compact](after/unified-settings-backups-compact.png) | 1520×1440 | `2d36002e892c699b9c3c44430b733741ba24b797c72e05993a5e197d0c0381d3` |
| After | [unified-settings-backups-default](after/unified-settings-backups-default.png) | 1920×1440 | `b8afb4fc616e70be6fe3fed6e6c1d0e1581a91668e29862b4f6a0ad3b074c626` |
| After | [unified-settings-backups-wide](after/unified-settings-backups-wide.png) | 2560×1440 | `6fdc8ff74538d58869648dbd1a2983750ffe494d993ea358b86f1f77549204da` |
| After | [unified-settings-general-compact](after/unified-settings-general-compact.png) | 1520×1440 | `e8b1f65cec38bbb618f96155de68949ca000a76b9bddc37930fef5d8834927ee` |
| After | [unified-settings-general-default](after/unified-settings-general-default.png) | 1920×1440 | `50ddac47113e11a7826a209f6a7510e0a3dce935874740884a748a5127982ed6` |
| After | [unified-settings-general-wide](after/unified-settings-general-wide.png) | 2560×1440 | `acf1f0802d261021b419479d5782cf9cd90d3f2aa23fe86eb438f42f5350d400` |
| After | [unified-settings-jobs-compact](after/unified-settings-jobs-compact.png) | 1520×1440 | `b5c89061ce36b404327ea9ef3b3306f5ab846a68e6aadd7993e6b626d53fa891` |
| After | [unified-settings-jobs-default](after/unified-settings-jobs-default.png) | 1920×1440 | `efcfce1c1000229a612f294decb39e17f44b5d172e1487992e67f0082b66b384` |
| After | [unified-settings-jobs-wide](after/unified-settings-jobs-wide.png) | 2560×1440 | `02be0f05113a73b018992ed17611475b471a6034fb9c6d75d21b3646593408e6` |
| After | [unified-settings-receipt-compact](after/unified-settings-receipt-compact.png) | 1520×1440 | `7482d02e47ecee2210cb089d3ebc651da41228835e4e76baa47fb2e8b1ae5cb0` |
| After | [unified-settings-receipt-default](after/unified-settings-receipt-default.png) | 1920×1440 | `d212efcb6616ef74a1648b0a0f551ea90bf0cd6e6edcaa855fa958dda802749f` |
| After | [unified-settings-receipt-wide](after/unified-settings-receipt-wide.png) | 2560×1440 | `4cfc382b086c9ab43a602f8f1c8393ec14feb50cd636138963f19666f69b143c` |

These renders came from the DEBUG-only render harness before `App.main()`, with app activation prohibited and a temporary Settings configuration. They did not open the canonical database, socket, collector, installed app, or production GUI. Accepted visual inspection covered the sidebar, Backups switch placement, capped wide content, compact rows, and save receipts. Installed visual QA was not performed.
