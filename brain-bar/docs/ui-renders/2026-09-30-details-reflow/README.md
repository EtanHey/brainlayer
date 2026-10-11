# Details collapse reflow render proof (#964)

These renders were produced by the DEBUG render harness (`BRAINBAR_RENDER_ONLY`) with fixture data, at a fixed 640 pt height. They are pixels from a source build, not proof from the installed app or live data. The harness runs before the app starts, so no DB, socket, collector or status UI exists during the run.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

## The new probe: `dashboard-fixed-<width>-collapsed-in-place`

The probe follows the #964 repro:
1. Expand Details (with Signal coverage open).
2. Scroll the dashboard to the bottom.
3. Collapse Details.
4. Capture exactly where the viewport settles, with **no manual re-scroll**.

The harness then requires that capture to be byte-identical to `dashboard-fixed-<width>-collapsed` (the collapsed dashboard scrolled to its bottom). If it isn't, the run fails with `collapse-in-place probe … blank tail or lost reflow`.

The existing #934 `after-collapse` render scrolls to the top and back before it captures. That re-scroll released the scroll floor, which is why it never showed this bug.

| Width | 1. Expanded, scrolled to bottom | 2. After collapse, this fix | 3. After collapse, origin/main (before) |
| --- | --- | --- | --- |
| Compact, 760 × 640 pt | [PNG](compact-1-expanded-scrolled.png) | [PNG](compact-2-after-collapse-fix.png) | [PNG](compact-3-after-collapse-before.png) |
| Default, 960 × 640 pt | [PNG](default-1-expanded-scrolled.png) | [PNG](default-2-after-collapse-fix.png) | [PNG](default-3-after-collapse-before.png) |
| Wide, 1280 × 640 pt | [PNG](wide-1-expanded-scrolled.png) | [PNG](wide-2-after-collapse-fix.png) | [PNG](wide-3-after-collapse-before.png) |

## Byte comparisons (`cmp`) at all three widths

- **This fix:** collapse-in-place is byte-identical to collapsed. The harness check passes.
- **origin/main** (`af130e78` with only this harness probe applied): collapse-in-place **differs** from collapsed at every width, and the harness exits with `collapse-in-place probe at 760.0pt … blank tail`. The column 3 PNGs show the tail: the Details header sits in the upper part of the viewport, with empty background below it.
- **The #934 receipts still hold.** `after-collapse` is byte-identical to `collapsed` at each width.
- **The fix changes no static layout.** The collapsed and expanded renders are byte-identical between origin/main and this fix at each width. Only what happens after the collapse changes.

I inspected all nine PNGs, at 760, 960 and 1280 across the three columns.
