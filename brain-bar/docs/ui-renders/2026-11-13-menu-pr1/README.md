# Contact sheets: native menu vs the mocks (menu PR 1, after review round 2, head 78be766a)

> **Not a screenshot of an open menu.** Assembled off-screen from the real hosted rows and the real menu items at the heights `NSMenu.size` measures; the material is flat.
> It cannot show motion (the turning busy arrows, the spinner), a disabled standard item's style, or AppKit's own title sizing and truncation beside a key equivalent.
> A real open-menu capture on the M1 will add the system material, the real selection highlight, the real Option alternate swap, those standard-item details and the motion.

`contact-sheet-light.png` and `contact-sheet-dark.png`: left column is the native menu (BrainBar, off-screen, 2x), right column
the mock render of the same state (A2 for the calm form: S1, S2-today; A1 for the needs form: S3, S5, S12), both at 2x, no
scaling. Made by `make_contact_sheets.py`, kept with the full capture set in the plan folder (`phase-8-native-build/captures/menu-pr1/`).

## Where native deliberately differs from the mocks, and why

Layout and type
1. **One width for both forms: 340 pt** (A1 340, A2 360). The menu must not change width when the form switches.
2. **One headline row in both forms: A2's** (26 pt disc, 15 pt semibold). A1 drew its needs headline at 13 pt bold.
3. **One text column in both forms: 52 pt (A2's)**, with each row's mark centred in the 26 pt gutter. A1's needs text sat at 48 pt.
4. **Standard items use the system's metrics**: titles at the 16 pt inset, a 24 pt row and an 11 pt separator, measured from
   `NSMenu.size` on macOS 27. The mocks match.
5. **One bottom block in both forms (ruling R4, A2's)**: no separator between Settings… and Quit, and key equivalents shown.
6. **A number never ends a line apart from its unit, and a month stays with its day (R28a).** This is done in the view, for
   line breaking only, so the words themselves do not change. S3 now breaks "…in Google Drive is / 21 hours old." and
   S2-after-seam breaks at the sentence: "To Google Drive. Started at 11:28. / 40% done, about 12:48."
7. **Digits are fixed-width (R28d)**, so a changing time or percentage cannot move a line break.

Colour and marks
8. **Secondary text is `secondaryLabelColor`**, the system's. It is lighter than the mocks' fixed greys. Under Increase
   Contrast it becomes the label colour.
9. **Marks are SF Symbols in system colours** on drawn discs.
10. **Every status line has a mark (R28b).** It is the value's mark if the line has one. Otherwise it comes from the line's
    state: a bare ellipsis, the size of the checkmark, for "can't tell" (S10's "Can’t confirm agents can search right now."),
    and the spinner for "behind". A1's mock left that gutter empty. A headline's detail line is the one exception, because the
    headline's disc already carries its state.
11. **The busy arrows turn (R28c)**: one even, linear turn every 2 s while shown (S2-today, S2-after-seam, S7), and still under
    Reduce Motion. The harness draws them at a fixed phase (upright), and no picture can show the motion.
12. **The button fill is the system accent colour**, blue by default. **Flag:** on 13 Nov the owner named the blue button as something he wanted kept. If it must stay blue
    under any accent setting, pinning it to `.systemBlue` is a one-word change.

Controls
13. **Button:** regular-weight 13 pt label, as the system draws one. It is at least 26 pt tall, and a long label wraps rather
    than truncates (the button grows).
14. **Pressed: the button only darkens, with no shrink**, as a native button does. Nothing animates except the bar, the spinner
    and the busy arrows.
15. **Keyboard focus uses the system focus ring**, concentric with the button. It appears only when the arrow keys highlight
    the row, never on hover (`menu-S3-focus-*.png`).
16. **"Everything else is fine." sits 10 pt clear of a Needs-you row above it**, so it does not read as part of that row. This
    is space, not a divider; the value decides dividers.
17. **The words come from the value**, not the mocks (R15, R2).
18. **The progress bar and the spinner are the system's** (`ProgressView`). Under Reduce Motion the spinner is a static arrows
    glyph.

Harness, not design
19. **Flat material, slightly heavier text**, a standard item's title and key drawn by the harness, and no motion. See the
    caption above.

## Left for the function lane (not accepted in this PR)
- "Google Drive" still splits across lines inside S32's bold title ("…A backup to Google / Drive started…"). Keeping a name
  together would need a word in a view, which ruling R28 does not allow.

## What only the real open menu on the M1 can show
System material and secondary-text contrast on it; the real selection highlight, and whether an arrow-key highlight of a
Needs-you row should also show the row highlight; whether Return on that row and a click on its button fire the action and
close the menu; whether the spinner and the turning arrows run inside the menu's event-tracking loop; text weight; the accent
in dark mode and under his own accent setting; the real Option swap; separator insets and corner radius; a disabled standard
item's style; AppKit's title sizing and truncation beside a key equivalent; and VoiceOver: each hosted row read as one element
in order, the button as its own element, and the marks silent.
