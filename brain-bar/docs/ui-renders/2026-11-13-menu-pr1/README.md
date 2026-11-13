# Contact sheets: native menu vs the mocks (menu PR 1, head c28f77d7)

> **Not a screenshot of an open menu.** Assembled off-screen from the real hosted rows and the real menu items at the heights `NSMenu.size` measures; the material is flat.
> A real open-menu capture on the M1 will add the system material, the real selection highlight and the real Option alternate swap.

`contact-sheet-light.png` and `contact-sheet-dark.png`: left column is the native menu (BrainBar, off-screen, 2x), right column
the mock render of the same state (A2 for the calm form: S1, S2-today; A1 for the needs form: S3, S5, S12), both at 2x, no
scaling. Made by `make_contact_sheets.py` (kept with the full capture set in the plan folder, `phase-8-native-build/captures/menu-pr1/`).

## Where native deliberately differs from the mocks, and why

Layout and type
1. **One width for both forms: 340 pt** (A1 340, A2 360). The menu must not change width when the form switches ("nothing
   shifts on refresh").
2. **One headline row in both forms: A2's**: 26 pt disc, 15 pt semibold. A1 drew its needs headline at 13 pt bold. Equal things
   equal: the headline is the same row in both forms; only its disc and words change.
3. **One text column in both forms: 52 pt (A2's)**, the row mark centred in the 26 pt gutter. A1's needs text sat at 48 pt. In
   S3 the 4 pt difference wraps "…is 21 / hours old." where A1 wrapped "…is 21 hours old." (A1's own sentence was also shorter,
   see 12).
4. **Standard items are the system's**: titles at the 16 pt inset and the 24 pt row and 11 pt separator measured from
   `NSMenu.size` on macOS 27. The mocks' values matched; nothing changed there.
5. **One bottom block in both forms (ruling R4, A2's)**: no separator between Settings… and Quit BrainBar, and the key
   equivalents shown (⌘, and ⌘Q). The A1 renders have a separator between them and no key equivalents.

Colour and marks
6. **Secondary text is `secondaryLabelColor`**, the system's, which is lighter than the mocks' fixed #56565B / #5E5E63. Under
   Increase Contrast it becomes the label colour.
7. **Marks are SF Symbols in system colours** (`checkmark`, `arrow.triangle.2.circlepath`, `ellipsis`, `exclamationmark`,
   `exclamationmark.circle.fill`) on drawn discs. The mocks used hand-drawn SVGs, so A2's busy arrows look slightly different.
8. **The button fill is the system accent colour** (`Color.accentColor`), blue by default, not a fixed #0062D6. It follows
   the owner's accent setting, as system prominent buttons do. **Flag for the lead:** he named "the blue button". If it must
   stay blue under any accent, this is a one-word change to `.systemBlue`.

Controls and motion
9. **The button is 26 pt tall** (A2's height; A1's was 24) with a 13 pt medium label, so its hit target is above the 24 pt
   standard row.
10. **Pressed state: a darker fill and a 0.97 scale, applied at once with no transition.** The mocks animate it over
    120–140 ms, but the brief says nothing animates except a bar and a spinner. Under Reduce Motion the scale is dropped and
    only the darker fill stays.
11. **Keyboard focus is the system focus ring** (`keyboardFocusIndicatorColor`, 3 pt, 1 pt outside, concentric with the
    button: radius 13 + 1 = 14 inner). It appears only when the arrow keys highlight the row, never on hover. The mock drew a
    3 px rgba glow. (Pictures: `menu-S3-focus-*.png`.)
12. **Words come from the presentation value, not the mocks.** S3 reads "Your newest backup in Google Drive is 21 hours old."
    (ruling R15) where A1 drew "Your newest backup is 21 hours old.". Apostrophes are typographic (’) everywhere (R2); A2's S7
    render had ASCII "Can't".
13. **Progress bar: the native linear `ProgressView`** (system height and tint) instead of A2's 6 pt custom bar (S2-after-seam).
14. **Spinner: the native mini `ProgressView`** in the gutter (S10). Under Reduce Motion it is a static arrows glyph
    (`menu-S10-reduce-motion-*.png`).

Harness, not design
15. **The material is flat in these pictures** (#ECECEE / #1F1F21, radius 13, the mocks' values). An off-screen render cannot
    draw the vibrant menu material; the real menu gets the system's material and corners.
16. **Text renders slightly heavier off-screen** than in a real menu window or in the mocks' browser renders. This comes from
    the bitmap context, not from the font weights set in code.

## Open taste point for the reviewer
17. **S3's Drive fact line now wraps to three lines** ("…Your newest backup in Google Drive is 21 / hours old."), because ruling
    R15 lengthened the sentence and the text column is 52 pt at a 340 pt width. Words and width are deliberately unchanged
    here; judge it in `menu-S3-light.png` / `menu-S3-dark.png` (also `menu-C32-*.png`, same line) and on the contact sheets.

## Owner note N1 (an uploading backup)
S2-today shows the blue busy disc with arrows, not a green light, and S2-after-seam adds the progress bar. N1 also asks for "a
spinner or pulse". The value does not set `spinner` on S2's headline, and the brief allows no other animation, so the busy disc
stays static. If the lead wants the disc to spin while uploading, the presentation value should carry that (the presenter sets
`spinner`), not the view.
