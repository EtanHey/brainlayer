# Drive retention Settings, round 2

After source: 44f69350a4e8bb11d33db4ffedcf564fd3db7ad9. Previous proof: 84b279cb2697221a9b315c7903590c4699cfb8d1.
Before: preserved round-1 baseline captures at ee75fad98, cropped to the rendered bounds.
After: real SwiftUI New Settings, Classic Drive row and shared sheet, drawn offscreen by
NDRenderHarness using invented fixtures only. Transparent capture margins are cropped.
New captures use their intrinsic fitting height and include the fixture's settings-backups line.
Classic uses its existing dark palette in either system appearance, so it is shown once.
The not_run payload has error=null. The measured fixture separates eight verified rule-kept
copies and two unverified copies on Mac A; Mac B has ten verified copies and no unverified copies.
Those cohorts are never added to on_drive, kept, or would_remove.

Cleanup is off and disabled. The confirmation is an unreachable component capture.
No app, window, menu, production data, log, socket, Drive writer or live interaction was used.
System sheet presentation, Done/Escape and installed-app acceptance remain release checks.
The first five round-2 regressions failed on 696316c1: 14 assertions, including a measured
108 pt increment for a three-line Mac. Updated checks include drawn copy and layout heights.

## Pictures

- [after-classic-settings-config-error-dark.png](after-classic-settings-config-error-dark.png)
- [after-classic-settings-measured-dark.png](after-classic-settings-measured-dark.png)
- [after-classic-settings-missing-dark.png](after-classic-settings-missing-dark.png)
- [after-confirmation-unreachable-dark.png](after-confirmation-unreachable-dark.png)
- [after-new-settings-config-error-dark.png](after-new-settings-config-error-dark.png)
- [after-new-settings-config-error-light.png](after-new-settings-config-error-light.png)
- [after-new-settings-measured-dark.png](after-new-settings-measured-dark.png)
- [after-new-settings-measured-light.png](after-new-settings-measured-light.png)
- [after-new-settings-missing-dark.png](after-new-settings-missing-dark.png)
- [after-new-settings-missing-light.png](after-new-settings-missing-light.png)
- [after-sheet-long-error-dark.png](after-sheet-long-error-dark.png)
- [after-sheet-measured-dark.png](after-sheet-measured-dark.png)
- [after-sheet-measured-light.png](after-sheet-measured-light.png)
- [after-sheet-missing-dark.png](after-sheet-missing-dark.png)
- [after-sheet-missing-light.png](after-sheet-missing-light.png)
- [after-sheet-not_run-dark.png](after-sheet-not_run-dark.png)
- [after-sheet-not_run-light.png](after-sheet-not_run-light.png)
- [after-sheet-null-summary-dark.png](after-sheet-null-summary-dark.png)
- [after-sheet-null-summary-light.png](after-sheet-null-summary-light.png)
- [after-sheet-stale-dark.png](after-sheet-stale-dark.png)
- [after-sheet-stale-light.png](after-sheet-stale-light.png)
- [after-sheet-three-line-dark.png](after-sheet-three-line-dark.png)
- [after-sheet-unmeasurable-dark.png](after-sheet-unmeasurable-dark.png)
- [after-sheet-unmeasurable-light.png](after-sheet-unmeasurable-light.png)
- [before-classic-settings-missing-dark.png](before-classic-settings-missing-dark.png)
- [before-new-settings-missing-dark.png](before-new-settings-missing-dark.png)
- [before-new-settings-missing-light.png](before-new-settings-missing-light.png)

## Lead follow-up at 6eecfda98

After source for every after-sheet-* picture: 6eecfda98 (round-2 review items R2-1, R2-2, R2-4 and the producer's pinned cohort). `after-sheet-measured-dark/light` re-rendered: kept lines first (Keeps, Monthly, Not verified), Would remove last, every count reads "N copies". `after-sheet-pinned-dark` is new: Mac A keeps 7 by rule, 2 not verified, 1 pinned, would remove 10 (7+2+1+10 = 20, the header). The other sheet pictures were re-rendered from the same head; the Settings and confirmation pictures are unchanged from 44f69350 (this change touches only the sheet). Cropped to the drawn bounds like the rest.
