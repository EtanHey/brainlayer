# Drive retention Settings proof

Before source: ee75fad98c621ad743b279e78ecc1acd9a8396ed (origin/main source bytes in the assigned worktree).
After source: e84f4b2dd3e93268942ca64890b65ff391f10ef9.

All inputs are invented. Real New Settings and Classic Settings' Google Drive section,
and the shared SwiftUI sheet, rendered off screen with NDRenderHarness. No application,
window, menu, production log, database or BrainBar socket was opened.
Classic section captures carry the actual Settings background and foreground colors.
New Settings includes its content and title-bar material; transparent canvas margins are
outside that content. The native system sheet animation and pointer interaction were
not exercised under this Mac's no-window restriction.

Daily/monthly counts are display values. Cleanup is OFF and DISABLED. Confirmation is
an unreachable component capture, not an enabled production state. No Drive writes exist.
Absent, stale, unmeasurable and null evidence never invents zero counts.

Baseline regression: 4 tests; 3 failing cases / 5 assertions, with the absent-section control passing.

## Pictures

- [after-classic-settings-measured-dark.png](after-classic-settings-measured-dark.png)
- [after-classic-settings-measured-light.png](after-classic-settings-measured-light.png)
- [after-classic-settings-missing-dark.png](after-classic-settings-missing-dark.png)
- [after-classic-settings-missing-light.png](after-classic-settings-missing-light.png)
- [after-confirmation-unreachable-dark.png](after-confirmation-unreachable-dark.png)
- [after-new-settings-measured-dark.png](after-new-settings-measured-dark.png)
- [after-new-settings-measured-light.png](after-new-settings-measured-light.png)
- [after-new-settings-missing-dark.png](after-new-settings-missing-dark.png)
- [after-new-settings-missing-light.png](after-new-settings-missing-light.png)
- [after-sheet-measured-dark.png](after-sheet-measured-dark.png)
- [after-sheet-measured-light.png](after-sheet-measured-light.png)
- [after-sheet-missing-dark.png](after-sheet-missing-dark.png)
- [after-sheet-missing-light.png](after-sheet-missing-light.png)
- [after-sheet-not_run-dark.png](after-sheet-not_run-dark.png)
- [after-sheet-stale-dark.png](after-sheet-stale-dark.png)
- [after-sheet-unmeasurable-dark.png](after-sheet-unmeasurable-dark.png)
- [before-classic-settings-missing-dark.png](before-classic-settings-missing-dark.png)
- [before-classic-settings-missing-light.png](before-classic-settings-missing-light.png)
- [before-new-settings-missing-dark.png](before-new-settings-missing-dark.png)
- [before-new-settings-missing-light.png](before-new-settings-missing-light.png)
