# L0 Details retention render evidence

These 1920×2336 PNGs were captured by the DEBUG-only `BrainBarL0StateTests` offscreen `NSWindow` harness. No installed BrainBar or production database was used. The fixture starts with Vector 71%, FTS5 83%, and Trigram 96%, then closes and reopens the outer Details disclosure. The reopened frames were captured 0.12 seconds after opening.

| Frame | Source state | What the pixels show |
| --- | --- | --- |
| [Retention off, early reopen](retention-off-reopened-early.png) | Reviewer's temporary RED change (`retainsContentWhenCollapsed: false`) | Each bar is partway through a new fill. |
| [Retention on, early reopen](retention-on-reopened-early.png) | Committed candidate | All three bars remain filled. |
| [Retention on, collapsed](retention-on-collapsed.png) | Committed candidate | Only the compact Details row remains visible. |

The before frame was produced from a throwaway review worktree with one temporary flag change, then removed. The after frames came from exact reviewed source commit `3d5be440888505688ee35c387038fbe4e1508b42`. The reviewer inspected both at original resolution and recorded the full RED/GREEN logs in the handoff. The test's fill-start event assertions are the temporal proof: RED 9 events across two cycles; GREEN 3. All runs kept the coverage provider at one call.

Scope: the outer Details toggle only. The inner Signal coverage toggle still remounts the bars. A real offscreen window starts each bar fill twice on first mount; this slice only prevents extra starts on Details reopen. Tab reachability of hidden controls was **not verified**: the offscreen test window never became key, so installed-app Tab acceptance remains open. The baseline `computing…` reset cause is unknown.
