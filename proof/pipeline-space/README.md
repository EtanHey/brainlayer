# Pipeline spacing: offscreen before and after

Invented design fixtures rendered through NDRenderHarness at 2x in dark and light appearances. No installed application, owner screenshot, production database, or production socket was used. These pictures show static offscreen layout; native interaction and installed acceptance remain separate checks.

- Before source: `ce9086e9bcf7cee722be6dbf6fb7bd1ec240f516` (origin/main at lane start).
- After source: `c419e2736ced6cdebb9387fe63050887aa302e31`. The integrated main commit `80059ee9c` has the same BrainBar tree as the before source.
- Each directory has a whole Overview at rest, Memory/Nightly/Weekly hover, and a whole faulty Overview with the invented Drive signed-out fixture.
- All 20 PNGs were opened for visual inspection. Alpha is fully opaque, so there are no transparent margins to crop.

| Measurement (pt) | Before | After |
| --- | ---: | ---: |
| Pipeline drawing height | 345 | 325 |
| Pipeline card height including 4 pt vertical padding | 353 | 333 |
| Lowest job words to return caption | 81 | 32 |
| Job station horizontal distance | 186 | 337 |
| Whole calm Overview height | 791 | 803 |
| Activity-to-Pipeline gap | 0 | 32 |

The Pipeline card shrinks by 20 pt. Restoring the existing L54 32 pt gap between Activity and Pipeline makes the whole window 12 pt taller; the window follows its content. Job purpose lines, Kept-safe dimensions/text padding, diagram side insets, and the agents column remain covered by focused geometry checks.

The three new behavior checks failed on the before source (351 assertions across three failing tests) and pass on the after source. The after source passes 49 focused Swift tests, 17 Python checks including R31, and 13 private heartbeat cases. Both BrainBar and BrainBarDaemon built. Lead-routed Claude review, CI, native interaction, and installed proof remain pending.
