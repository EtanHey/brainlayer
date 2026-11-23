# Compact Pipeline: offscreen before and after

Invented design fixtures rendered through NDRenderHarness at 2x, in dark and light appearances. Static offscreen proof only; native interaction and installed acceptance remain separate. No installed application, production database, production socket, or personal screenshot was used.

Before source: `ce9086e9bcf7cee722be6dbf6fb7bd1ec240f516`. After source: `bfb05d571bf2b8f31c0db81f773b38356f8a2a03`.

`before/` and `after/` contain the same ten states: whole calm Overview, Memory/Nightly/Weekly hovers, and whole Drive-fault Overview, each in both themes. `wide/` adds six after-only views of Local copy, Drive fault, and Read lag to show the widened column layout. All 26 images were opened for inspection and are fully opaque; no transparent margins needed cropping.

| Measurement (pt) | Before | After |
| --- | ---: | ---: |
| Whole calm Overview | 791 | 742 |
| Whole Drive-fault Overview | 841 | 792 |
| Pipeline chart | 345 | 296 |
| Pipeline card, including 4 pt padding on each side | 353 | 304 |
| Lowest job words to return caption | 81 | 32 |
| Job station horizontal distance | 186 | 337 |
| Upper station row y | 44 | 44 |
| Activity-to-Pipeline gap token | 0 | 0 |

The card and both measured windows shrink by 49 pt. Memory opens below at the measured gap between the job words, with 4 pt clearance. The widened Local copy, Drive and Read bodies arrange the same title, explanation, facts and action in columns. Normal-width bodies keep the existing stack and 14/10 pt padding. The wide variant uses 14/8 pt padding; fonts and 18 pt rows stay the same. Kept-safe dimensions and text spacing, diagram side insets, purpose lines, notch targets and connector endpoint gaps remain checked.

The original three behavior guards failed on the before source (351 assertions). Three further guards for the original positions/gap, smaller windows and below-Memory pocket failed on the first PR head c419e273 (93 assertions). The final source passes 53 focused Swift tests, 17 Python checks including R31, and 13 private heartbeat cases. Both products built. Claude review, CI and installed proof remain pending; local CodeRabbit produced no completed verdict within its bounded runs.
