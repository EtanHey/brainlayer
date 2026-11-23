# Compact Pipeline and station clearance

The Pipeline card shrinks from 353 to 304 pt, the calm window from 791 to 742 pt, and the Drive-fault window from 841 to 792 pt. Every bubble keeps 4 pt from other station icons and words.

Invented fixtures rendered offscreen through NDRenderHarness at 2x in dark and light appearances. No installed application, production database, production socket, real logs or personal screenshot was used.

Before source: `ce9086e9bcf7cee722be6dbf6fb7bd1ec240f516`. After source: `62976c7f8750067b1b17de5d219d089351ade17f`.

`before/` retains the original ten pictures. `after/` refreshes those ten and adds Agents, Save queue, Read, Search calm and Search fault bubbles in both themes. `wide/` refreshes Local copy, Drive fault and Read lag in both themes. All 36 opaque PNGs were opened for inspection; no transparent margin needed cropping.

| Measurement (pt) | Original main | After |
| --- | ---: | ---: |
| Calm Overview window | 791 | 742 |
| Drive-fault Overview window | 841 | 792 |
| Pipeline chart | 345 | 296 |
| Pipeline card | 353 | 304 |
| Job horizontal distance | 186 | 337 |
| Lowest job words to return caption | 81 | 32 |
| Station row y | 44 | 44 |
| Activity gap token | 0 | 0 |

Memory remains below between the spread jobs. Every bubble clears other station icons and drawn words by 4 pt; bodies also clear the Kept-safe one-point outline by 4 pt, including half its stroke width. Agents also clears the return caption. The Weekly port into Memory is 16 pt inward from the Memory-to-Search port. Both jobs move 4 pt right to clear the Save queue notch, preserving their 337 pt separation. Search uses a 5 pt compact action gap instead of 8 pt. The three-column wide layout remains; the wide vertical inset is 5.5 pt instead of 8 pt so the longest 59 pt body fits above the frame without adding height. Fonts, 18 pt fact rows, normal stacked bodies, text and action labels remain unchanged.

Five round-two guards failed at bfb05d57 with 608 assertions, covering measured station margins, screen-fit margins, oversized margins, frame clearance and separate ports. Existing Agents-caption and trimmed-link guards also ran red at 916257ce (23 assertions) before correction. The final verification uses 57 explicit safe method selectors. Final source: 57 focused Swift tests, 17 Python checks including R31, 13/13 private producer-to-queue-to-production-Swift-reader cases pass. Both products built.

The local Claude review at bfb05d57 found one blocking margin defect; these are the round-two corrections. Final lead verification, CI and installed acceptance remain separate and pending.
