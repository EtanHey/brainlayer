# BrainBar L3a card-shape comparison

The 36 offscreen frames compare exact #932 base `8fe696e50a944cb9f27bb2ebe49fd5d5924014ff` with L3a source `4aef233d51bc6370c9261135c659f17474c4aeb9`. The 18 after frames were re-rendered at review-fix head `b6076db2` and remained byte-identical. Both captures use the same injected L3 fixture and debug render harness. No installed app or live database was used; the window never became key, so keyboard focus and VoiceOver remain unverified.

| State | 760 pt | 960 pt | 1280 pt |
| --- | --- | --- | --- |
| Live | [before](before-l1/compact-live.png) / [after](after-l3a/compact-live.png) | [before](before-l1/default-live.png) / [after](after-l3a/default-live.png) | [before](before-l1/wide-live.png) / [after](after-l3a/wide-live.png) |
| Loading cards | [before](before-l1/compact-loadingCards.png) / [after](after-l3a/compact-loadingCards.png) | [before](before-l1/default-loadingCards.png) / [after](after-l3a/default-loadingCards.png) | [before](before-l1/wide-loadingCards.png) / [after](after-l3a/wide-loadingCards.png) |
| Stale | [before](before-l1/compact-stale.png) / [after](after-l3a/compact-stale.png) | [before](before-l1/default-stale.png) / [after](after-l3a/default-stale.png) | [before](before-l1/wide-stale.png) / [after](after-l3a/wide-stale.png) |
| Unavailable | [before](before-l1/compact-unavailable.png) / [after](after-l3a/compact-unavailable.png) | [before](before-l1/default-unavailable.png) / [after](after-l3a/default-unavailable.png) | [before](before-l1/wide-unavailable.png) / [after](after-l3a/wide-unavailable.png) |
| Empty | [before](before-l1/compact-empty.png) / [after](after-l3a/compact-empty.png) | [before](before-l1/default-empty.png) / [after](after-l3a/default-empty.png) | [before](before-l1/wide-empty.png) / [after](after-l3a/wide-empty.png) |
| Error | [before](before-l1/compact-error.png) / [after](after-l3a/compact-error.png) | [before](before-l1/default-error.png) / [after](after-l3a/default-error.png) | [before](before-l1/wide-error.png) / [after](after-l3a/wide-error.png) |

The unavailable fixture carries a long `open("/Users/fixture/.local/share/brainlayer/fixture.db", 14)` coverage error. The error and failed search/store receipts, including their age, fit at compact width in the after frames. The source tests measure fixed card frames across all six states and intrinsic text fit for the coverage reason and failed store receipt. Activity and Runtime definition-list height must be exempted from that fixed-frame assertion when #927's wrapping Details facts are integrated later.
