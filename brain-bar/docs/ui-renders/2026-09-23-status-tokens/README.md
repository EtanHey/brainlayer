# BrainBar status-token renders

Visual comparison for the L1 semantic status palette and minimum type-size changes.

- **Before:** `origin/main` at `580747e67fb2d64bd7059c62ee627e95d810e188` (#930).
- **After:** L1 implementation snapshot `6fa303ebd177873fe01b099cb8becdddb798e8ac`.
- **Widths:** compact 760 pt, default 960 pt, wide 1280 pt.
- **States:** readable, stale, Details expanded, plus a dedicated Vector 100% coverage frame.
- Renders use the `BRAINBAR_RENDER_ONLY` debug harness. It creates offscreen PNGs and exits before normal app startup. No installed BrainBar or live database was used.

The ordinary before/after readable and stale frames use the same dashboard fixture values. The Vector 100% before frame uses a temporary debug-fixture-only override: `vectorIndexedChunkCount` is set to the fixture's `signalEligibleChunkCount` (297,412), and the debug renderer opens Signal coverage. This makes its data match the after `vector-at-100` scenario without changing product code on the baseline. The after frame uses the committed `vectorAt100Stats` scenario.

## Readable layout

| Width | Before | After |
| --- | --- | --- |
| Compact | [PNG](before-main-580747e6/compact-readable.png) | [PNG](after-l1/compact-readable.png) |
| Default | [PNG](before-main-580747e6/default-readable.png) | [PNG](after-l1/default-readable.png) |
| Wide | [PNG](before-main-580747e6/wide-readable.png) | [PNG](after-l1/wide-readable.png) |

## Stale status

| Width | Before | After |
| --- | --- | --- |
| Compact | [PNG](before-main-580747e6/compact-stale.png) | [PNG](after-l1/compact-stale.png) |
| Default | [PNG](before-main-580747e6/default-stale.png) | [PNG](after-l1/default-stale.png) |
| Wide | [PNG](before-main-580747e6/wide-stale.png) | [PNG](after-l1/wide-stale.png) |

## Details expanded

| Width | Before | After |
| --- | --- | --- |
| Compact | [PNG](before-main-580747e6/compact-readable-details-expanded.png) | [PNG](after-l1/compact-readable-details-expanded.png) |
| Default | [PNG](before-main-580747e6/default-readable-details-expanded.png) | [PNG](after-l1/default-readable-details-expanded.png) |
| Wide | [PNG](before-main-580747e6/wide-readable-details-expanded.png) | [PNG](after-l1/wide-readable-details-expanded.png) |

## Vector at 100%

| Before | After |
| --- | --- |
| [PNG](before-main-580747e6/vector-at-100-default-details-expanded.png) | [PNG](after-l1/vector-at-100-default-details-expanded.png) |

The status tokens use Apple system green, yellow, and red (`#32D74B`, `#FFD60A`, `#FF453A`); unknown remains muted gray. The Vector series remains amber. The before/after Vector pair shows the semantic green completion marker beside the unchanged amber series color. The test suite enforces CIE76 ΔE ≥ 25 between status tokens and all accent, signal, and series colors.
