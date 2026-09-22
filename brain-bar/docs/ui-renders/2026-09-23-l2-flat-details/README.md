# L2 Details row render comparison

Base: `origin/main` at `f5994eb5` (#919 merged). Before is the base worktree; after is L2 on that base. Both were rendered with the DEBUG-only `BRAINBAR_RENDER_ONLY` harness, which exits before normal app startup and reads fixture data. These images establish source render layout only; installed visual QA was not performed.

## Pairs

| State | Before | After |
| --- | --- | --- |
| Compact, readable, Details expanded | [before](before-dashboard-cli-compact-readable-details-expanded.png) | [after](after-dashboard-cli-compact-readable-details-expanded.png) |
| Default, readable, Details expanded | [before](before-dashboard-cli-default-readable-details-expanded.png) | [after](after-dashboard-cli-default-readable-details-expanded.png) |
| Wide, readable, Details expanded | [before](before-dashboard-cli-wide-readable-details-expanded.png) | [after](after-dashboard-cli-wide-readable-details-expanded.png) |
| Default, unreadable, Details expanded | [before](before-dashboard-cli-default-unreadable-details-expanded.png) | [after](after-dashboard-cli-default-unreadable-details-expanded.png) |
| Default, stale, Details collapsed | [before](before-dashboard-cli-default-stale.png) | [after](after-dashboard-cli-default-stale.png) |

The existing Activity and Runtime sections were already flat definition lists on the base. In the unreadable default render, the old 24-point row ellipsized the `Agent writes` error. The after render wraps it within the same unboxed list. The stale pair is byte-identical. Disclosure state and the remaining facts are unchanged.
