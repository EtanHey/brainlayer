# BrainBar Dashboard Flow Collab

## Goal
Finish the second-pass BrainBar dashboard refactor without mixing `live agent CLIs` into `indexed writes` or letting the queue/runtime cards dominate the layout.

## Agents
- `main-codex`
  Ownership:
  - `brain-bar/Sources/BrainBar/Dashboard/AgentActivityMonitor.swift`
  - `brain-bar/Sources/BrainBar/Dashboard/StatsCollector.swift`
  - `src/brainlayer/pipeline/enrichment_tiers.py`
  - `brain-bar/Tests/BrainBarTests/AgentActivityMonitorTests.swift`
  - `tests/test_enrichment_tiers_agent_sources.py`
- `clean-claude-ui`
  Ownership:
  - `brain-bar/Sources/BrainBar/BrainBarWindowRootView.swift`

## Constraints
- Do not edit files outside your ownership.
- Do not revert or restage another agent's work.
- `clean-claude-ui` must `git add` only its owned file before commit.
- `clean-claude-ui` may commit and push its owned UI change after tests pass.

## Task Board
| Task | Owner | Status |
|------|-------|--------|
| Add process-based agent family monitor | main-codex | done |
| Extend enrichment tiering to non-Claude agent sources | main-codex | done |
| Refactor dashboard layout to add agent strip and compact queue rail | clean-claude-ui | in_progress |
| Collapse/demote runtime details behind disclosure | clean-claude-ui | in_progress |

## Gates
- `main-codex`: `swift test --filter AgentActivityMonitorTests`, `swift test --filter BrainBarUXLogicTests`, `pytest tests/test_enrichment_tiers.py tests/test_enrichment_tiers_agent_sources.py -q`
- `clean-claude-ui`: `swift test --filter BrainBarUXLogicTests` before commit

## Notes
- Existing `brainlayerClaude -s` pane is bound to the dirty main repo checkout and stays read-only for critique only.
- The clean-worktree Claude worker must run from `/Users/etanheyman/.config/superpowers/worktrees/brainlayer/fix-brainbar-dashboard-flow`.
