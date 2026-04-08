---
name: memory
description: "Use BrainLayer memory deliberately: search before assumptions, store decisions after they are made, and recall current context when resuming work."
---

# BrainLayer Memory

Use BrainLayer when session context is incomplete or when the task depends on prior decisions.

- Run `brain_search` before answering architecture, project-history, preference, or "what did we decide" questions.
- Run `brain_recall` when you need the current working context, recent session state, or session-linked summaries.
- Run `brain_store` after decisions, corrections, failures, or milestones so the next Claude session can recover the why, not just the code diff.

Before answering from memory, verify with `brain_search` instead of assuming.
