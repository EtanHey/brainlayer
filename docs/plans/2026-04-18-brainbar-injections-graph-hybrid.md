# BrainBar Injections + Graph Hybrid Design

## Decision

Ship a hybrid of Claude Design `Concept A: Retrieval Theater / Atlas` with two selective ideas from `Concept B: Signal Stream / Observatory`.

Use `A` as the primary visual language:
- `Injections` becomes a retrieval surface instead of a raw list.
- `Graph` becomes a stable atlas with a dominant inspector.

Steal only two things from `B`:
- grouped retrieval bursts so the feed shows rhythm instead of flat recency
- a graph altitude control so low-importance nodes can be faded out without losing focus

## Why

`Concept A` maps best onto the real BrainBar data model and the current codebase.

It works with:
- `InjectionEvent` data that already has timestamp, session, query, chunk IDs, and token count
- `KGViewModel` data that already has entity type, importance, relations, and chunk drilldown

It also avoids the two main failure modes in the other concepts:
- too much dashboard overlap (`Control Room`)
- too much metaphor or custom behavior for a first ship (`Investigation Desk`, `Heatmap × Starfield`, `Command Transcript / Codex`)

## Injections Target

### Information hierarchy

1. Top summary strip
   - `last hour` window label
   - query count
   - chunk count
   - token count
   - active session count

2. Activity ribbon
   - 60-minute spark ribbon
   - visible "now" marker
   - burst intensity visible at a glance

3. Burst feed
   - events grouped into session bursts
   - each burst has a compact header: time span, session, query count, token total
   - each event row keeps:
     - timestamp
     - query text
     - chunk count
     - chunk attribution ribbon

4. Right rail
   - active sessions
   - token pressure
   - signal notes

### Rules

- The page must still work when there is only one event.
- Filtering must work across session ID, query text, and chunk IDs.
- Empty space is not acceptable; the rail should collapse when data is sparse.
- The raw thread-opening affordance must remain.

## Graph Target

### Information hierarchy

1. Stable atlas canvas
   - stop presenting the graph as a black force-sim toy
   - treat the canvas as named regions by entity type
   - use a calmer background that works in both light and dark

2. Inspector-first drilldown
   - selected entity sidebar becomes the main reading surface
   - relations and linked chunks stay prominent
   - canvas supports the inspector instead of competing with it

3. Atlas controls
   - type legend / regions
   - altitude slider backed by node importance
   - optional minimap / overview overlay if it fits cleanly

### Rules

- Region naming should be deterministic from entity type.
- Altitude filtering should never hide the currently selected node.
- The graph must remain navigable if the sidebar is open.
- The visual system must stop relying on pure black backgrounds.

## Implementation Shape

### New presentation helpers

Add testable presentation helpers instead of embedding all behavior in SwiftUI bodies.

- `InjectionPresentation`
  - filters events
  - groups events into session bursts
  - computes top-strip stats
  - computes ribbon buckets

- `KGPresentation`
  - groups nodes into atlas regions
  - computes altitude-filtered visible nodes
  - preserves selected node visibility

### View changes

- `InjectionFeedView`
  - replace raw `List` layout with a custom scrollable composition
  - summary strip + ribbon + burst feed + right rail

- `KGCanvasView`
  - add calmer canvas framing
  - add top controls / legend / altitude
  - support atlas overlays

- `KGSidebarView`
  - upgrade hierarchy and chunk cards so it feels like the graph's primary inspector

## Out of Scope For This Pass

- fixing the blank-state runtime bug on `Injections` and `Graph`
- replacing the graph simulation with a full deterministic layout engine
- adding new backend data not already available in BrainBar
- changing dashboard again

## Verification Target

- Add failing tests first for burst grouping and altitude filtering.
- Run targeted BrainBar Swift tests.
- Relaunch the menu bar app and verify the redesigned tabs visually from the running build.
