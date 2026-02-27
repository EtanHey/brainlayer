# Knowledge Graph Specification

BrainLayer's knowledge graph (KG) stores structured entities and relations extracted from indexed conversations. This document specifies the entity types, relation types, and time-decay scoring model.

## Entity Types

| Type | Description |
|------|-------------|
| `person` | People mentioned in conversations |
| `constraint` | Scheduling or resource constraints |
| `preference` | User preferences and choices |
| `life_event` | Date-bounded life events |
| `meeting` | Meetings and appointments |
| `location` | Physical or virtual locations |
| `organization` | Companies, teams, groups |

## Relation Types

| Relation | Description |
|----------|-------------|
| `has_constraint` | Entity has a constraint |
| `has_preference` | Entity has a preference |
| `blocked_during` | Entity unavailable during period |
| `attended` | Person attended a meeting/event |
| `organized_by` | Meeting/event organized by person |
| `knows` | Person knows another person |
| `works_at` | Person works at organization |
| `supersedes` | Newer entity replaces older one |
| `held_at` | Meeting/event held at location |

## Time-Decay Scoring

Entity relevance decays over time using an exponential model:

```
score = confidence * importance * exp(-lambda * age_days)
```

| Entity Type | Lambda | Half-Life |
|-------------|--------|-----------|
| `constraint` | 0.0019 | ~365 days |
| `preference` | 0.0077 | ~90 days |
| `life_event` | 0 | No decay (date-bounded) |
| `casual` | 0.0231 | ~30 days |
| `meeting` | 0.0046 | ~150 days |

- **confidence** (0-1): How certain we are about this fact
- **importance** (0-1): How important this fact is
- **age_days**: Days since entity creation
- Default decay rate (when type is unknown): preference rate (0.0077)

## Compatibility

The KG spec is shared with Convex (`kgSpec.ts` in 6PM) to ensure consistent entity modeling across the ecosystem.

## Source

Defined in `src/brainlayer/kg/__init__.py`.
