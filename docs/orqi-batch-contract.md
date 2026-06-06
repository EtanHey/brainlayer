# ORQI Batch Contract

This contract covers KG flag-review batches consumed by ORQI/voice review
surfaces and produced by BrainLayer batch tooling. The canonical in-repo
location for batch and decisions artifacts is `eval_results/`.

## Batch Shape

Batches are JSON objects keyed by queue/category name:

```json
{
  "<queue-name>": [
    {
      "stem": "<stable item label>",
      "size": 2,
      "item_kind": "question",
      "members": [
        {"id": "q1", "name": "Question text", "type": "question", "chunks": 0}
      ]
    }
  ]
}
```

`item_kind` is required on newly emitted items. Older batches may omit it;
readers may derive it from `members` for backward compatibility.

## Item Kinds

`item_kind` is a string and is intentionally extensible.

- `question`: any member has `type: "question"`.
- `cluster`: no member has `type: "question"`.

Question items are note-only decisions. They must not expose member-mapping or
merge/keep/split cluster controls. Cluster items use the full decision menu.

## Member Types

Known member types in the 2026-06-06 Etan session batch:

| type | count |
| --- | ---: |
| company | 3 |
| concept | 24 |
| context | 34 |
| person | 2 |
| project | 19 |
| question | 6 |
| technology | 12 |
| tool | 14 |

`context` members are synthetic review context. They are useful for UI
presentation but must not be treated as real KG entities by cleanup appliers.

## Decisions Schema

Decision files use top-level schema id:

```json
{
  "schema": "kg-flag-decisions-v1"
}
```

Every writer must stamp `"schema": "kg-flag-decisions-v1"` when creating or
updating a decisions file that lacks the field. A file declaring another schema
must fail validation instead of being silently rewritten.

Versioning rule: breaking changes require a new schema id, starting with
`kg-flag-decisions-v2`, plus a migration note explaining how `v1` files are
converted or intentionally rejected.
