# Enrichment (retired)

LLM chunk and session enrichment is retired. There are no supported enrichment
run, resume, provider, or scheduler activation instructions.
History: [CHANGELOG](https://github.com/EtanHey/brainlayer/blob/main/CHANGELOG.md).

Local indexing, search, digest, and knowledge graph operations remain available.
Existing chunk metadata and historical session analysis remain readable; retirement
does not delete stored rows or their tags, importance, summaries, and provenance.
Offline prompt emission/collection, graders, fixtures, and local checkpoint replay
remain available without a cloud model client.

## Historical chunk metadata

The fields below describe existing records, not an active producer:

| Field | Description | Example |
|-------|-------------|---------|
| `summary` | 2-4 dense sentences of key facts, decisions, names, numbers | "Debugging Telegram bot message drops under load" |
| `key_facts` | Verbatim specific values (PR numbers, dates, paths, versions) | "PR #727, grammy 1.32" |
| `tags` | Topic tags (comma-separated) | "telegram, debugging, performance" |
| `importance` | Relevance score 1-10 | 8 (architectural decision) vs 2 (directory listing) |
| `intent` | What was happening | `debugging`, `designing`, `implementing`, `configuring`, `deciding`, `reviewing` |
| `primary_symbols` | Key code entities | "TelegramBot, handleMessage, grammy" |
| `resolved_queries` | HyDE-style question, keyword-dense query, and hypothetical answer snippet | "How does the Telegram bot handle rate limiting?" |
| `epistemic_level` | How proven is this | `hypothesis`, `substantiated`, `validated` |
| `version_scope` | System state context | "grammy 1.32, Node 22" |
| `debt_impact` | Technical debt signal | `introduction`, `resolution`, `none` |
| `external_deps` | Libraries/APIs mentioned | "grammy, Supabase, Railway" |
| `entities` | Name/type/subtype/relation records fed into the knowledge graph | `{"name": "grammy", "type": "technology"}` |
| `sentiment_label` | Chunk-level sentiment | `frustration`, `confusion`, `positive`, `satisfaction`, `neutral` |
| `sentiment_score` | Sentiment magnitude | `-1.0` to `1.0` |
| `sentiment_signals` | Words/phrases that indicate the sentiment | "still broken, third time" |

## Installed credential compatibility

Keep `GOOGLE_API_KEY`, `BRAINLAYER_REQUIRE_GOOGLE_API_KEY`, and the env-run exit-78
gate until the release re-renders installed hotlane plists.
See [Configuration](configuration.md) for the retained 1Password-backed key guidance.
