# ABCDE Enrichment Results Summary

| Variant | label | n | ok | gen errors | schema pass % | banned % | mean importance | mean tags | mean key facts | mean entities | mean unsupported entities | mean resolved queries | mean completion tokens | total cost USD | error reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | production | 24 | 23 | 1 | 0.0 | 0.0 | 7.13 | 6.70 | 3.61 | 3.22 | 0.57 | 3.00 | 418.2 | 0.807056 | transport_error: HTTPSConnectionPool(host='api.x.ai', port=443): Read timed out. (read timeout=60) x1 |
| B | faceted-v2 | 24 | 23 | 1 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 92.0 | 0.390148 | transport_error: HTTPSConnectionPool(host='api.x.ai', port=443): Read timed out. (read timeout=60) x1 |
| C | density-max | 24 | 24 | 0 | 87.5 | 0.0 | 6.88 | 6.62 | 14.38 | 2.58 | 0.75 | 3.00 | 609.6 | 0.793276 | - |
| D | entity-first | 24 | 23 | 1 | 91.3 | 0.0 | 6.52 | 6.13 | 2.26 | 5.30 | 1.00 | 3.00 | 447.2 | 0.588029 | transport_error: HTTPSConnectionPool(host='api.x.ai', port=443): Read timed out. (read timeout=60) x1 |
| E | hyde-structure | 24 | 23 | 1 | 95.7 | 0.0 | 6.43 | 6.91 | 6.09 | 3.22 | 1.00 | 3.00 | 538.7 | 0.634110 | transport_error: HTTPSConnectionPool(host='api.x.ai', port=443): Read timed out. (read timeout=60) x1 |

TOTALS: 120 calls, 116 ok, 4 generation errors, $3.212619 metered in this JSONL. Smoke added ~$0.55 separately.

Read: variant E looks strongest on deterministic signals among variants with 0.0% banned-pattern hits: schema pass 95.7%, mean key facts 6.09, mean entities 3.22, mean unsupported entities 1.00. Variant A fails schema_gate only on the missing legacy `resolved_query` key; it has `resolved_queries` x3. Variant B uses the faceted-tagger schema, not the enrichment schema. LLM judge pass is separate/pending.
