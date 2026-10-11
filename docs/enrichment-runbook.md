# Enrichment Runbook (retired)

LLM enrichment jobs, schedulers, and cloud backfill are retired. This URL remains
a retirement notice; do not run or resume the historical enrichment jobs.
History: [CHANGELOG](https://github.com/EtanHey/brainlayer/blob/main/CHANGELOG.md).

Existing metadata remains readable. Local indexing, search, digest, knowledge graph,
offline evaluation, prompt emission/collection, and local checkpoint replay remain
available. See [historical metadata](enrichment.md) for the retained field reference.

## Google credential compatibility gate

Retain the 1Password-backed `GOOGLE_API_KEY` configuration and
`BRAINLAYER_REQUIRE_GOOGLE_API_KEY` flag for older installed hotlane plists.
`brainlayer-env-run.sh` can still exit **78** when required key configuration is
missing. Remove neither the credential gate nor its configuration until the release
re-renders those installed plists. This compatibility requirement does not enable
enrichment. See [Configuration](configuration.md) for the key reference and gate details.
