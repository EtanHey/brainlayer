# Enrichment Provider-Agnostic Design

## Scope

This note documents the provider seam for cloud enrichment. It does not implement
additional providers. The current realtime path is Gemini-specific, while the
local enrichment backend already uses `BRAINLAYER_ENRICH_BACKEND` for `mlx` and
`ollama`, and `build_external_prompt()` already produces a provider-neutral
prompt plus sanitized chunk context.

## Provider Selector

Use `BRAINLAYER_ENRICH_PROVIDER` as the cloud provider selector. The current
default should remain `gemini` for backwards compatibility. Keep
`BRAINLAYER_ENRICH_BACKEND` for the existing local backend selector; do not
reuse it for cloud provider choice.

Related launchd config should use `BRAINLAYER_LAUNCHD_DRAIN_ENABLED` for the
drain service gate, matching the unified config-file worker's key namespace.

## Provider Interface

Add one narrow adapter seam behind realtime enrichment:

- client factory: create the provider client from environment/config
- request adapter: map the `build_external_prompt()` output to provider request
  fields
- response adapter: normalize provider output into the existing enrichment JSON
  shape consumed by `parse_enrichment()`
- error adapter: normalize rate-limit, daily-cap, timeout, and retryable errors

The enrichment controller should keep owning candidate selection, duplicate
marking, rate limiting, telemetry, and database writes. Provider adapters should
not know about SQLite rows or write queues.

## Adding a Provider

A new provider needs:

- a client factory keyed by `BRAINLAYER_ENRICH_PROVIDER`
- model/config environment keys under a provider-specific prefix
- a prompt/request adapter that accepts the provider-neutral prompt
- a response parser that returns the same JSON fields used today
- retry/cap error mapping compatible with the existing supervisor behavior

## Current Blockers

The realtime path still names Gemini-specific helpers and config builders
directly. Before adding OpenAI, Anthropic, or another cloud provider, split those
helpers into a small provider module and leave `enrich_realtime()` calling only
the provider interface. Keep the first refactor provider-preserving so Gemini
behavior remains unchanged.
