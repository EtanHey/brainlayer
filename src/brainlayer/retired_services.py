"""Retired launchd labels and the historical queue hold identity.

The hold label is retained only to interpret existing maintenance sentinels.
It never authorizes starting an enrichment producer.
"""

RETIRED_ENRICHMENT_LABELS = frozenset({"com.brainlayer.enrich", "com.brainlayer.enrichment"})
LEGACY_ENRICHMENT_HOLD_LABEL = "com.brainlayer.enrichment"
