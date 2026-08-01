"""Source-specific ingestion adapters."""

from .t3 import (
    DEFAULT_T3_HEALTH_PATH,
    DEFAULT_T3_STATE_DB,
    T3_PROVENANCE_CLASS,
    T3_SOURCE,
    T3IngestionResult,
    T3Message,
    T3Reader,
    T3SchemaError,
    T3Thread,
    ingest_t3,
    read_t3_threads,
)

__all__ = [
    "DEFAULT_T3_HEALTH_PATH",
    "DEFAULT_T3_STATE_DB",
    "T3_PROVENANCE_CLASS",
    "T3_SOURCE",
    "T3IngestionResult",
    "T3Message",
    "T3Reader",
    "T3SchemaError",
    "T3Thread",
    "ingest_t3",
    "read_t3_threads",
]
