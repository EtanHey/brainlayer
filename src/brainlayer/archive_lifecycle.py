"""Canonical chunk archive predicate: archived_at is UTC time or NULL.

Etan 2026-08-16: "archived_at should be time or null, not that+archived."
Lineage columns superseded_by / aggregated_into stay; they are not archive flags.
"""

from __future__ import annotations


def lifecycle_active_clauses(alias: str = "") -> list[str]:
    """Default-search predicates: lineage null and archived_at null."""
    prefix = f"{alias}." if alias else ""
    return [
        f"{prefix}superseded_by IS NULL",
        f"{prefix}aggregated_into IS NULL",
        f"{prefix}archived_at IS NULL",
    ]


def lifecycle_active_sql(alias: str = "") -> str:
    return " AND ".join(lifecycle_active_clauses(alias))


def lifecycle_active_clauses_present(cols: set[str], alias: str = "") -> list[str]:
    """Same predicates, skipping columns the live schema does not have."""
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    for column in ("superseded_by", "aggregated_into", "archived_at"):
        if column in cols:
            clauses.append(f"{prefix}{column} IS NULL")
    return clauses
