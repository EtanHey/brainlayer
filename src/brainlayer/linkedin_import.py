"""Copy-only LinkedIn KG import. Output contains counts, never contact records.

Run: python -m brainlayer.linkedin_import EXPORT.json --db-copy COPY.db
Identity matching is exact: LinkedIn URL, email, then normalized full name.
First/last-name aliases are discoverable but never used as identity keys.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from .pipeline.entity_resolution import _normalize_name
from .runtime_store import open_writer_store

SOURCE = "linkedin-export 2026-10-31 (portfolio scrub)"
ATTRIBUTES = ("linkedin_url", "company", "position", "connected_on", "relationship_strength", "has_messages")


def normalized(value):
    return " ".join(_normalize_name(value or "").split())


def linkedin_key(value):
    if not value:
        return ""
    parsed = urlsplit(value if "://" in value else "https://" + value)
    if parsed.hostname not in {"linkedin.com", "www.linkedin.com"} or not parsed.path.startswith("/in/"):
        raise ValueError("invalid_linkedin_url")
    return "https://linkedin.com" + parsed.path.rstrip("/").casefold()


def require_copy(path, live=None):
    """Refuse canonical, configured-live, symlink and hardlink aliases before opening."""
    from .paths import get_db_path

    path = Path(path).expanduser().resolve(strict=True)
    protected = [Path.home() / ".local/share/brainlayer/brainlayer.db", get_db_path()]
    if live is not None:
        protected.append(Path(live))
    if any(path == p.expanduser().resolve() or (p.exists() and os.path.samefile(path, p)) for p in protected):
        raise ValueError("db-copy must be a separate copy")
    return path


def import_connections(store, export):
    """Apply an export to an explicitly supplied store using supported KG writers.

    Existing metadata is retained. Ambiguous identities are skipped for review.
    The caller owns the store; this function never opens the canonical database.
    """
    started = time.monotonic()
    raw = Path(export).read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("invalid_export")
    provenance = {"source": SOURCE, "sha256": hashlib.sha256(raw).hexdigest()}
    counts = {key: {"new": 0, "merged": 0} for key in ("persons", "companies", "relations")}
    skipped = Counter()
    indexes = {kind: defaultdict(set) for kind in ("person", "company", "linkedin_url", "email")}
    for eid, kind, name, meta in store.conn.execute(
        "SELECT id, entity_type, name, metadata FROM kg_entities WHERE entity_type IN ('person','company')"
    ):
        indexes[kind][normalized(name)].add(eid)
        if kind == "person":
            metadata = json.loads(meta or "{}")
            try:
                indexes["linkedin_url"][linkedin_key(metadata.get("linkedin_url"))].add(eid)
            except ValueError:
                pass
            contact = metadata.get("contact_info") or {}
            indexes["email"][normalized(metadata.get("email") or contact.get("email"))].add(eid)
    for alias, eid, kind in store.conn.execute(
        "SELECT a.alias, a.entity_id, a.alias_type FROM kg_entity_aliases a "
        "JOIN kg_entities e ON e.id=a.entity_id WHERE e.entity_type='person' "
        "AND a.alias_type IN ('linkedin_url','email')"
    ):
        key = linkedin_key(alias) if kind == "linkedin_url" else normalized(alias)
        indexes[kind][key].add(eid)
    owner_ids = indexes["person"].get(normalized("Etan Heyman"), set())
    owner = next(iter(owner_ids)) if len(owner_ids) == 1 else None
    seen_people, seen_companies, seen_relations = set(), set(), set()

    def upsert(kind, name, eid, attributes, row_number=None):
        old = store.get_entity(eid) if eid else None
        metadata = dict(old["metadata"]) if old else {}
        metadata.update({k: v for k, v in attributes.items() if v is not None and v != ""})
        evidence = dict(provenance)
        if row_number is not None:
            evidence["row_number"] = row_number
        sources = list(metadata.get("linkedin_imports", []))
        if evidence not in sources:
            sources.append(evidence)
        metadata["linkedin_imports"] = sources
        stored = store.upsert_entity(
            eid or f"{kind}-{uuid4().hex}", kind, old["name"] if old else name, metadata=metadata
        )
        indexes[kind][normalized(name)].add(stored)
        store.add_entity_alias(name, stored)
        return stored

    def relation(source, target, kind, properties):
        key = (source, target, kind)
        if key in seen_relations:
            return
        seen_relations.add(key)
        existing = next(
            (
                r
                for r in store.get_entity_relations(source, direction="outgoing")
                if r["target_id"] == target and r["relation_type"] == kind
            ),
            None,
        )
        props = dict(existing["properties"]) if existing else {}
        props.update({k: v for k, v in properties.items() if v is not None and v != ""})
        sources = list(props.get("linkedin_imports", []))
        if provenance not in sources:
            sources.append(provenance)
        props["linkedin_imports"] = sources
        store.add_relation(
            f"rel-{uuid4().hex}",
            source,
            target,
            kind,
            properties=props,
            confidence=existing["confidence"] if existing else 1.0,
            importance=existing["importance"] if existing else 0.5,
        )
        counts["relations"]["merged" if existing else "new"] += 1

    for number, row in enumerate(rows, 1):
        from .disk_space import measure_disk_space

        if measure_disk_space(store.db_path.parent, include_available=False).free_now_bytes < 2 * 1024**3:
            raise ValueError("disk_floor")
        if (
            not isinstance(row, dict)
            or any(
                value is not None and not isinstance(value, str) for key, value in row.items() if key != "has_messages"
            )
            or row.get("has_messages") not in (None, True, False)
        ):
            skipped["invalid_row"] += 1
            continue
        name = " ".join((row.get("full_name") or "").split())
        if not name:
            skipped["missing_name"] += 1
            continue
        try:
            url = linkedin_key(row.get("linkedin_url"))
        except ValueError:
            skipped["invalid_linkedin_url"] += 1
            continue
        email = normalized(row.get("email"))
        matches = next(
            (
                indexes[kind][key]
                for kind, key in (("linkedin_url", url), ("email", email), ("person", normalized(name)))
                if key and indexes[kind].get(key)
            ),
            set(),
        )
        if len(matches) > 1:
            skipped["ambiguous_identity"] += 1
            continue
        eid = next(iter(matches), None)
        if eid in seen_people:
            skipped["duplicate_in_file"] += 1
            continue
        eid = upsert("person", name, eid, {k: row.get(k) for k in ATTRIBUTES}, number)
        counts["persons"]["merged" if matches else "new"] += 1
        seen_people.add(eid)
        for field in ("first_name", "last_name"):
            if row.get(field):
                store.add_entity_alias(row[field].strip(), eid, alias_type=field)
        for kind, key in (("linkedin_url", url), ("email", email)):
            if key:
                indexes[kind][key].add(eid)
                store.add_entity_alias(key, eid, alias_type=kind)
        if owner and owner != eid:
            relation(
                eid,
                owner,
                "linkedin_connection",
                {k: row.get(k) for k in ("connected_on", "relationship_strength", "has_messages")},
            )
        company = " ".join((row.get("company_normalized") or row.get("company") or "").split())
        if not company:
            continue
        key = normalized(company)
        matches = indexes["company"].get(key, set())
        if len(matches) > 1:
            skipped["ambiguous_company"] += 1
            continue
        cid = next(iter(matches), None)
        if key not in seen_companies:
            cid = upsert("company", company, cid, {})
            counts["companies"]["merged" if matches else "new"] += 1
            seen_companies.add(key)
        relation(eid, cid, "works_at", {"position": row.get("position")})
    return {
        **counts,
        "rows": len(rows),
        "skipped": dict(skipped),
        "owner_found": owner is not None,
        "duplicates_within_file": skipped["duplicate_in_file"],
        "rows_missing_name": skipped["missing_name"],
        "seconds": round(time.monotonic() - started, 3),
        "sha256": provenance["sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--db-copy", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)  # Library diagnostics can contain personal data.
    store = None
    try:
        path = require_copy(args.db_copy)
        store = open_writer_store(path)
        print(json.dumps(import_connections(store, args.export), sort_keys=True))
    except Exception:
        print(json.dumps({"error": "import_failed; no row details emitted"}))
        return 1
    finally:
        if store is not None:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
