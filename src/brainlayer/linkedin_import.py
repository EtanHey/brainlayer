"""LinkedIn KG import with an explicit copy or SHA-confirmed live mode. Output contains counts, never contact records.

Run: python -m brainlayer.linkedin_import EXPORT.json --db-copy COPY.db
Identity matching is exact: LinkedIn URL, email, then normalized full name.
First/last names stay in metadata; name aliases must contain multiple tokens.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from .disk_space import measure_disk_space
from .kg_repo import KGMixin
from .paths import resolve_db_path
from .pipeline.entity_resolution import _normalize_name
from .runtime_store import RuntimeStoreModeError, open_writer_store

SOURCE = "linkedin-export 2026-10-31 (portfolio scrub)"
ATTRIBUTES = (
    "first_name",
    "last_name",
    "linkedin_url",
    "company",
    "position",
    "connected_on",
    "relationship_strength",
    "has_messages",
)


class ImportRefusal(ValueError):
    """An importer-owned static reason; never carries source values."""

    def __init__(self, reason):
        if reason not in {
            "copy_refusal",
            "invalid_export",
            "disk_floor",
            "confirmation_required",
            "confirmation_sha256_mismatch",
        }:
            raise ValueError("unknown_import_reason")
        self.reason = reason
        super().__init__(reason)


def normalized(value):
    return " ".join(_normalize_name(value or "").split())


def email_key(value):
    return (value or "").strip().casefold()


def linkedin_key(value):
    if not value:
        return ""
    parsed = urlsplit(value if "://" in value else "https://" + value)
    if not re.fullmatch(r"(?:www\.|[a-z]{2}\.)?linkedin\.com", parsed.hostname or "") or not parsed.path.startswith(
        "/in/"
    ):
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
        raise ImportRefusal("copy_refusal")
    return path


class _ImportGraph(KGMixin):
    """Use the supported KG writers on one connection, including read-your-writes."""

    def __init__(self, store):
        self.conn, self.db_path = store.conn, store.db_path

    def _read_cursor(self):
        return self.conn.cursor()


def import_connections(store, export):
    """Import atomically on the caller's supported writer; never opens a database."""
    started = time.monotonic()
    raw = export if isinstance(export, bytes) else Path(export).read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ImportRefusal("invalid_export")
    store.conn.execute("BEGIN IMMEDIATE")
    try:
        result = _import_rows(_ImportGraph(store), rows, hashlib.sha256(raw).hexdigest(), started)
        store.conn.execute("COMMIT")
        return result
    except BaseException:
        store.conn.execute("ROLLBACK")
        raise


def _import_rows(store, rows, sha256, started):
    provenance = {"source": SOURCE, "sha256": sha256}
    counts = {key: {"new": 0, "merged": 0} for key in ("persons", "companies", "relations")}
    counts["persons"].update(merged_by=dict.fromkeys(("linkedin_url", "email", "name"), 0), new_name_conflict=0)
    counts["companies"]["merged_by"] = {"name": 0}
    skipped = Counter()
    invalid_existing_alias = 0
    identities = defaultdict(lambda: defaultdict(set))

    def index_identity(kind, key, eid):
        if key:
            indexes[kind][key].add(eid)
            identities[eid][kind].add(key)

    indexes = {kind: defaultdict(set) for kind in ("person", "company", "linkedin_url", "email")}
    for eid, kind, name, meta in store.conn.execute(
        "SELECT id, entity_type, name, metadata FROM kg_entities WHERE entity_type IN ('person','company')"
    ):
        indexes[kind][normalized(name)].add(eid)
        if kind == "person":
            metadata = json.loads(meta or "{}")
            if metadata.get("linkedin_full_name"):
                indexes[kind][normalized(metadata["linkedin_full_name"])].add(eid)
            try:
                index_identity("linkedin_url", linkedin_key(metadata.get("linkedin_url")), eid)
            except ValueError:
                pass
            contact = metadata.get("contact_info") or {}
            for value in (metadata.get("email"), contact.get("email")):
                index_identity("email", email_key(value), eid)
    for alias, eid, kind in store.conn.execute(
        "SELECT a.alias, a.entity_id, a.alias_type FROM kg_entity_aliases a "
        "JOIN kg_entities e ON e.id=a.entity_id WHERE e.entity_type='person' "
        "AND a.alias_type IN ('name','linkedin_url','email')"
    ):
        if kind == "name":
            indexes["person"][normalized(alias)].add(eid)
            continue
        try:
            key = linkedin_key(alias) if kind == "linkedin_url" else email_key(alias)
        except ValueError:
            invalid_existing_alias += 1
            continue
        index_identity(kind, key, eid)
    owner_ids = indexes["person"].get(normalized("Etan Heyman"), set())
    owner = next(iter(owner_ids)) if len(owner_ids) == 1 else None
    seen_people, seen_companies, seen_relations = set(), set(), set()

    def upsert(kind, name, eid, attributes, row_number=None, name_conflict=False, identity_key=None):
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
        storage_name = old["name"] if old else name
        if name_conflict:
            # The KG enforces UNIQUE(type, name). Keep the original as a full-name alias.
            suffix = hashlib.sha256(identity_key.encode()).hexdigest()[:16]
            storage_name = f"{name} [linkedin-{suffix}]"
            metadata["linkedin_full_name"] = name
        stored = store.upsert_entity(eid or f"{kind}-{uuid4().hex}", kind, storage_name, metadata=metadata)
        indexes[kind][normalized(name)].add(stored)
        if kind == "person" and len(name.split()) > 1:
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
        if measure_disk_space(store.db_path.parent, include_available=False).free_now_bytes < 2 * 1024**3:
            raise ImportRefusal("disk_floor")
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
        email = email_key(row.get("email"))
        match_kind, matches = next(
            (
                (kind, indexes[kind][key])
                for kind, key in (("linkedin_url", url), ("email", email))
                if key and indexes[kind].get(key)
            ),
            ("name", set()),
        )
        name_candidates = indexes["person"].get(normalized(name), set())
        name_conflict = False
        if not matches:
            matches = {
                eid
                for eid in name_candidates
                if not any(
                    key and identities[eid][kind] and identities[eid][kind] != {key}
                    for kind, key in (("linkedin_url", url), ("email", email))
                )
            }
            name_conflict = bool(name_candidates and not matches)
        if matches and name_candidates - matches:
            skipped["ambiguous_identity"] += 1
            continue
        if len(matches) > 1:
            skipped["ambiguous_identity"] += 1
            continue
        eid = next(iter(matches), None)
        if eid and indexes["person"].get(normalized(store.get_entity(eid)["name"]), set()) - {eid}:
            skipped["ambiguous_identity"] += 1
            continue
        if eid in seen_people:
            skipped["duplicate_in_file"] += 1
            continue
        eid = upsert("person", name, eid, {k: row.get(k) for k in ATTRIBUTES}, number, name_conflict, url or email)
        counts["persons"]["merged" if matches else "new"] += 1
        counts["persons"]["new_name_conflict"] += int(name_conflict)
        if matches:
            counts["persons"]["merged_by"][match_kind] += 1
        seen_people.add(eid)
        for kind, key in (("linkedin_url", url), ("email", email)):
            if key:
                index_identity(kind, key, eid)
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
            counts["companies"]["merged_by"]["name"] += int(bool(matches))
            seen_companies.add(key)
        relation(eid, cid, "works_at", {"position": row.get("position")})
    return {
        **counts,
        "rows": len(rows),
        "invalid_existing_alias": invalid_existing_alias,
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--db-copy", type=Path)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-sha256")
    args = parser.parse_args()
    previous_logging_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)  # Library diagnostics can contain personal data.
    store = None
    try:
        if args.live and not args.confirm_sha256:
            raise ImportRefusal("confirmation_required")
        raw = args.export.read_bytes()  # Hash and import these same bytes; never reopen the export.
        if args.live and hashlib.sha256(raw).hexdigest() != args.confirm_sha256:
            raise ImportRefusal("confirmation_sha256_mismatch")
        path = resolve_db_path() if args.live else require_copy(args.db_copy)
        store = open_writer_store(path)
        print(json.dumps(import_connections(store, raw), sort_keys=True))
    except Exception as exc:
        reason = (
            exc.reason
            if isinstance(exc, ImportRefusal)
            else "invalid_export"
            if isinstance(exc, json.JSONDecodeError)
            else "invalid_store_mode"
            if isinstance(exc, RuntimeStoreModeError)
            else "import_failed"
        )
        print(json.dumps({"error_type": type(exc).__name__, "reason": reason}))
        return 1
    finally:
        if store is not None:
            store.close()
        logging.disable(previous_logging_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
