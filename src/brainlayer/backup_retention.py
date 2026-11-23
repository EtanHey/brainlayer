"""Pure database retention selection; no filesystem or Drive mutations."""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_DAILY = re.compile(r"(\d{4}-\d{2}-\d{2})\.db\.(?:gz|zst)\Z")
_MD5 = re.compile(r"[a-fA-F0-9]{32}\Z")


@dataclass(frozen=True)
class DriveRetentionRule:
    daily_keep: int = 7
    monthly_keep: int = 3

    def __post_init__(self):
        if type(self.daily_keep) is not int or self.daily_keep < 1:
            raise ValueError("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP must be a positive integer")
        if type(self.monthly_keep) is not int or not 0 <= self.monthly_keep <= 120:
            raise ValueError("BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP must be an integer from 0 to 120")

    def data(self) -> dict[str, Any]:
        return {
            "daily_keep": self.daily_keep,
            "monthly_keep": self.monthly_keep,
            "per_machine": True,
            "description": (
                f"{self.daily_keep} daily + one per month for {self.monthly_keep} months, per Mac"
                if self.monthly_keep
                else f"{self.daily_keep} daily, per Mac"
            ),
        }


def configured_drive_rule(env: Mapping[str, str] | None = None) -> DriveRetentionRule:
    """Runtime configuration affects read-only selection, never Drive-write authority."""
    env = os.environ if env is None else env
    values = []
    for name, default in (("DAILY", 7), ("MONTHLY", 3)):
        key = f"BRAINLAYER_BACKUP_DRIVE_{name}_KEEP"
        raw = env.get(key, str(default))
        if not re.fullmatch(r"[0-9]+", raw):
            raise ValueError(f"{key} must be an integer")
        values.append(int(raw))
    return DriveRetentionRule(*values)


def empty_drive_selection(machine_id: str, *, state: str = "not_run", selected_at=None, error=None) -> dict:
    return {
        "machine_id": machine_id,
        "state": state,
        "selected_at": selected_at,
        "error": error,
        "on_drive": None,
        "named_one_offs_count": None,
        "kept": None,
        "kept_by_rule": None,
        "unverified": None,
        "would_remove": None,
        "monthly_representatives": [],
    }


def daily_date(name: str) -> dt.date | None:
    match = _DAILY.fullmatch(name)
    if match:
        try:
            return dt.date.fromisoformat(match[1])
        except ValueError:
            pass
    return None


def verified_drive_receipts(receipts: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Latest receipt per Drive object wins, including a later revoked proof."""
    latest = {}
    for receipt in receipts:
        if receipt.get("event") == "local_sweep":
            continue
        drive = receipt.get("drive_file")
        if isinstance(drive, dict) and isinstance(drive.get("id"), str):
            latest[drive["id"]] = receipt
    return {
        file_id: receipt
        for file_id, receipt in latest.items()
        if receipt.get("uploaded") is True
        and receipt.get("verified") is True
        and not receipt.get("verification_error")
        and receipt.get("drive_md5_match") is not False
        and isinstance(receipt.get("local_md5"), str)
        and _MD5.fullmatch(receipt["local_md5"])
        and receipt["drive_file"].get("md5Checksum") == receipt["local_md5"]
    }


def matching_verified_dailies(
    files: Iterable[Mapping[str, Any]],
    receipts: Iterable[Mapping[str, Any]],
    *,
    machine_id: str,
    folder_id: str,
) -> list[Mapping[str, Any]]:
    """Current receipt-MD5-proven compressed dailies in this exact machine folder."""
    proofs = verified_drive_receipts(receipts)
    verified = {}
    for item in files:
        name = item.get("name")
        file_id = item.get("id")
        if not isinstance(file_id, str) or not file_id:
            continue
        properties = item.get("appProperties")
        proof = proofs.get(file_id)
        if (
            not isinstance(name, str)
            or daily_date(name) is None
            or item.get("trashed")
            or not isinstance(item.get("parents"), list)
            or item["parents"] != [folder_id]
            or not isinstance(properties, dict)
            or properties.get("brainlayer_machine") != machine_id
            or proof is None
            or proof.get("machine_id") != machine_id
        ):
            continue
        recorded = proof["drive_file"]
        if (
            recorded.get("name") == name
            and item.get("size") is not None
            and str(recorded.get("size")) == str(item["size"])
            and item.get("md5Checksum") == proof["local_md5"]
        ):
            verified[file_id] = item
    return sorted(verified.values(), key=lambda item: (item["name"][:10], item["name"], item["id"]), reverse=True)


def select_drive_dailies(
    files: Iterable[Mapping[str, Any]],
    receipts: Iterable[Mapping[str, Any]],
    *,
    machine_id: str,
    folder_id: str,
    keep_latest: int = 7,
    protected_names: set[str] | frozenset[str] = frozenset(),
) -> list[Mapping[str, Any]]:
    """Read-only hypothetical selection. This never authorizes a Drive mutation."""
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    ordered = matching_verified_dailies(files, receipts, machine_id=machine_id, folder_id=folder_id)
    return [item for item in ordered[keep_latest:] if item["name"] not in protected_names]


def drive_retention_report(
    files: Iterable[Mapping[str, Any]],
    *,
    machine_id: str,
    folder_id: str,
    as_of: dt.date,
    rule: DriveRetentionRule | None = None,
    receipts: Iterable[Mapping[str, Any]] = (),
    protected_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Metadata-only choices, including historical raw dated .db files; no action API."""
    rule = configured_drive_rule() if rule is None else rule
    files = list(files)
    verified_ids = {
        item["id"] for item in matching_verified_dailies(files, receipts, machine_id=machine_id, folder_id=folder_id)
    }
    dated, one_offs, seen = [], set(), {}
    for item in files:
        name = item.get("name", "")
        if item.get("parents") != [folder_id] or item.get("trashed"):
            continue
        properties = item.get("appProperties")
        if not isinstance(properties, dict) or properties.get("brainlayer_machine") != machine_id:
            continue
        if not isinstance(item.get("id"), str) or not item["id"] or not isinstance(name, str):
            continue
        if item.get("mimeType") == "application/vnd.google-apps.folder":
            continue
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.db(?:\.(?:gz|zst))?", name)
        if not match:
            one_offs.add(item["id"])
            continue
        try:
            day = dt.date.fromisoformat(match[1])
        except ValueError:
            one_offs.add(item["id"])
            continue
        try:
            size = int(item["size"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Drive daily backup has no measurable size") from exc
        if size < 0 or isinstance(item["size"], bool):
            raise ValueError("Drive daily backup size is invalid")
        row = {"id": item["id"], "name": name, "bytes": size, "date": day.isoformat()}
        if item["id"] in seen and seen[item["id"]] != row:
            raise ValueError("Drive listing repeats an object with conflicting metadata")
        if item["id"] not in seen:
            dated.append(row)
            seen[item["id"]] = row
    dated.sort(key=lambda item: (item["date"], item["name"], item["id"]), reverse=True)
    verified = [item for item in dated if item["id"] in verified_ids]
    unverified = [item for item in dated if item["id"] not in verified_ids]

    def summary(rows):
        return {
            "count": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "oldest": min((row["date"] for row in rows), default=None),
            "newest": max((row["date"] for row in rows), default=None),
        }

    def select(policy):
        daily = verified[: policy.daily_keep]
        daily_ids = {item["id"] for item in daily}
        representatives, months = [], []
        for offset in range(1, policy.monthly_keep + 1):
            year, month0 = divmod(as_of.year * 12 + as_of.month - 1 - offset, 12)
            month = f"{year:04}-{month0 + 1:02}"
            months.append(month)
            representative = next((item for item in verified if item["date"].startswith(month)), None)
            if representative is not None:
                representatives.append(representative)
        extra = [item for item in representatives if item["id"] not in daily_ids]
        kept_ids = daily_ids | {item["id"] for item in extra}
        removed = [item for item in verified if item["id"] not in kept_ids and item["name"] not in protected_names]
        removed_ids = {item["id"] for item in removed}
        return (
            months,
            representatives,
            extra,
            [item for item in dated if item["id"] not in removed_ids],
            removed,
            daily + extra,
        )

    months, _, monthly, kept, removed, _ = select(DriveRetentionRule())
    _, representatives, _, configured_kept, configured_removed, kept_by_rule = select(rule)
    return {
        "mode": "report-only",
        "machine": machine_id,
        "daily": summary(dated),
        "keep_7_plus_monthly": {
            "months": months,
            "extra": summary(monthly),
            "extra_files": monthly,
            "kept": summary(kept),
            "would_remove": summary(removed),
        },
        "configured": {
            "as_of": as_of.isoformat(),
            "rule": rule.data(),
            "on_drive": summary(dated),
            "named_one_offs_count": len(one_offs),
            "kept": summary(configured_kept),
            "kept_by_rule": summary(kept_by_rule),
            "unverified": summary(unverified),
            "would_remove": summary(configured_removed),
            "monthly_representatives": [
                {key: item[key] for key in ("name", "date", "bytes")} for item in representatives
            ],
        },
    }


def local_keep_names(names: Iterable[str], verified_names: set[str], *, keep_latest: int) -> set[str]:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    verified = sorted({name for name in names if daily_date(name) is not None and name in verified_names}, reverse=True)
    return set(verified[:keep_latest])
