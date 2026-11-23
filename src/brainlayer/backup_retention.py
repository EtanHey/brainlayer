"""Pure database retention selection; no filesystem or Drive mutations."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping
from typing import Any

_DAILY = re.compile(r"(\d{4}-\d{2}-\d{2})\.db\.(?:gz|zst)\Z")
_MD5 = re.compile(r"[a-fA-F0-9]{32}\Z")


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
) -> dict[str, Any]:
    """Metadata-only choices, including historical raw dated .db files; no action API."""
    dated = []
    for item in files:
        name = item.get("name", "")
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.db(?:\.(?:gz|zst))?", name)
        if not match or item.get("parents") != [folder_id] or item.get("trashed"):
            continue
        if item.get("appProperties", {}).get("brainlayer_machine") != machine_id:
            continue
        try:
            day = dt.date.fromisoformat(match[1])
            size = int(item["size"])
        except (ValueError, TypeError, KeyError):
            continue
        if size >= 0:
            dated.append({"id": item["id"], "name": name, "bytes": size, "date": day.isoformat()})
    dated.sort(key=lambda item: (item["date"], item["name"], item["id"]), reverse=True)

    def summary(rows):
        return {
            "count": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "oldest": min((row["date"] for row in rows), default=None),
            "newest": max((row["date"] for row in rows), default=None),
        }

    keep = dated[:7]
    keep_ids = {item["id"] for item in keep}
    monthly = []
    months = []
    for offset in (1, 2, 3):
        year, month0 = divmod(as_of.year * 12 + as_of.month - 1 - offset, 12)
        month = f"{year:04}-{month0 + 1:02}"
        months.append(month)
        representative = next((item for item in dated if item["date"].startswith(month)), None)
        if representative is not None and representative["id"] not in keep_ids:
            monthly.append(representative)
    extra_ids = {item["id"] for item in monthly}
    return {
        "mode": "report-only",
        "machine": machine_id,
        "daily": summary(dated),
        "keep_7": {"kept": summary(keep), "would_remove": summary(dated[7:])},
        "keep_7_plus_monthly": {
            "months": months,
            "extra": summary(monthly),
            "extra_files": monthly,
            "kept": summary(keep + monthly),
            "would_remove": summary([item for item in dated if item["id"] not in keep_ids | extra_ids]),
        },
    }


def local_keep_names(names: Iterable[str], verified_names: set[str], *, keep_latest: int) -> set[str]:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    verified = sorted({name for name in names if daily_date(name) is not None and name in verified_names}, reverse=True)
    return set(verified[:keep_latest])
