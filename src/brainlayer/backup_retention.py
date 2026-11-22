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


def select_drive_dailies(
    files: Iterable[Mapping[str, Any]],
    receipts: Iterable[Mapping[str, Any]],
    *,
    machine_id: str,
    folder_id: str,
    keep_latest: int = 7,
    protected_names: set[str] | frozenset[str] = frozenset(),
) -> list[Mapping[str, Any]]:
    """Only current, receipt-proven objects in this machine's exact folder count.

    Unknown files are neither counted nor selected. Pin protection is additional
    to the count so it cannot displace the newest verified daily snapshots.
    """
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    proofs = verified_drive_receipts(receipts)
    verified = {}
    for item in files:
        name = item.get("name")
        file_id = item.get("id")
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
    ordered = sorted(verified.values(), key=lambda item: (item["name"][:10], item["name"], item["id"]), reverse=True)
    return [item for item in ordered[keep_latest:] if item["name"] not in protected_names]


def local_keep_names(names: Iterable[str], verified_names: set[str], *, keep_latest: int) -> set[str]:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    verified = sorted({name for name in names if daily_date(name) is not None and name in verified_names}, reverse=True)
    return set(verified[:keep_latest])
