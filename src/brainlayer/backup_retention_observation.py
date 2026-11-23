"""Project recorded read-only selections; no Drive service or filesystem access."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

from .backup_retention import configured_drive_rule, empty_drive_selection


def _summary(value):
    if not isinstance(value, dict):
        raise ValueError("missing selection summary")
    result = {key: value[key] for key in ("count", "bytes", "oldest", "newest")}
    if any(type(result[k]) is not int or result[k] < 0 for k in ("count", "bytes")):
        raise ValueError("invalid selection counts")
    if result["count"] == 0:
        if result["bytes"] != 0 or result["oldest"] is not None or result["newest"] is not None:
            raise ValueError("invalid empty selection summary")
    elif dt.date.fromisoformat(result["oldest"]) > dt.date.fromisoformat(result["newest"]):
        raise ValueError("invalid selection date range")
    return result


def drive_retention_observation(
    records: Sequence[Mapping],
    *,
    env: Mapping[str, str],
    now: dt.datetime,
    machine_id: str,
    writes_enabled: bool,
    input_error: str | None = None,
) -> dict:
    try:
        rule, config_error = configured_drive_rule(env).data(), None
    except ValueError as exc:
        rule, config_error = None, str(exc)
    latest = {}
    for row in records:
        observation = row.get("drive_retention_selection")
        if row.get("backup_log_provenance") != "real" or not isinstance(observation, dict):
            continue
        machine = observation.get("machine_id")
        if isinstance(machine, str) and machine:
            latest[machine] = observation
    latest.setdefault(machine_id, {})
    machines = []
    for machine, observation in sorted(latest.items()):
        result = empty_drive_selection(machine)
        try:
            if observation:
                at = dt.datetime.fromisoformat(observation["selected_at"].replace("Z", "+00:00"))
                if at.tzinfo is None or at > now:
                    raise ValueError("selection timestamp is naive or in the future")
                result["selected_at"] = at.astimezone(dt.UTC).isoformat()
            if input_error or config_error:
                raise ValueError(input_error or config_error)
            if not observation:
                machines.append(result)
                continue
            if observation.get("state") == "unmeasurable":
                raise ValueError(observation.get("error") or "selection failed without a reason")
            if observation.get("state") != "measured":
                raise ValueError("unknown recorded selection state")
            if observation.get("rule") != rule:
                result.update(state="stale", error="Drive rule changed since the last selection")
                machines.append(result)
                continue
            as_of = dt.date.fromisoformat(observation.get("as_of", at.date().isoformat()))
            local_now = now.astimezone()
            if rule["monthly_keep"] and (as_of.year, as_of.month) != (local_now.year, local_now.month):
                result.update(state="stale", error="Monthly retention window changed since the last selection")
                machines.append(result)
                continue
            for key in ("on_drive", "kept", "would_remove"):
                result[key] = _summary(observation[key])
            for key in ("count", "bytes"):
                if result["on_drive"][key] != result["kept"][key] + result["would_remove"][key]:
                    raise ValueError("recorded selection totals are inconsistent")
            count = observation["named_one_offs_count"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid named one-off count")
            representatives = []
            for item in observation["monthly_representatives"]:
                if not isinstance(item["name"], str) or type(item["bytes"]) is not int or item["bytes"] < 0:
                    raise ValueError("invalid monthly representative")
                dt.date.fromisoformat(item["date"])
                representatives.append({key: item[key] for key in ("name", "date", "bytes")})
            if len(representatives) > rule["monthly_keep"]:
                raise ValueError("too many monthly representatives")
            result.update(state="measured", named_one_offs_count=count, monthly_representatives=representatives)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result = empty_drive_selection(
                machine, state="unmeasurable", selected_at=result["selected_at"], error=str(exc)
            )
        machines.append(result)
    return {"rule": rule, "writes_enabled": writes_enabled, "config_error": config_error, "machines": machines}
