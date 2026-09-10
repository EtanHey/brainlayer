"""Extract declared routing targets from native tool-call envelopes only."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

MAX_CALLS = 25
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_TARGET_BYTES = 4096
MAX_CALL_ID_BYTES = 4096
MAX_DIAGNOSTICS = 32
_SUPPORTED = {
    "mcp__cmuxlayer__send_to": (("agent_id", "surface"), "agent_handle"),
    "mcp__codex_app__send_message_to_thread": (("threadId",), "task_handle"),
    "collaboration.send_message": (("target",), "agent_handle"),
}


@dataclass(frozen=True)
class NativeTarget:
    tool_name: str
    target: str
    target_kind: str
    declared_unverified: bool
    call_id: str | None
    raw_container_pointer: str | None
    decoded_pointer: str
    pointer_kind: str


@dataclass(frozen=True)
class NativeTargetReport:
    targets: list[dict[str, Any]]
    diagnostics: list[str]


def _pointer(parent: str, key: str | int) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def _scalar(value: Any, *, byte_cap: int = MAX_TARGET_BYTES) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        if len(value.encode("utf-8")) > byte_cap:
            return None
    except UnicodeEncodeError:
        return None
    return value


def _bounded_json_size(value: Any, *, depth: int = 0) -> int:
    if depth > 100:
        raise ValueError("structured arguments exceed nesting cap")
    if isinstance(value, str):
        if len(value) > MAX_ARGUMENT_BYTES:
            raise ValueError("structured arguments exceed byte cap")
        try:
            return len(value.encode("utf-8")) * 6 + 2
        except UnicodeEncodeError as exc:
            raise ValueError("structured arguments contain invalid text") from exc
    if value is None or isinstance(value, bool):
        return 8
    if isinstance(value, int):
        return value.bit_length() + 2
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("structured arguments contain nonfinite number")
        return 32
    if isinstance(value, list):
        total = 2
        for item in value:
            total += _bounded_json_size(item, depth=depth + 1) + 1
            if total > MAX_ARGUMENT_BYTES:
                raise ValueError("structured arguments exceed byte cap")
        return total
    if isinstance(value, dict):
        total = 2
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("structured arguments contain a non-string key")
            total += _bounded_json_size(key, depth=depth + 1) + _bounded_json_size(item, depth=depth + 1) + 2
            if total > MAX_ARGUMENT_BYTES:
                raise ValueError("structured arguments exceed byte cap")
        return total
    raise ValueError("structured arguments contain an unsupported value")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _one_target(
    *, tool_name: str, container: dict[str, Any], pointer: str, call_id: str | None,
    raw_container_pointer: str | None = None,
) -> tuple[NativeTarget | None, str | None]:
    fields, kind = _SUPPORTED[tool_name]
    present = [(key, _scalar(container.get(key))) for key in fields if key in container]
    if len(present) > 1:
        return None, f"ambiguous target fields for {tool_name}"
    if not present:
        return None, f"missing target field {fields[0]} for {tool_name}"
    field, value = present[0]
    if value is None:
        return None, f"invalid or oversized target for {tool_name}"
    return NativeTarget(
        tool_name=tool_name,
        target=value,
        target_kind=("surface" if field == "surface" else kind),
        declared_unverified=True,
        call_id=call_id,
        raw_container_pointer=raw_container_pointer,
        decoded_pointer=(_pointer(pointer, field) if raw_container_pointer is None else _pointer("", field)),
        pointer_kind=("direct_raw_json" if raw_container_pointer is None else "decoded_json_string"),
    ), None


def extract_native_targets(entry: Any) -> NativeTargetReport:
    """Extract at most MAX_CALLS supported native calls; never inspect prose/results."""
    diagnostics: list[str] = []
    targets: list[dict[str, Any]] = []
    seen_calls = 0

    def diagnose(message: str) -> None:
        if len(diagnostics) < MAX_DIAGNOSTICS:
            diagnostics.append(message)

    def consume(tool_name: Any, container: Any, pointer: str, call_id: Any, raw_pointer: str | None) -> None:
        nonlocal seen_calls
        if not isinstance(tool_name, str) or tool_name not in _SUPPORTED:
            return
        if seen_calls >= MAX_CALLS:
            if seen_calls == MAX_CALLS:
                diagnose(f"omitted calls beyond cap {MAX_CALLS}")
            seen_calls += 1
            return
        seen_calls += 1
        if not isinstance(container, dict):
            diagnose(f"omitted call {pointer}: input is not an object")
            return
        try:
            _bounded_json_size(container)
        except (RecursionError, UnicodeEncodeError, ValueError) as exc:
            diagnose(f"omitted call {pointer}: invalid or oversized structured arguments ({exc})")
            return
        if call_id is not None:
            if _scalar(call_id, byte_cap=MAX_CALL_ID_BYTES) is None:
                diagnose(f"omitted call {pointer}: invalid or oversized call ID")
                return
        target, diagnostic = _one_target(
            tool_name=tool_name, container=container, pointer=pointer,
            call_id=call_id, raw_container_pointer=raw_pointer,
        )
        if target is not None:
            targets.append(asdict(target))
        elif diagnostic:
            diagnose(diagnostic)

    if not isinstance(entry, dict):
        return NativeTargetReport([], ["ignored non-object envelope"])
    if entry.get("type") == "assistant":
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) and message.get("role") == "assistant" else None
        if isinstance(content, list):
            for index, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                consume(block.get("name"), block.get("input"), _pointer("/message/content", index) + "/input", block.get("id"), None)
    elif entry.get("type") == "response_item":
        payload = entry.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "function_call":
            tool_name = payload.get("name")
            if not isinstance(tool_name, str) or tool_name not in _SUPPORTED:
                return NativeTargetReport(targets, diagnostics)
            arguments = payload.get("arguments")
            if not isinstance(arguments, str):
                diagnose("omitted Codex call: arguments must be a JSON string")
            elif len(arguments.encode("utf-8")) > MAX_ARGUMENT_BYTES:
                diagnose("omitted Codex call: arguments exceed byte cap")
            else:
                try:
                    decoded = json.loads(arguments, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
                except (TypeError, RecursionError, json.JSONDecodeError, ValueError, UnicodeEncodeError) as exc:
                    diagnose("omitted Codex call: invalid arguments JSON")
                else:
                    consume(tool_name, decoded, "/payload/arguments", payload.get("call_id"), "/payload/arguments")
    return NativeTargetReport(targets, diagnostics)
