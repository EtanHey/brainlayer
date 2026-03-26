"""Behavioral assertion helpers for mock MCP call logs.

These work with MockMcpServer instances or raw call name lists.
Designed for pytest — raise AssertionError with descriptive messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import MockMcpServer


def _get_names(source: MockMcpServer | list[str]) -> list[str]:
    """Extract tool name list from a server or raw list."""
    if isinstance(source, list):
        return source
    return source.call_names


def assert_called_before(source: MockMcpServer | list[str], first: str, second: str) -> None:
    """Assert first tool was called before second tool.

    >>> assert_called_before(["create", "check", "merge"], "create", "merge")  # passes
    >>> assert_called_before(["merge", "create"], "create", "merge")  # fails
    """
    names = _get_names(source)
    first_idx = next((i for i, n in enumerate(names) if n == first), None)
    second_idx = next((i for i, n in enumerate(names) if n == second), None)

    if first_idx is None:
        raise AssertionError(f"'{first}' was never called. Call log: {names}")
    if second_idx is None:
        raise AssertionError(f"'{second}' was never called. Call log: {names}")
    if first_idx >= second_idx:
        raise AssertionError(
            f"Expected '{first}' (index {first_idx}) before '{second}' (index {second_idx}). "
            f"Call log: {names}"
        )


def assert_called_between(
    source: MockMcpServer | list[str], before: str, middle: str, after: str
) -> None:
    """Assert middle tool was called between before and after.

    >>> assert_called_between(["create", "check", "merge"], "create", "check", "merge")  # passes
    """
    names = _get_names(source)
    before_idx = next((i for i, n in enumerate(names) if n == before), None)
    middle_idx = next((i for i, n in enumerate(names) if n == middle), None)
    after_idx = next((i for i, n in enumerate(names) if n == after), None)

    for name, idx in [(before, before_idx), (middle, middle_idx), (after, after_idx)]:
        if idx is None:
            raise AssertionError(f"'{name}' was never called. Call log: {names}")

    if not (before_idx < middle_idx < after_idx):
        raise AssertionError(
            f"Expected '{middle}' between '{before}' (idx {before_idx}) and '{after}' (idx {after_idx}), "
            f"but '{middle}' was at index {middle_idx}. Call log: {names}"
        )


def assert_call_count(source: MockMcpServer | list[str], tool_name: str, expected: int) -> None:
    """Assert a tool was called exactly N times."""
    names = _get_names(source)
    actual = names.count(tool_name)
    if actual != expected:
        raise AssertionError(
            f"Expected '{tool_name}' called {expected} times, got {actual}. Call log: {names}"
        )


def assert_call_sequence(source: MockMcpServer | list[str], expected_sequence: list[str]) -> None:
    """Assert tools were called in this exact subsequence order.

    Does NOT require the sequence to be contiguous — other calls can appear between.

    >>> assert_call_sequence(["a", "x", "b", "y", "c"], ["a", "b", "c"])  # passes
    >>> assert_call_sequence(["c", "b", "a"], ["a", "b", "c"])  # fails
    """
    names = _get_names(source)
    pos = 0
    for tool in expected_sequence:
        found = False
        while pos < len(names):
            if names[pos] == tool:
                found = True
                pos += 1
                break
            pos += 1
        if not found:
            raise AssertionError(
                f"Expected sequence {expected_sequence} but '{tool}' not found after position {pos}. "
                f"Call log: {names}"
            )


def assert_never_called(source: MockMcpServer | list[str], tool_name: str) -> None:
    """Assert a tool was never called."""
    names = _get_names(source)
    if tool_name in names:
        raise AssertionError(f"Expected '{tool_name}' to never be called, but it was. Call log: {names}")
