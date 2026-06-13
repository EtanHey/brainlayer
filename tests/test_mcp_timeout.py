from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_with_timeout_reads_timeout_budget_at_call_time(monkeypatch):
    from brainlayer import mcp

    monkeypatch.setattr(mcp, "_mcp_query_timeout", lambda: 0.001)

    result = await asyncio.wait_for(mcp._with_timeout(asyncio.sleep(10)), timeout=0.1)

    assert result.isError is True
    assert "BrainLayer timeout (0.001s)" in result.content[0].text
