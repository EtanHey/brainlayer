"""Tests for realtime enrichment defaults shared across entrypoints."""

import importlib


def test_invalid_realtime_enrich_since_hours_env_falls_back(monkeypatch):
    import brainlayer.cli as cli
    import brainlayer.config as config
    import brainlayer.mcp as mcp
    import brainlayer.mcp.enrich_handler as enrich_handler

    monkeypatch.setenv("BRAINLAYER_DEFAULT_ENRICH_SINCE_HOURS", "24h")
    try:
        importlib.reload(config)
        cli = importlib.reload(cli)
        mcp = importlib.reload(mcp)
        enrich_handler = importlib.reload(enrich_handler)

        assert config.DEFAULT_REALTIME_ENRICH_SINCE_HOURS == 8760
        assert cli.DEFAULT_REALTIME_ENRICH_SINCE_HOURS == 8760
        assert mcp.DEFAULT_REALTIME_ENRICH_SINCE_HOURS == 8760
        assert enrich_handler.DEFAULT_REALTIME_ENRICH_SINCE_HOURS == 8760
    finally:
        monkeypatch.delenv("BRAINLAYER_DEFAULT_ENRICH_SINCE_HOURS", raising=False)
        importlib.reload(config)
        importlib.reload(cli)
        importlib.reload(mcp)
        importlib.reload(enrich_handler)
