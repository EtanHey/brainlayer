"""Tests for realtime enrichment defaults shared across entrypoints."""

import importlib


def test_invalid_realtime_enrich_since_hours_env_falls_back(monkeypatch):
    import brainlayer.cli as cli
    import brainlayer.config as config

    monkeypatch.setenv("BRAINLAYER_DEFAULT_ENRICH_SINCE_HOURS", "24h")
    try:
        importlib.reload(config)
        cli = importlib.reload(cli)

        assert config.DEFAULT_REALTIME_ENRICH_SINCE_HOURS == 8760
        assert cli.DEFAULT_REALTIME_ENRICH_SINCE_HOURS == 8760
    finally:
        monkeypatch.delenv("BRAINLAYER_DEFAULT_ENRICH_SINCE_HOURS", raising=False)
        importlib.reload(config)
        importlib.reload(cli)
