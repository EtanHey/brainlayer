"""`brainlayer dashboard` must fail loudly when the dashboard package is absent.

The wheel excludes src/brainlayer/dashboard, so every installed BrainLayer hits the
ImportError branch. It used to print advice and exit 0.
"""

from __future__ import annotations

import builtins

from typer.testing import CliRunner

from brainlayer.cli import app


def test_dashboard_missing_package_exits_nonzero(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (name == "dashboard" and level == 2) or name == "brainlayer.dashboard":
            raise ImportError("No module named 'brainlayer.dashboard'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = CliRunner().invoke(app, ["dashboard"])

    assert result.exit_code == 1, result.output
    assert "Dashboard unavailable" in result.output
    assert "source checkout" in result.output
