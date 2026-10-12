import pytest

from tests.test_maintenance_control import control, native  # noqa: F401


def test_existing_maintenance_uses_component_hold_without_legacy_mutation(request, monkeypatch):
    from brainlayer import maintenance

    data = request.getfixturevalue("control")
    monkeypatch.setattr(maintenance, "run_command", lambda *_a, **_kw: pytest.fail("legacy mutation escaped"))
    held = {}
    assert maintenance._service_is_loaded("index") is False
    try:
        maintenance._quiesce_services(("watch", "index", "drain"), held)
        assert all(v["pid"] is None for v in data[2]()["components"].values())
    finally:
        assert maintenance._resume_services(data[0], ("watch", "index", "drain"), held) == []
