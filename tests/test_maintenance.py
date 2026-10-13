import pytest

import tests.test_service_supervisor as native_tests
from tests.test_maintenance_control import control, native  # noqa: F401


@pytest.mark.parametrize("startup_delay", [0, 3.2])
def test_existing_maintenance_uses_component_hold_without_legacy_mutation(request, monkeypatch, startup_delay):
    from brainlayer import maintenance

    # Delay before the guard is armed; the fixture must wait for startup, not a fixed sleep.
    monkeypatch.setattr(
        native_tests,
        "BOOT",
        native_tests.BOOT.replace(
            "p=Path(sys.argv[1])", f"import time; time.sleep({startup_delay})\np=Path(sys.argv[1])"
        ),
    )
    data = request.getfixturevalue("control")
    monkeypatch.setattr(maintenance, "run_command", lambda *_a, **_kw: pytest.fail("legacy mutation escaped"))
    held = {}
    assert maintenance._service_is_loaded("index") is False
    try:
        maintenance._quiesce_services(("watch", "index", "drain"), held)
        assert all(v["pid"] is None for v in data[2]()["components"].values())
        # ACK must already mean every original group (including descendants) is gone.
        assert any(v["pid"] for v in data[3].values())
        assert all(not native_tests.exists(v["pid"]) for v in data[3].values() if v["pid"])
    finally:
        assert maintenance._resume_services(data[0], ("watch", "index", "drain"), held) == []
