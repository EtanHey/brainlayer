def test_read_only_alarm_keeps_fatal_decision_without_persistence(tmp_path):
    import pytest

    from brainlayer import t3_linkage_alarm
    from brainlayer.alarm import BrainLayerAlarm
    from brainlayer.t3_provenance import codex_header_identity

    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with t3_linkage_alarm.read_only():
        with pytest.raises(BrainLayerAlarm) as pending:
            t3_linkage_alarm.observe(tmp_path / "missing.jsonl", "fixture", [], pending=True)
        assert pending.value.code == "t3_runtime_linkage_pending"
        with pytest.raises(BrainLayerAlarm) as invalid:
            codex_header_identity(
                tmp_path / "missing.jsonl", header=({"type": "session_meta", "payload": {"id": "invalid"}}, 100)
            )
        assert invalid.value.code == "t3_session_identity_invalid"
    assert not t3_linkage_alarm.read_only_active()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
