from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.provenance_integration import ProvenanceConflictReport

runner = CliRunner()


class _DummyStore:
    def close(self) -> None:
        pass


def test_provenance_confirm_exits_nonzero_when_no_confirmation_consumed(monkeypatch):
    import brainlayer.cli as cli
    import brainlayer.provenance_integration as provenance_integration

    monkeypatch.setattr(cli, "_open_vector_store_for_cli", lambda: _DummyStore())
    monkeypatch.setattr(
        provenance_integration,
        "confirm_pending",
        lambda _store, _claim_id: ProvenanceConflictReport(
            entity="",
            entity_ids=[],
            resolutions={},
            dry_run=False,
            notes=["Ambiguous pending provenance confirmation matched claim_id=c-shared-infer"],
        ),
    )

    result = runner.invoke(app, ["provenance", "confirm", "c-shared-infer"])

    assert result.exit_code == 1
    assert "Ambiguous pending provenance confirmation matched claim_id=c-shared-infer" in result.output
    assert "confirmed=c-shared-infer" not in result.output
