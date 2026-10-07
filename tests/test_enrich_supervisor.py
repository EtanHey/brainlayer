import pytest


def _result(*, attempted: int, enriched: int = 0, skipped: int = 0, failed: int = 0, errors=None):
    from brainlayer.enrichment_controller import EnrichmentResult

    return EnrichmentResult(
        mode="realtime",
        attempted=attempted,
        enriched=enriched,
        skipped=skipped,
        failed=failed,
        errors=list(errors or []),
    )


@pytest.fixture(autouse=True)
def _isolate_live_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
