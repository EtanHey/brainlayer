"""Deepchecks regression for stale-index embedding drift."""

from __future__ import annotations

import pandas as pd
import pytest

try:
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import FeatureDrift
except (ImportError, ValueError) as exc:  # pragma: no cover - optional drift tooling compatibility
    pytest.skip(f"Deepchecks unavailable or incompatible: {exc}", allow_module_level=True)

try:
    import httpx
except ImportError:  # pragma: no cover - optional transitive dependency
    httpx = None

try:
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
except ImportError:  # pragma: no cover - fixture test dependencies provide this in CI
    HfHubHTTPError = None
    LocalEntryNotFoundError = None

try:
    from huggingface_hub.errors import OfflineModeIsEnabled
except ImportError:  # pragma: no cover - older huggingface_hub releases may omit this
    OfflineModeIsEnabled = None

from tests.regression._stale_index_fixture import (
    baseline_embedding_rows,
    cosine_similarity,
    current_embedding_rows,
    load_fixture,
)

HF_MODEL_UNAVAILABLE_SKIP_REASON = "HF model unavailable in CI (429/offline) — infra, not a regression"


def _embedding_frame(rows: list[list[float]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("embedding fixture rows must be non-empty")
    width = len(rows[0])
    columns = [f"dim_{index:04d}" for index in range(width)]
    return pd.DataFrame(rows, columns=columns)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    seen: set[int] = set()
    pending = [exc]
    chain = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if not current.__suppress_context__ and current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)

    return chain


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    status_code = getattr(exc, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _is_hf_model_unavailable_error(exc: BaseException) -> bool:
    for chained in _exception_chain(exc):
        if httpx is not None and isinstance(chained, httpx.HTTPStatusError) and _status_code(chained) == 429:
            return True
        if HfHubHTTPError is not None and isinstance(chained, HfHubHTTPError) and _status_code(chained) == 429:
            return True
        if LocalEntryNotFoundError is not None and isinstance(chained, LocalEntryNotFoundError):
            return True
        if OfflineModeIsEnabled is not None and isinstance(chained, OfflineModeIsEnabled):
            return True
    return False


def _current_embedding_rows_or_skip() -> list[list[float]]:
    try:
        return current_embedding_rows()
    except Exception as exc:
        if _is_hf_model_unavailable_error(exc):
            pytest.skip(HF_MODEL_UNAVAILABLE_SKIP_REASON)
        raise


def test_current_embedding_rows_or_skip_skips_hf_429(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("HEAD", "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/config.json")
    response = httpx.Response(429, request=request)

    def raise_hf_429() -> list[list[float]]:
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr("tests.regression.test_drift_detection.current_embedding_rows", raise_hf_429)

    with pytest.raises(pytest.skip.Exception, match="HF model unavailable"):
        _current_embedding_rows_or_skip()


def test_current_embedding_rows_or_skip_reraises_non_hf_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_real_failure() -> list[list[float]]:
        raise ValueError("embedding math regressed")

    monkeypatch.setattr("tests.regression.test_drift_detection.current_embedding_rows", raise_real_failure)

    with pytest.raises(ValueError, match="embedding math regressed"):
        _current_embedding_rows_or_skip()


def test_fixture_embeddings_pass_deepchecks_and_cosine_threshold() -> None:
    fixture = load_fixture()
    baseline_rows = baseline_embedding_rows()
    current_rows = _current_embedding_rows_or_skip()
    min_cosine_similarity = fixture["sample_text"]["min_cosine_similarity"]

    assert len(baseline_rows) == len(current_rows)
    for baseline_row, current_row in zip(baseline_rows, current_rows):
        assert cosine_similarity(current_row, baseline_row) > min_cosine_similarity

    baseline_frame = _embedding_frame(baseline_rows)
    current_frame = _embedding_frame(current_rows)
    drift_check = FeatureDrift(min_samples=len(baseline_rows))
    # With five rows, Deepchecks' KS-based numeric drift bottoms out around 0.2
    # even when the distributions are effectively unchanged across platforms.
    drift_check.add_condition_drift_score_less_than(max_allowed_numeric_score=0.21)
    result = drift_check.run(
        train_dataset=Dataset(baseline_frame, cat_features=[]),
        test_dataset=Dataset(current_frame, cat_features=[]),
        with_display=False,
    )

    assert result.passed_conditions()
