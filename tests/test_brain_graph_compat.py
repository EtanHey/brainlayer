"""brain-export umap/sklearn compatibility shim.

Regression guard for the env where scikit-learn is pinned <1.6 (e.g. by
category-encoders) while umap-learn calls check_array(ensure_all_finite=...),
which only exists in sklearn>=1.6 — crashing brain-export for >=4096 nodes.
"""

import numpy as np
import pytest

from brainlayer.pipeline.brain_graph import _ensure_umap_sklearn_compat, compute_layout


def test_check_array_tolerates_ensure_all_finite_after_shim():
    import sklearn.utils as sk_utils

    _ensure_umap_sklearn_compat()
    # After the shim, check_array must accept the >=1.6 kwarg regardless of the
    # installed sklearn version (native on >=1.6, translated on <1.6).
    arr = sk_utils.check_array([[1.0, 2.0], [3.0, 4.0]], ensure_all_finite=True)
    assert arr.shape == (2, 2)


def test_shim_is_idempotent():
    _ensure_umap_sklearn_compat()
    _ensure_umap_sklearn_compat()  # second call must be a no-op, not double-wrap
    import sklearn.utils as sk_utils

    sk_utils.check_array([[1.0]], ensure_all_finite=True)  # still works


def test_compute_layout_runs_under_pinned_sklearn():
    pytest.importorskip("umap")
    rng = np.random.default_rng(0)
    sessions = [{"embedding": list(rng.random(16))} for _ in range(120)]
    coords = compute_layout(sessions)
    assert coords.shape == (120, 3)
    # normalized into the Three.js [-50, 50] box
    assert coords.min() >= -50.0001, f"min out of bounds: {coords.min()}"
    assert coords.max() <= 50.0001, f"max out of bounds: {coords.max()}"
