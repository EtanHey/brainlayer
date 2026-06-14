"""Reusable isolation-proof harness for Happy Camper scoping."""

from brainlayer.isolation_proof import (
    BASIC_PROOF_EXPECTATIONS,
    EXTENSION_PROOF_EXPECTATIONS,
    ScopeProbe,
    run_basic_isolation_proof,
    run_full_isolation_proof,
    seed_isolation_proof_db,
)


def test_seeded_isolation_proof_db_exercises_basic_split(tmp_path):
    db_path = tmp_path / "happy-camper-isolation-proof.db"

    fixture = seed_isolation_proof_db(db_path)
    report = run_basic_isolation_proof(
        fixture.db_path,
        probes=[
            ScopeProbe(name="worker-repo-a", consumer="worker", project="repo-a"),
            ScopeProbe(name="orchestrator", consumer="orchestrator", project=None, include_checkpoints=True),
            ScopeProbe(name="coach", consumer="coach", project=None),
        ],
    )

    assert fixture.seeded_ids == {
        "repo-a-main-proof",
        "repo-a-worktree-proof",
        "repo-a-checkpoint-proof",
        "repo-b-main-proof",
        "repo-b-worktree-proof",
        "repo-b-checkpoint-proof",
        "personal-checkpoint-proof",
        "null-user-local-proof",
    }
    assert report.visible_ids_by_probe == BASIC_PROOF_EXPECTATIONS
    assert report.failures == []


def test_full_isolation_proof_db_exercises_extension_roles(tmp_path):
    db_path = tmp_path / "happy-camper-full-isolation-proof.db"

    fixture = seed_isolation_proof_db(db_path)
    report = run_full_isolation_proof(fixture.db_path)

    assert report.visible_ids_by_probe == {
        **BASIC_PROOF_EXPECTATIONS,
        **EXTENSION_PROOF_EXPECTATIONS,
    }
    assert report.failures == []
