"""Regression tests for correction-detection false-fire gates."""

from __future__ import annotations

import pytest

from brainlayer.pipeline.correction_detection import (
    build_correction_tags,
    detect_correction,
    looks_like_live_correction,
)

FALSE_FIRE_PROMPTS = [
    pytest.param(
        "[orc gen-12 -> brainlayerCodex] No need to escalate; the worker is still running.",
        id="agent-relay-header",
    ),
    pytest.param(
        "FLEET TICK (gen-12 orc, R5/R6.5/R8)\nCodex s:13 working, no PR yet; do not narrate and do not stop the loop.",
        id="fleet-tick-cron",
    ),
    pytest.param(
        "<task-notification>\n"
        "<summary>Worker completed</summary>\n"
        '<result>Historical Etan quote: "No, use the launcher."</result>\n'
        "</task-notification>",
        id="task-notification",
    ),
    pytest.param(
        "<task-notification>\n"
        "<summary>Worker completed</summary>\n"
        "<result>No, the watcher should skip this quoted worker result.</result>\n"
        "</task-notification>",
        id="wrapped-task-notification-correction-looking-line",
    ),
    pytest.param(
        "Worker Brief - Correction-Detection Gate Port\n"
        "Mission: port the gate. No PR exists yet; do not use the real DB in tests.",
        id="worker-dispatch-brief",
    ),
    pytest.param(
        "Worker Brief\nQuoted in dispatch:\nNo, use the launcher.",
        id="quoted-dispatch-correction",
    ),
    pytest.param(
        "Worker Brief\nQuoted in dispatch:\nI told you, no -- use the launcher.",
        id="quoted-dispatch-strong-cue",
    ),
    pytest.param(
        "No PR exists yet.\n"
        "Worker Brief - Correction-Detection Gate Port\n"
        "Dispatch this to a worker; do not use the real DB in tests.",
        id="leading-no-worker-brief",
    ),
    pytest.param(
        "No --print/-p mode ever.\nStanding rules block for worker dispatch; do not use source ~/.zshrc.",
        id="leading-no-standing-rules",
    ),
    pytest.param(
        "You are the NEW brainlayerClaude-LEAD. Read the outgoing lead's grill answer: "
        "run nohup ./scripts/worker.sh > /tmp/worker.log 2>&1 &.",
        id="boot-grill-answer-with-nohup",
    ),
    pytest.param(
        "The note now describes normal startup behavior and should not be tagged.",
        id="no-substrings",
    ),
    pytest.param("The stopper process stayed healthy during the run.", id="stop-substring"),
    pytest.param("Nevermind the previous timing estimate; this is neutral.", id="never-substring"),
    pytest.param("Wrongness scoring belongs to the eval, not the hook.", id="wrong-substring"),
    pytest.param("The answer was incorrectly routed by a different tool.", id="incorrect-substring"),
    pytest.param("Formatters and stylelint ran before the tonearm calibration note.", id="style-substrings"),
]


@pytest.mark.parametrize("prompt", FALSE_FIRE_PROMPTS)
def test_detect_correction_suppresses_non_user_payloads(prompt: str):
    assert detect_correction(prompt) is None


@pytest.mark.parametrize("prompt", FALSE_FIRE_PROMPTS)
def test_build_correction_tags_suppresses_non_user_payloads(prompt: str):
    assert build_correction_tags(prompt) == []


@pytest.mark.parametrize(
    ("prompt", "category"),
    [
        ("No - I told you, use the launcher", "factual"),
        ("that's wrong, the DB is at ~/.local/share", "factual"),
        ("stop. you keep doing this", "preference"),
        ("Worker Brief\nNo, that's wrong - use the launcher", "factual"),
        ("I have no understanding of no --print; that's wrong, use the launcher", "factual"),
        ("Please fix the formatting", "style"),
        ("Please change the styling", "style"),
        ("Some context.\nNo, that's wrong - use the launcher", "factual"),
        ("No, [Entity: Avi Simon -- person] is wrong; Avi works at Lightricks.", "factual"),
        ("No, the <task-notification> watcher is wrong; do not store those chunks.", "factual"),
        ("Some context\nNo, the <task-notification> watcher is wrong; do not store those chunks.", "factual"),
        ('No, "commandMode": "task-notification" is wrong; keep user corrections.', "factual"),
        ("<task-notification> is wrong; do not treat this user correction as a wrapper.", "factual"),
        ("[old -> new] is wrong; it should be old -> newer", "factual"),
        ("No --print/-p standing rule is wrong; allow -p for debug.", "factual"),
        ("No --print/-p standing rule\nis wrong; allow -p for debug.", "factual"),
        ("לא נכון", "factual"),
    ],
)
def test_detect_correction_keeps_true_positives(prompt: str, category: str):
    assert detect_correction(prompt) == category


def test_live_correction_cues_match_multiline_direct_correction():
    assert looks_like_live_correction("Some context.\nNo, that's wrong - use the launcher")
