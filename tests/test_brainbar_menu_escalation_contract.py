"""Keep R31 app escalation decisions at their single production source."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "brain-bar/Sources/BrainBar"


def function(source, name):
    start = source.index("{", source.index("func " + name + "("))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def assert_one_escalation_source(presenter, sentences):
    outside = presenter.replace(function(presenter, "appEscalationCauses"), "")
    outside = outside.replace(function(outside, "lagPastLimit"), "")
    assert not re.search(r"maintenanceNightFailures[^\n]*>=\s*3|==\s*\.escalated|>\s*1800", outside + sentences)
    for name in ("classify", "finalGuard", "appFacts"):
        assert "appEscalationCauses(c)" in function(presenter, name)
    assert "appEscalationCauses(c)" in sentences.split("static let checkSoon", 1)[1].split("static let checkingMemory", 1)[0]


def test_escalation_policy_has_one_source_and_rejects_a_fourth_site():
    presenter = (ROOT / "BrainBarMenuPresenter.swift").read_text()
    sentences = (ROOT / "BrainBarMenuSentences.swift").read_text()
    assert_one_escalation_source(presenter, sentences)
    for decision in ("c.maintenanceNightFailures.map { $0 >= 3 }", "ingestLagDisposition(c) == .escalated", "age > 1800"):
        with pytest.raises(AssertionError):
            assert_one_escalation_source(presenter + "\nfunc fourthSite() { " + decision + " }", sentences)
