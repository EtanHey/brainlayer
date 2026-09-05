"""UserPromptSubmit hook: injection cap, skip gates, and fail-open behaviour.

Sized against a 30-day transcript window (4,759 hook fires paired with the prompt
that produced them). The numbers those rules were chosen from are recorded in the
PR body; the contracts they must keep are recorded here.
"""

import importlib.util
import io
import json
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def load_hook_module():
    spec = importlib.util.spec_from_file_location(
        "brainlayer_prompt_search_cap",
        HOOKS_DIR / "brainlayer-prompt-search.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hook():
    return load_hook_module()


class TestInjectionCap:
    """Injected output is bounded, and bounded at a line boundary."""

    def test_output_under_cap_is_untouched(self, hook):
        lines = ["BrainLayer memory available.", "- [2026-09-01] a short result"]
        assert hook.cap_injection(lines) == lines

    def test_empty_input_returns_empty(self, hook):
        assert hook.cap_injection([]) == []

    def test_output_over_cap_is_held_to_the_cap(self, hook):
        lines = [f"- [2026-09-01] {'x' * 100}" for _ in range(20)]
        capped = hook.cap_injection(lines)
        assert len("\n".join(capped)) <= hook.MAX_INJECTION_CHARS

    def test_cap_never_splits_a_line(self, hook):
        lines = [f"- [2026-09-01] result number {i} {'x' * 90}" for i in range(20)]
        capped = hook.cap_injection(lines)
        for line in capped[:-1]:
            assert line in lines

    def test_cap_appends_a_pointer_naming_the_dropped_count(self, hook):
        lines = [f"- [2026-09-01] {'x' * 100}" for _ in range(20)]
        capped = hook.cap_injection(lines)
        dropped = len(lines) - (len(capped) - 1)
        assert capped[-1] == f"[+{dropped} more in BrainLayer -- use brain_search for the rest]"

    def test_no_pointer_is_added_when_nothing_was_dropped(self, hook):
        lines = ["BrainLayer memory available.", "- [2026-09-01] short"]
        assert "more in BrainLayer" not in "\n".join(hook.cap_injection(lines))

    def test_first_line_survives_even_when_it_alone_exceeds_the_cap(self, hook):
        """A cap must never turn a search that found something into nothing."""
        lines = ["- [2026-09-01] " + "x" * 5000, "- [2026-09-02] second"]
        capped = hook.cap_injection(lines)
        assert capped[0] == lines[0]
        assert capped[-1].startswith("[+1 more in BrainLayer")

    def test_cap_is_configurable_per_call(self, hook):
        lines = ["aaaa", "bbbb", "cccc", "dddd"]
        capped = hook.cap_injection(lines, max_chars=70)
        assert len("\n".join(capped)) <= 70


class TestRelaySkip:
    """Machine relays are not prompts; the hook must not search on them."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "[report] changed — read /Users/etanheyman/.cmux/agents/golemsClaude-ff428de6/report.md",
            'Another Claude session sent a message:\n<cross-session-message from="uds:/tmp/x.sock">\norc, status?\n</cross-session-message>',
            '<cross-session-message from="uds:/tmp/x.sock">\nping\n</cross-session-message>',
            "<command-name>/feedback</command-name>\n<command-message>feedback</command-message>",
            "<local-command-stdout>(no content)</local-command-stdout>",
            "<system-reminder>budget note</system-reminder>",
        ],
    )
    def test_relay_shapes_are_skipped(self, hook, prompt):
        assert hook.is_operational_noise_prompt(prompt) is True

    def test_task_notification_is_still_skipped(self, hook):
        prompt = "<task-notification>\n<task-id>bmxiwdbvk</task-id>\n<tool-use-id>toolu_01GPTk</tool-use-id>\n</task-notification>"
        assert hook.is_operational_noise_prompt(prompt) is True

    @pytest.mark.parametrize(
        "prompt",
        [
            "How does the watcher handle a file shrink, and where is the offset stored?",
            "Explain the enrichment backend fallback order for BrainLayer.",
        ],
    )
    def test_real_prompts_are_not_skipped(self, hook, prompt):
        assert hook.is_operational_noise_prompt(prompt) is False

    def test_a_prompt_that_merely_discusses_relays_is_not_skipped(self, hook):
        """The marker must be an envelope, not a topic."""
        prompt = (
            "I want to change how we handle worker pings. Right now the lead gets a line and "
            "has to open the file itself, which is slow when six workers report at once. Could "
            "we instead inline the first paragraph? The shape I mean is the <cross-session-message "
            "envelope that another Claude session sends."
        )
        assert hook.is_operational_noise_prompt(prompt) is False


class TestShortPromptSkip:
    """A short prompt with no content words has nothing to search on."""

    @pytest.mark.parametrize("prompt", ["Nope.", "you forgot the -E", "answer me.", "you closable?"])
    def test_low_signal_short_prompts_are_skipped(self, hook, prompt):
        keywords = hook.extract_keywords(prompt)
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", keywords) is True

    def test_long_prompt_is_never_short_skipped(self, hook):
        prompt = "a " * 40
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", []) is False

    def test_short_prompt_with_real_keywords_is_kept(self, hook):
        prompt = "where does enrichment write the WAL checkpoint?"
        keywords = hook.extract_keywords(prompt)
        assert len(keywords) > hook.SHORT_PROMPT_MAX_KEYWORDS
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", keywords) is False

    @pytest.mark.parametrize("route", ["follow_up", "entity_lookup", "hebrew_query"])
    def test_narrow_routes_are_exempt(self, hook, route):
        """follow_up rewrites from session context; the other two are already narrow."""
        assert hook.is_low_signal_short_prompt("Nope.", route, []) is False

    def test_word_boundary_is_inclusive_at_the_limit(self, hook):
        prompt = " ".join(["word"] * hook.SHORT_PROMPT_MAX_WORDS)
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", []) is False


class TestFailOpen:
    """The hook must never block a prompt, whatever it is handed."""

    def _run(self, hook, monkeypatch, payload):
        monkeypatch.setattr(hook.sys, "stdin", io.StringIO(payload))
        monkeypatch.setattr(hook, "get_db_path", lambda: None)
        with pytest.raises(SystemExit) as exc:
            hook.main()
        return exc.value.code

    def test_malformed_json_exits_zero(self, hook, monkeypatch, capsys):
        assert self._run(hook, monkeypatch, "not json at all") == 0
        assert capsys.readouterr().out == ""

    def test_empty_stdin_exits_zero(self, hook, monkeypatch):
        assert self._run(hook, monkeypatch, "") == 0

    def test_missing_prompt_key_exits_zero(self, hook, monkeypatch, capsys):
        assert self._run(hook, monkeypatch, json.dumps({"session_id": "s1"})) == 0
        assert capsys.readouterr().out == ""

    def test_relay_payload_exits_zero_and_injects_nothing(self, hook, monkeypatch, capsys):
        payload = json.dumps({"prompt": "[report] changed — read /tmp/report.md", "session_id": ""})
        assert self._run(hook, monkeypatch, payload) == 0
        assert capsys.readouterr().out == ""

    def test_short_prompt_exits_zero_and_injects_nothing(self, hook, monkeypatch, capsys):
        payload = json.dumps({"prompt": "you closable?", "session_id": ""})
        assert self._run(hook, monkeypatch, payload) == 0
        assert capsys.readouterr().out == ""

    def test_short_prompt_still_reports_a_detected_correction(self, hook, monkeypatch, capsys):
        """Short prompts are where corrections live; the skip must not eat them."""
        monkeypatch.setattr(hook, "detect_correction", lambda prompt: "factual")
        payload = json.dumps({"prompt": "Nope.", "session_id": ""})
        assert self._run(hook, monkeypatch, payload) == 0
        out = capsys.readouterr().out
        assert "[Correction detected: factual]" in out
        assert "brain_store" in out


class TestImportCost:
    """The hook is on every prompt's critical path: keep heavy deps off module load.

    `from brainlayer.pipeline.correction_detection import detect_correction` at module
    scope pulled pipeline/__init__ -> semantic_style -> sklearn, measured at ~790ms
    on every fire. It is loaded lazily instead; this test is what stops it coming back.
    """

    # brainlayer.pipeline is on the list because importing it is what drags the rest
    # in: __init__ re-exports enrichment -> vector_store -> numpy/sqlite_vec, and
    # semantic_style -> sklearn. The hook must reach detect_correction lazily.
    HEAVY = (
        "sklearn",
        "torch",
        "sentence_transformers",
        "transformers",
        "numpy",
        "requests",
        "sqlite_vec",
        "brainlayer.pipeline",
    )

    def test_module_load_does_not_import_heavy_deps(self):
        import subprocess
        import sys

        repo = Path(__file__).parent.parent
        code = (
            "import importlib.util, sys, json\n"
            f"spec = importlib.util.spec_from_file_location('h', {str(HOOKS_DIR / 'brainlayer-prompt-search.py')!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"print(json.dumps([m for m in {list(TestImportCost.HEAVY)!r} if m in sys.modules]))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=repo,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "src")},
        )
        assert proc.returncode == 0, proc.stderr
        loaded = json.loads(proc.stdout.strip().splitlines()[-1])
        assert loaded == [], f"hook module load pulled heavy deps: {loaded}"

    def test_semantic_style_probes_sklearn_without_importing_it(self):
        """The same lazy contract, at the root: pipeline/__init__ is on the MCP startup path."""
        import subprocess
        import sys

        repo = Path(__file__).parent.parent
        code = (
            "import sys, json\n"
            "from brainlayer.pipeline import semantic_style\n"
            "print(json.dumps({'flag': semantic_style.HAS_SKLEARN, 'imported': 'sklearn' in sys.modules}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=repo,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "src")},
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["flag"] is True, "sklearn is installed here, so the probe must say so"
        assert result["imported"] is False, "sklearn must not be imported at module load"
