"""UserPromptSubmit hook: injection cap, skip gates, and fail-open behaviour.

Sized against a 30-day transcript window (4,759 hook fires paired with the prompt
that produced them). The numbers those rules were chosen from are recorded in the
PR body; the contracts they must keep are recorded here.

Test methods are `@staticmethod` because none of them uses the bound instance
(DeepSource PTC-W0049). Older test modules in this suite predate that and still
carry the plain-`self` shape.
"""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"
REPO_ROOT = Path(__file__).parent.parent


def load_hook_module():
    spec = importlib.util.spec_from_file_location(
        "brainlayer_prompt_search_cap",
        HOOKS_DIR / "brainlayer-prompt-search.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_probe(code):
    """Run `code` in a clean interpreter with only the repo's src on the path."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO_ROOT / "src"),
            # conftest arms this for every unmarked test and relies on subprocesses
            # inheriting it -- "a test that SPAWNS a re-embedding script loads a model
            # just as surely as one that imports it". An explicit env dict drops it, so
            # it is re-armed here by hand. It also strengthens these two probes: a model
            # loaded at hook-import time now trips the repo guard as well as the
            # assertion below.
            "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
        },
        # Explicit: the assert below reports the child's stderr, which is a far
        # more useful failure than CalledProcessError's bare exit code.
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture
def hook():
    return load_hook_module()


class TestInjectionCap:
    """Injected output is bounded, and bounded at a line boundary."""

    @staticmethod
    def test_output_under_cap_is_untouched(hook):
        lines = ["BrainLayer memory available.", "- [2026-09-01] a short result"]
        assert hook.cap_injection(lines) == (lines, 0)

    @staticmethod
    def test_empty_input_returns_empty(hook):
        assert hook.cap_injection([]) == ([], 0)

    @staticmethod
    def test_output_over_cap_is_held_to_the_cap(hook):
        lines = [f"- [2026-09-01] {'x' * 100}" for _ in range(20)]
        capped, dropped = hook.cap_injection(lines)
        assert len("\n".join(capped)) <= hook.MAX_INJECTION_CHARS
        assert dropped > 0

    @staticmethod
    def test_cap_never_splits_a_line(hook):
        lines = [f"- [2026-09-01] result number {i} {'x' * 90}" for i in range(20)]
        capped, _ = hook.cap_injection(lines)
        for line in capped[:-1]:
            assert line in lines

    @staticmethod
    def test_cap_appends_a_pointer_naming_the_dropped_count(hook):
        lines = [f"- [2026-09-01] {'x' * 100}" for _ in range(20)]
        capped, dropped = hook.cap_injection(lines)
        assert dropped == len(lines) - (len(capped) - 1)
        assert capped[-1] == f"[+{dropped} more in BrainLayer -- use brain_search for the rest]"

    @staticmethod
    def test_no_pointer_is_added_when_nothing_was_dropped(hook):
        lines = ["BrainLayer memory available.", "- [2026-09-01] short"]
        capped, dropped = hook.cap_injection(lines)
        assert dropped == 0
        assert "more in BrainLayer" not in "\n".join(capped)

    @staticmethod
    def test_first_line_survives_even_when_it_alone_exceeds_the_cap(hook):
        """A cap must never turn a search that found something into nothing."""
        lines = ["- [2026-09-01] " + "x" * 5000, "- [2026-09-02] second"]
        capped, dropped = hook.cap_injection(lines)
        assert capped[0] == lines[0]
        assert dropped == 1
        assert capped[-1].startswith("[+1 more in BrainLayer")

    @staticmethod
    def test_a_budget_filled_to_the_byte_still_respects_the_cap(hook):
        """Regression: a hand-set note reserve was one char short and let this reach 602."""
        lines = ["x" * 44] + ["y" * 49] * 30
        capped, _ = hook.cap_injection(lines)
        assert len("\n".join(capped)) <= hook.MAX_INJECTION_CHARS

    @staticmethod
    @pytest.mark.parametrize("n_lines", [2, 3, 7, 15, 40, 120])
    def test_cap_holds_across_line_counts(hook, n_lines):
        lines = [f"- [2026-09-01] line {i} " + "z" * 70 for i in range(n_lines)]
        capped, _ = hook.cap_injection(lines)
        assert len("\n".join(capped)) <= hook.MAX_INJECTION_CHARS

    @staticmethod
    def test_cap_is_configurable_per_call(hook):
        lines = ["aaaa", "bbbb", "cccc", "dddd"]
        capped, _ = hook.cap_injection(lines, max_chars=70)
        assert len("\n".join(capped)) <= 70


class TestCappedResultsAreNotRegisteredAsInjected:
    """A result the cap withheld must not be marked as injected.

    Registering it would put its chunk_id in the session dedup file, suppressing
    that chunk from every later prompt in the session -- a silent memory loss.
    Found by Macroscope on PR #782.
    """

    @staticmethod
    def test_dropped_count_lets_a_caller_identify_the_survivors(hook):
        header = "BrainLayer memory available -- use brain_search before answering domain questions."
        results = [f"- [2026-09-0{i}] " + "r" * 120 for i in range(1, 6)]
        lines = [header, *results]

        capped, dropped = hook.cap_injection(lines)

        result_line_start = 1  # index of the first result line in `lines`
        survived = max(0, (len(capped) - 1) - result_line_start)
        assert dropped > 0, "this fixture must overflow the cap for the test to mean anything"
        assert survived < len(results)
        # every surviving id maps to a line the agent actually received
        for shown in results[:survived]:
            assert shown in capped
        for withheld in results[survived:]:
            assert withheld not in capped

    @staticmethod
    def test_survivor_count_is_exact_when_nothing_is_dropped(hook):
        lines = ["header", "- [2026-09-01] one", "- [2026-09-02] two"]
        capped, dropped = hook.cap_injection(lines)
        assert dropped == 0
        assert max(0, (len(capped)) - 1) == 2


class TestRelaySkip:
    """Machine relays are not prompts; the hook must not search on them."""

    @staticmethod
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
    def test_relay_shapes_are_skipped(hook, prompt):
        assert hook.is_operational_noise_prompt(prompt) is True

    @staticmethod
    def test_task_notification_is_still_skipped(hook):
        prompt = "<task-notification>\n<task-id>bmxiwdbvk</task-id>\n<tool-use-id>toolu_01GPTk</tool-use-id>\n</task-notification>"
        assert hook.is_operational_noise_prompt(prompt) is True

    @staticmethod
    @pytest.mark.parametrize(
        "prompt",
        [
            "How does the watcher handle a file shrink, and where is the offset stored?",
            "Explain the enrichment backend fallback order for BrainLayer.",
        ],
    )
    def test_real_prompts_are_not_skipped(hook, prompt):
        assert hook.is_operational_noise_prompt(prompt) is False

    @staticmethod
    def test_a_prompt_that_merely_discusses_relays_is_not_skipped(hook):
        """The marker must be an envelope, not a topic."""
        prompt = (
            "I want to change how we handle worker pings. Right now the lead gets a line and "
            "has to open the file itself, which is slow when six workers report at once. Could "
            "we instead inline the first paragraph? The shape I mean is the <cross-session-message "
            "envelope that another Claude session sends."
        )
        assert hook.is_operational_noise_prompt(prompt) is False


class TestRelayNeedsEnvelopeStructure:
    """A prefix alone is not a relay -- the envelope structure must be there too.

    Found by Macroscope on PR #782: `startswith` alone skipped a prompt that
    opened by quoting one of these tokens while asking about it.
    """

    @staticmethod
    @pytest.mark.parametrize(
        "prompt",
        [
            "[report] is the prefix I mean — should we keep parsing it that way?",
            "<command-name is the tag cmux wraps slash commands in, right?",
            "<local-command-stdout without a closing tag is what I keep seeing, is that a bug?",
            "<system-reminder is that injected by the harness or by us?",
            "<cross-session-message is the envelope name I could not remember",
        ],
    )
    def test_a_bare_prefix_without_its_structure_is_not_a_relay(hook, prompt):
        assert hook.is_relay_envelope(prompt) is False
        assert hook.is_operational_noise_prompt(prompt) is False

    @staticmethod
    @pytest.mark.parametrize(
        "prompt",
        [
            "[report] changed — read /Users/etanheyman/.cmux/agents/x/report.md",
            "<command-name>/model</command-name>",
            "<local-command-stdout>(no content)</local-command-stdout>",
            "<system-reminder>note</system-reminder>",
            '<cross-session-message from="uds:/tmp/cc-socks/3515.sock">hi</cross-session-message>',
        ],
    )
    def test_the_real_shapes_still_read_as_relays(hook, prompt):
        assert hook.is_relay_envelope(prompt) is True

    @staticmethod
    def test_a_report_ping_without_a_path_is_not_a_relay(hook):
        assert hook.is_relay_envelope("[report] changed, take a look when you can") is False

    @staticmethod
    def test_cross_session_tag_without_a_from_attribute_is_not_a_relay(hook):
        assert hook.is_relay_envelope("<cross-session-message>orphan</cross-session-message>") is False


class TestShortPromptSkip:
    """A short prompt with no content words has nothing to search on."""

    @staticmethod
    @pytest.mark.parametrize("prompt", ["Nope.", "you forgot the -E", "answer me.", "you closable?"])
    def test_low_signal_short_prompts_are_skipped(hook, prompt):
        keywords = hook.extract_keywords(prompt)
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", keywords) is True

    @staticmethod
    def test_long_prompt_is_never_short_skipped(hook):
        prompt = "a " * 40
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", []) is False

    @staticmethod
    def test_short_prompt_with_real_keywords_is_kept(hook):
        prompt = "where does enrichment write the WAL checkpoint?"
        keywords = hook.extract_keywords(prompt)
        assert len(keywords) > hook.SHORT_PROMPT_MAX_KEYWORDS
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", keywords) is False

    @staticmethod
    @pytest.mark.parametrize("route", ["follow_up", "entity_lookup", "hebrew_query"])
    def test_narrow_routes_are_exempt(hook, route):
        """follow_up rewrites from session context; the other two are already narrow."""
        assert hook.is_low_signal_short_prompt("Nope.", route, []) is False

    @staticmethod
    def test_word_boundary_is_inclusive_at_the_limit(hook):
        prompt = " ".join(["word"] * hook.SHORT_PROMPT_MAX_WORDS)
        assert hook.is_low_signal_short_prompt(prompt, "knowledge_question", []) is False


class TestShortPromptGateRunsAfterEntityDetection:
    """The entity_lookup exemption has to be reachable to mean anything.

    Found by Macroscope on PR #782: the gate first ran where `classify_prompt`
    had not yet seen any entities, so it could never return `entity_lookup` --
    the exemption was dead code and a one-word entity question ("Etan?") was
    skipped instead of answered.
    """

    @staticmethod
    def test_classify_prompt_cannot_reach_entity_lookup_without_entities():
        from brainlayer.classify import classify_prompt

        assert classify_prompt("Etan?") != "entity_lookup"
        assert classify_prompt("Etan?", detected_entities=[{"name": "Etan"}]) == "entity_lookup"

    @staticmethod
    def test_a_short_entity_prompt_is_exempt_once_entities_are_known(hook):
        """The shape the dead exemption used to drop."""
        from brainlayer.classify import classify_prompt

        prompt = "Etan?"
        with_entities = classify_prompt(prompt, detected_entities=[{"name": "Etan"}])
        assert hook.is_low_signal_short_prompt(prompt, with_entities, hook.extract_keywords(prompt)) is False

    @staticmethod
    def test_the_gate_is_wired_after_entity_reclassification():
        """Order is the whole fix, so assert on order, not just on the predicate."""
        source = (HOOKS_DIR / "brainlayer-prompt-search.py").read_text()
        reclassify = source.index("classify_prompt(prompt, detected_entities=detected_entities)")
        gate = source.index("if is_low_signal_short_prompt(prompt, classification, extract_keywords(prompt)):")
        assert reclassify < gate, "the short-prompt gate must run after entity reclassification"


class TestSearchDeadlineExcludesDeferredImports:
    """A deferred import must not eat the search budget.

    Found by Macroscope on PR #782: `start` is captured inside main(), so moving
    the ~105ms pipeline import from module scope into main() charged it to
    DEADLINE_MS, where a slow first import could silently suppress retrieval.
    """

    @staticmethod
    def test_lazy_import_ms_starts_at_zero(hook):
        assert hook.lazy_import_ms() == 0.0

    @staticmethod
    def test_deferred_import_time_is_subtracted_from_the_search_budget(hook):
        start = hook.time.monotonic()
        wall = hook.elapsed_ms(start)
        hook._LAZY["import_ms"] = 400.0
        assert hook.search_elapsed_ms(start) < wall + 400.0
        assert hook.search_elapsed_ms(start) == pytest.approx(hook.elapsed_ms(start) - 400.0, abs=5)

    @staticmethod
    def test_a_slow_import_cannot_exhaust_the_deadline(hook):
        """The whole point: import cost must not push the budget past DEADLINE_MS."""
        start = hook.time.monotonic()
        hook._LAZY["import_ms"] = hook.DEADLINE_MS * 3
        assert hook.search_elapsed_ms(start) < hook.DEADLINE_MS

    @staticmethod
    def test_calling_detect_correction_records_its_import_cost(hook):
        assert hook.lazy_import_ms() == 0.0
        hook.detect_correction("no, that is wrong")
        assert hook.lazy_import_ms() > 0.0
        first = hook.lazy_import_ms()
        hook.detect_correction("no, that is wrong again")
        assert hook.lazy_import_ms() == first, "the import is memoised, so it is charged once"


class TestFailOpen:
    """The hook must never block a prompt, whatever it is handed."""

    @staticmethod
    def _run(hook, monkeypatch, payload):
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

    def test_no_db_exits_zero_on_a_short_prompt(self, hook, monkeypatch):
        """With no DB the hook degrades before the gate; it must still exit 0."""
        payload = json.dumps({"prompt": "you closable?", "session_id": ""})
        assert self._run(hook, monkeypatch, payload) == 0


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

    @staticmethod
    def test_module_load_does_not_import_heavy_deps():
        hook_path = HOOKS_DIR / "brainlayer-prompt-search.py"
        loaded = run_probe(
            "import importlib.util, sys, json\n"
            f"spec = importlib.util.spec_from_file_location('h', {str(hook_path)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"print(json.dumps([m for m in {list(TestImportCost.HEAVY)!r} if m in sys.modules]))\n"
        )
        assert loaded == [], f"hook module load pulled heavy deps: {loaded}"

    @staticmethod
    def test_semantic_style_probes_sklearn_without_importing_it():
        """The same lazy contract, at the root: pipeline/__init__ is on the MCP startup path."""
        result = run_probe(
            "import sys, json\n"
            "from brainlayer.pipeline import semantic_style\n"
            "print(json.dumps({'flag': semantic_style.HAS_SKLEARN, 'imported': 'sklearn' in sys.modules}))\n"
        )
        assert result["flag"] is True, "sklearn is installed here, so the probe must say so"
        assert result["imported"] is False, "sklearn must not be imported at module load"
