"""Sanitized S42 missing-chunk structures: bullets separated by Markdown rules."""

import pytest

from brainlayer.watcher_bridge import _is_pure_deletion_diff


@pytest.mark.parametrize(
    "text",
    [
        "- **Sample outcomes:** 12 queued, 2 late.\n\n---\n\nResults are provisional.",
        "- **superseded:** a later message replaced the request.\n  - **Acknowledged:** recorded.\n---",
        "### Ledger notes\n- Name the platforms you built.\n---\nKeep the evidence.",
        "### Call questions: what to drop\n- Fleet ownership was answered and filed.\n---",
        "PR is open and not merged.\n---\n- Protocol support still needs verification.",
    ],
)
def test_s42_prose_is_retained(text):
    assert not _is_pure_deletion_diff(text)


@pytest.mark.parametrize(
    "text",
    [
        "--- a/file\n+++ b/file\n@@ -1,2 +0,0 @@\n-old\n-code",
        "diff --git a/file b/file\ndeleted file mode 100644\nindex abc..000\n--- a/file\n+++ /dev/null\n@@ -1 +0,0 @@\n-old",
    ],
)
def test_deletion_patch_is_filtered(text):
    assert _is_pure_deletion_diff(text)


@pytest.mark.parametrize("tail", ["\nExplanation survives.", "\n+new", "\n context"])
def test_patch_with_surviving_content_is_retained(tail):
    assert not _is_pure_deletion_diff("--- a/file\n+++ b/file\n@@ -1 +0,0 @@\n-old" + tail)
