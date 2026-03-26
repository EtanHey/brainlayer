"""Behavioral tests for PR workflow using mock MCP servers.

Tests that the correct tool call sequence is followed in PR workflows.
The key behavioral contract: create PR → check reviews → merge.
Never merge before review completes.

Run: pytest tests/test_behavioral_pr_loop.py -v
"""

import json

import pytest

from tests.mock_mcp import (
    assert_call_count,
    assert_call_sequence,
    assert_called_before,
    assert_called_between,
    assert_never_called,
)
from tests.mock_mcp.mock_brainlayer import MockBrainLayer
from tests.mock_mcp.mock_github import MockGitHub
from tests.mock_mcp.mock_voicelayer import MockVoiceLayer

# =============================================================================
# Test: Mock MCP infrastructure works
# =============================================================================


class TestMockGitHubInfra:
    """Verify mock-github server registers tools and captures calls."""

    @pytest.mark.asyncio
    async def test_server_lists_tools(self):
        github = MockGitHub()
        async with github.connect() as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert "gh_pr_create" in tool_names
            assert "gh_pr_view" in tool_names
            assert "gh_pr_checks" in tool_names
            assert "gh_pr_reviews" in tool_names
            assert "gh_pr_merge" in tool_names

    @pytest.mark.asyncio
    async def test_pr_create_returns_fixture(self):
        github = MockGitHub()
        async with github.connect() as client:
            result = await client.call_tool("gh_pr_create", {"title": "Fix auth"})
            data = json.loads(result.content[0].text)
            assert data["number"] == 42
            assert data["title"] == "Fix auth"
            assert "url" in data

    @pytest.mark.asyncio
    async def test_call_log_captures_sequence(self):
        github = MockGitHub()
        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "Test"})
            await client.call_tool("gh_pr_checks", {"pr_number": 42})
            await client.call_tool("gh_pr_merge", {"pr_number": 42})

        assert github.call_names == ["gh_pr_create", "gh_pr_checks", "gh_pr_merge"]
        assert github.call_count("gh_pr_create") == 1
        assert github.call_count("gh_pr_merge") == 1

    @pytest.mark.asyncio
    async def test_call_args_captured(self):
        github = MockGitHub()
        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "My PR", "base": "develop"})

        args = github.get_call_args("gh_pr_create")
        assert args["title"] == "My PR"
        assert args["base"] == "develop"

    @pytest.mark.asyncio
    async def test_reset_clears_log(self):
        github = MockGitHub()
        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "Test"})
            assert github.call_count("gh_pr_create") == 1

            github.reset()
            assert github.call_count("gh_pr_create") == 0
            assert github.call_names == []


class TestMockBrainLayerInfra:
    """Verify mock-brainlayer captures stores and returns fixtures."""

    @pytest.mark.asyncio
    async def test_store_captures_content(self):
        brain = MockBrainLayer()
        async with brain.connect() as client:
            await client.call_tool("brain_store", {
                "content": "Auth uses JWT tokens",
                "tags": ["auth", "decision"],
                "importance": 8,
            })

        assert len(brain.stored_items) == 1
        assert brain.stored_items[0]["content"] == "Auth uses JWT tokens"
        assert brain.stored_items[0]["tags"] == ["auth", "decision"]
        assert brain.stored_items[0]["importance"] == 8

    @pytest.mark.asyncio
    async def test_search_returns_fixtures(self):
        brain = MockBrainLayer()
        async with brain.connect() as client:
            result = await client.call_tool("brain_search", {"query": "JWT"})
            data = json.loads(result.content[0].text)
            assert data["total"] >= 1
            assert any("JWT" in r["content"] for r in data["results"])

    @pytest.mark.asyncio
    async def test_recall_returns_context(self):
        brain = MockBrainLayer()
        async with brain.connect() as client:
            result = await client.call_tool("brain_recall", {"mode": "context"})
            data = json.loads(result.content[0].text)
            assert data["mode"] == "context"
            assert "summary" in data


class TestMockVoiceLayerInfra:
    """Verify mock-voicelayer captures messages."""

    @pytest.mark.asyncio
    async def test_speak_captures_text(self):
        voice = MockVoiceLayer()
        async with voice.connect() as client:
            await client.call_tool("voice_speak", {"text": "PR merged!"})

        assert voice.spoken_messages == ["PR merged!"]

    @pytest.mark.asyncio
    async def test_ask_captures_prompt_and_returns_response(self):
        voice = MockVoiceLayer(ask_response="Yes, deploy it.")
        async with voice.connect() as client:
            result = await client.call_tool("voice_ask", {"prompt": "Deploy now?"})
            data = json.loads(result.content[0].text)
            assert data["response"] == "Yes, deploy it."

        assert voice.asked_prompts == ["Deploy now?"]


# =============================================================================
# Test: Behavioral assertion helpers
# =============================================================================


class TestBehavioralAssertions:
    """Test the assertion helpers against known call sequences."""

    def test_called_before_passes(self):
        assert_called_before(["create", "check", "merge"], "create", "merge")

    def test_called_before_fails(self):
        with pytest.raises(AssertionError, match="Expected 'create'.*before.*'merge'"):
            assert_called_before(["merge", "check", "create"], "create", "merge")

    def test_called_before_missing_tool(self):
        with pytest.raises(AssertionError, match="never called"):
            assert_called_before(["create"], "create", "merge")

    def test_called_between_passes(self):
        assert_called_between(["create", "check", "merge"], "create", "check", "merge")

    def test_called_between_fails_wrong_order(self):
        with pytest.raises(AssertionError, match="between"):
            assert_called_between(["create", "merge", "check"], "create", "check", "merge")

    def test_call_sequence_passes_with_gaps(self):
        assert_call_sequence(["a", "x", "b", "y", "c"], ["a", "b", "c"])

    def test_call_sequence_fails(self):
        with pytest.raises(AssertionError, match="not found"):
            assert_call_sequence(["c", "b", "a"], ["a", "b", "c"])

    def test_call_count(self):
        assert_call_count(["a", "b", "a", "c", "a"], "a", 3)

    def test_call_count_fails(self):
        with pytest.raises(AssertionError, match="Expected 'a' called 2 times, got 3"):
            assert_call_count(["a", "b", "a", "c", "a"], "a", 2)

    def test_never_called_passes(self):
        assert_never_called(["create", "check"], "merge")

    def test_never_called_fails(self):
        with pytest.raises(AssertionError, match="never be called"):
            assert_never_called(["create", "merge"], "merge")


# =============================================================================
# Test: PR-loop behavioral contracts
# =============================================================================


class TestPrLoopBehavior:
    """Behavioral tests for the PR workflow contract.

    The pr-loop skill MUST follow this sequence:
    1. Create PR (gh_pr_create)
    2. Check reviews/status (gh_pr_checks, gh_pr_reviews)
    3. Only then merge (gh_pr_merge)

    These tests simulate correct and incorrect sequences against
    the mock server to validate the assertion infrastructure.
    """

    @pytest.mark.asyncio
    async def test_correct_pr_loop_sequence(self):
        """The happy path: create → checks → reviews → merge."""
        github = MockGitHub()
        async with github.connect() as client:
            # Step 1: Create PR
            result = await client.call_tool("gh_pr_create", {
                "title": "feat: add mock MCP harness",
                "body": "Behavioral testing infrastructure",
            })
            pr = json.loads(result.content[0].text)
            pr_number = pr["number"]

            # Step 2: Check CI status
            await client.call_tool("gh_pr_checks", {"pr_number": pr_number})

            # Step 3: Check reviews
            await client.call_tool("gh_pr_reviews", {"pr_number": pr_number})

            # Step 4: Merge
            await client.call_tool("gh_pr_merge", {"pr_number": pr_number})

        # Behavioral assertions
        assert_called_before(github, "gh_pr_create", "gh_pr_merge")
        assert_called_between(github, "gh_pr_create", "gh_pr_checks", "gh_pr_merge")
        assert_called_between(github, "gh_pr_create", "gh_pr_reviews", "gh_pr_merge")
        assert_call_count(github, "gh_pr_create", 1)
        assert_call_count(github, "gh_pr_merge", 1)

    @pytest.mark.asyncio
    async def test_merge_before_create_fails_assertion(self):
        """Detect the anti-pattern: merging before creating a PR."""
        github = MockGitHub()
        async with github.connect() as client:
            # BAD: merge first
            await client.call_tool("gh_pr_merge", {"pr_number": 42})
            await client.call_tool("gh_pr_create", {"title": "Oops"})

        with pytest.raises(AssertionError):
            assert_called_before(github, "gh_pr_create", "gh_pr_merge")

    @pytest.mark.asyncio
    async def test_merge_without_review_fails_assertion(self):
        """Detect the anti-pattern: merging without checking reviews."""
        github = MockGitHub()
        async with github.connect() as client:
            # Create PR
            await client.call_tool("gh_pr_create", {"title": "Skip review"})
            # BAD: merge without checking reviews
            await client.call_tool("gh_pr_merge", {"pr_number": 42})

        # create → merge is fine
        assert_called_before(github, "gh_pr_create", "gh_pr_merge")

        # But reviews was never checked
        with pytest.raises(AssertionError):
            assert_called_between(github, "gh_pr_create", "gh_pr_reviews", "gh_pr_merge")

    @pytest.mark.asyncio
    async def test_merge_without_checks_fails_assertion(self):
        """Detect the anti-pattern: merging without checking CI."""
        github = MockGitHub()
        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "Skip CI"})
            await client.call_tool("gh_pr_reviews", {"pr_number": 42})
            # BAD: merge without checking CI
            await client.call_tool("gh_pr_merge", {"pr_number": 42})

        with pytest.raises(AssertionError):
            assert_called_between(github, "gh_pr_create", "gh_pr_checks", "gh_pr_merge")

    @pytest.mark.asyncio
    async def test_full_sequence_assertion(self):
        """Verify full expected tool call subsequence."""
        github = MockGitHub()
        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "Full flow"})
            await client.call_tool("gh_pr_view", {"pr_number": 42})
            await client.call_tool("gh_pr_checks", {"pr_number": 42})
            await client.call_tool("gh_pr_reviews", {"pr_number": 42})
            await client.call_tool("gh_pr_merge", {"pr_number": 42})

        assert_call_sequence(
            github,
            ["gh_pr_create", "gh_pr_checks", "gh_pr_reviews", "gh_pr_merge"],
        )

    @pytest.mark.asyncio
    async def test_checks_failing_scenario(self):
        """When checks fail, merge should not happen."""
        github = MockGitHub()
        github.set_checks_failing()

        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "Failing CI"})
            result = await client.call_tool("gh_pr_checks", {"pr_number": 42})
            checks = json.loads(result.content[0].text)

            # Agent should NOT merge when checks fail
            assert checks["all_passed"] is False

            # Correct behavior: no merge call
            # (agent logic would decide here — we just verify the fixture works)

        assert_never_called(github, "gh_pr_merge")

    @pytest.mark.asyncio
    async def test_changes_requested_scenario(self):
        """When review has changes_requested, merge should not happen."""
        github = MockGitHub()
        github.set_review_changes_requested()

        async with github.connect() as client:
            await client.call_tool("gh_pr_create", {"title": "Needs fixes"})
            await client.call_tool("gh_pr_checks", {"pr_number": 42})
            result = await client.call_tool("gh_pr_reviews", {"pr_number": 42})
            reviews = json.loads(result.content[0].text)

            assert reviews["approved"] is False
            # Correct behavior: no merge

        assert_never_called(github, "gh_pr_merge")


# =============================================================================
# Test: Multi-server behavioral scenario
# =============================================================================


class TestMultiServerBehavior:
    """Test using multiple mock servers together — the real use case."""

    @pytest.mark.asyncio
    async def test_pr_loop_with_brainlayer_checkpoint(self):
        """Full workflow: search brain → create PR → check → store checkpoint → merge."""
        github = MockGitHub()
        brain = MockBrainLayer()

        async with github.connect() as gh_client, brain.connect() as bl_client:
            # Step 1: Search brain for prior context
            await bl_client.call_tool("brain_search", {"query": "mock MCP harness"})

            # Step 2: Create PR
            await gh_client.call_tool("gh_pr_create", {
                "title": "feat: mock MCP harness",
                "body": "Behavioral testing infra",
            })

            # Step 3: Check CI
            await gh_client.call_tool("gh_pr_checks", {"pr_number": 42})

            # Step 4: Check reviews
            await gh_client.call_tool("gh_pr_reviews", {"pr_number": 42})

            # Step 5: Store checkpoint in brain
            await bl_client.call_tool("brain_store", {
                "content": "PR #42 created for mock MCP harness. CI passed, review approved.",
                "tags": ["pr-merged", "brainlayer", "mock-mcp"],
                "importance": 7,
            })

            # Step 6: Merge
            await gh_client.call_tool("gh_pr_merge", {"pr_number": 42})

        # Assert GitHub sequence
        assert_called_before(github, "gh_pr_create", "gh_pr_merge")
        assert_called_between(github, "gh_pr_create", "gh_pr_checks", "gh_pr_merge")
        assert_called_between(github, "gh_pr_create", "gh_pr_reviews", "gh_pr_merge")

        # Assert BrainLayer was used
        assert brain.was_called("brain_search")
        assert brain.was_called("brain_store")
        assert len(brain.stored_items) == 1
        assert "PR #42" in brain.stored_items[0]["content"]

    @pytest.mark.asyncio
    async def test_pr_loop_with_voice_notification(self):
        """PR workflow with voice notification on completion."""
        github = MockGitHub()
        voice = MockVoiceLayer()

        async with github.connect() as gh_client, voice.connect() as v_client:
            await gh_client.call_tool("gh_pr_create", {"title": "Voice test"})
            await gh_client.call_tool("gh_pr_checks", {"pr_number": 42})
            await gh_client.call_tool("gh_pr_reviews", {"pr_number": 42})
            await gh_client.call_tool("gh_pr_merge", {"pr_number": 42})

            # Notify via voice
            await v_client.call_tool("voice_speak", {
                "text": "PR 42 merged successfully",
            })

        assert_called_before(github, "gh_pr_create", "gh_pr_merge")
        assert voice.spoken_messages == ["PR 42 merged successfully"]
