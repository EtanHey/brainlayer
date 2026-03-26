"""Mock GitHub MCP server — simulates gh pr create/view/merge/checks.

Deterministic responses for behavioral testing of PR workflows.
Configurable fixtures per tool. Call log captures full sequence.

Usage:
    github = MockGitHub()
    async with github.connect() as client:
        result = await client.call_tool("gh_pr_create", {"title": "Fix auth", "body": "..."})
        result = await client.call_tool("gh_pr_checks", {"pr_number": 42})
        result = await client.call_tool("gh_pr_merge", {"pr_number": 42})

        assert github.called_before("gh_pr_create", "gh_pr_merge")
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .base import MockMcpServer

# --- Default fixtures ---

DEFAULT_PR = {
    "number": 42,
    "url": "https://github.com/EtanHey/brainlayer/pull/42",
    "title": "Fix auth bug",
    "state": "open",
    "head": {"ref": "feat/fix-auth"},
    "base": {"ref": "main"},
    "mergeable": True,
}

DEFAULT_CHECKS = {
    "pr_number": 42,
    "total_count": 3,
    "checks": [
        {"name": "lint", "status": "completed", "conclusion": "success"},
        {"name": "test", "status": "completed", "conclusion": "success"},
        {"name": "build", "status": "completed", "conclusion": "success"},
    ],
    "all_passed": True,
}

DEFAULT_REVIEWS = {
    "pr_number": 42,
    "reviews": [
        {
            "user": "coderabbit[bot]",
            "state": "APPROVED",
            "body": "LGTM! No issues found.",
        },
    ],
    "approved": True,
}

DEFAULT_MERGE = {
    "pr_number": 42,
    "merged": True,
    "message": "Pull request #42 merged successfully.",
    "sha": "abc123def456",
}


class MockGitHub(MockMcpServer):
    """Mock GitHub MCP server with configurable PR workflow responses."""

    def __init__(
        self,
        pr_fixture: dict[str, Any] | None = None,
        checks_fixture: dict[str, Any] | None = None,
        reviews_fixture: dict[str, Any] | None = None,
        merge_fixture: dict[str, Any] | None = None,
    ):
        self._pr_fixture = pr_fixture if pr_fixture is not None else copy.deepcopy(DEFAULT_PR)
        self._checks_fixture = checks_fixture if checks_fixture is not None else copy.deepcopy(DEFAULT_CHECKS)
        self._reviews_fixture = reviews_fixture if reviews_fixture is not None else copy.deepcopy(DEFAULT_REVIEWS)
        self._merge_fixture = merge_fixture if merge_fixture is not None else copy.deepcopy(DEFAULT_MERGE)
        super().__init__("mock-github")

    def _register_tools(self) -> None:
        self.register_tool(
            "gh_pr_create",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "PR title"},
                    "body": {"type": "string", "description": "PR body"},
                    "base": {"type": "string", "description": "Base branch", "default": "main"},
                    "head": {"type": "string", "description": "Head branch"},
                    "draft": {"type": "boolean", "description": "Create as draft", "default": False},
                },
                "required": ["title"],
            },
            handler=self._handle_pr_create,
            description="Create a pull request",
        )

        self.register_tool(
            "gh_pr_view",
            {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
                "required": ["pr_number"],
            },
            handler=self._handle_pr_view,
            description="View pull request details",
        )

        self.register_tool(
            "gh_pr_checks",
            {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
                "required": ["pr_number"],
            },
            handler=self._handle_pr_checks,
            description="View PR check status",
        )

        self.register_tool(
            "gh_pr_reviews",
            {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
                "required": ["pr_number"],
            },
            handler=self._handle_pr_reviews,
            description="View PR reviews",
        )

        self.register_tool(
            "gh_pr_merge",
            {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                    "method": {
                        "type": "string",
                        "description": "Merge method",
                        "enum": ["merge", "squash", "rebase"],
                        "default": "squash",
                    },
                },
                "required": ["pr_number"],
            },
            handler=self._handle_pr_merge,
            description="Merge a pull request",
        )

    def _handle_pr_create(self, args: dict[str, Any]) -> str:
        pr = dict(self._pr_fixture)
        if "title" in args:
            pr["title"] = args["title"]
        if "head" in args:
            pr["head"] = {"ref": args["head"]}
        return json.dumps(pr)

    def _handle_pr_view(self, args: dict[str, Any]) -> str:
        return json.dumps(self._pr_fixture)

    def _handle_pr_checks(self, args: dict[str, Any]) -> str:
        return json.dumps(self._checks_fixture)

    def _handle_pr_reviews(self, args: dict[str, Any]) -> str:
        return json.dumps(self._reviews_fixture)

    def _handle_pr_merge(self, args: dict[str, Any]) -> str:
        return json.dumps(self._merge_fixture)

    def reset(self) -> None:
        """Clear call log and restore default fixtures."""
        super().reset()
        self._pr_fixture = copy.deepcopy(DEFAULT_PR)
        self._checks_fixture = copy.deepcopy(DEFAULT_CHECKS)
        self._reviews_fixture = copy.deepcopy(DEFAULT_REVIEWS)
        self._merge_fixture = copy.deepcopy(DEFAULT_MERGE)

    # --- Convenience: configure failure scenarios ---

    def set_checks_failing(self) -> None:
        """Configure checks to return failures."""
        self._checks_fixture = {
            "pr_number": 42,
            "total_count": 3,
            "checks": [
                {"name": "lint", "status": "completed", "conclusion": "success"},
                {"name": "test", "status": "completed", "conclusion": "failure"},
                {"name": "build", "status": "completed", "conclusion": "success"},
            ],
            "all_passed": False,
        }

    def set_review_changes_requested(self) -> None:
        """Configure reviews to return changes_requested."""
        self._reviews_fixture = {
            "pr_number": 42,
            "reviews": [
                {
                    "user": "coderabbit[bot]",
                    "state": "CHANGES_REQUESTED",
                    "body": "Found 3 issues that need fixing.",
                },
            ],
            "approved": False,
        }

    def set_merge_conflict(self) -> None:
        """Configure merge to return conflict."""
        self._merge_fixture = {
            "pr_number": 42,
            "merged": False,
            "message": "Pull request #42 has merge conflicts.",
            "error": "MERGE_CONFLICT",
        }
