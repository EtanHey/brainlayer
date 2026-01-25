# Ralph Agent Instructions - Zikaron

You are executing PRD stories for **Zikaron** (זיכרון), a local knowledge pipeline for Claude Code conversations.

## Project Context

This is a **Python project** using:
- **uv** for package management
- **ChromaDB** for vector storage
- **Ollama** for local embeddings (nomic-embed-text)
- **tree-sitter** for AST-aware code chunking
- **MCP** (Model Context Protocol) for Claude Code integration

## Working Directory

All work happens in: `~/Gits/zikaron`

## Before Starting

1. Activate virtual environment: `source .venv/bin/activate`
2. Ensure Ollama is running: `ollama list`
3. Check dependencies: `uv pip list`

## Key Files

| File | Purpose |
|------|---------|
| `src/zikaron/pipeline/*.py` | Processing stages |
| `src/zikaron/cli/__init__.py` | CLI commands |
| `src/zikaron/mcp/__init__.py` | MCP server |
| `tests/` | Test files |
| `CLAUDE.md` | Project documentation |

## Testing Commands

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_classify.py -v

# Test CLI
zikaron --help
zikaron index --help
zikaron search "test"

# Test MCP server
zikaron serve
```

## Git Rules

- Commit after each story completes
- Use conventional commits: `feat:`, `fix:`, `test:`
- Include story ID in commit message

## Context Wiring (US-006, US-007)

These stories wire zikaron into the claude-golem context system:

**Required contexts:**
- `base` - Universal rules
- `skill-index` - Available skills
- `workflow/interactive` - CLAUDE_COUNTER, git safety

**Available skills:**
| Skill | When to Use |
|-------|-------------|
| `/golem-powers:context-audit` | Diagnose missing contexts |
| `/golem-powers:ralph-commit` | For "Commit:" criteria |
| `/golem-powers:coderabbit` | Code review before commits |

## CodeRabbit Iteration Rule

For "Run CodeRabbit review" criteria:
1. Run: `cr review --prompt-only --type uncommitted`
2. If issues found → Fix them
3. Repeat until clean

## Research Reference

The pipeline is based on research findings in `~/Gits/claude-golem/docs.local/`:
- `compass_artifact_wf-92919407-*.md` - Main research on code RAG
- Key finding: Observation masking often beats LLM summarization
