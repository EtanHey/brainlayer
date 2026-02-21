# Contributing to BrainLayer

Thanks for your interest in contributing! BrainLayer is open source under Apache 2.0.

## Development Setup

```bash
git clone https://github.com/EtanHey/brainlayer.git
cd brainlayer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/brainlayer/
├── mcp/                  # MCP server (14 tools, stdio-based)
│   └── __init__.py
├── pipeline/             # Indexing pipeline stages
│   ├── extract.py        #   Stage 1: Parse JSONL conversations
│   ├── classify.py       #   Stage 2: Content classification
│   ├── chunk.py          #   Stage 3: AST-aware chunking
│   ├── enrichment.py     #   LLM enrichment (Ollama/MLX)
│   └── session_enrichment.py  # Session-level analysis
├── vector_store.py       # SQLite + sqlite-vec storage layer
├── embeddings.py         # bge-large-en-v1.5 embedding model
├── store.py              # Write API (store_memory)
├── paths.py              # DB path resolution
├── daemon.py             # FastAPI HTTP daemon
├── client.py             # Python client for daemon API
├── cli/                  # Typer CLI
│   └── __init__.py
└── dashboard/            # Rich TUI dashboard
    ├── search.py
    └── views.py
```

## Running Tests

```bash
# Full test suite (268 tests, ~6s)
pytest tests/

# Skip integration tests (need production DB + embedding model)
pytest tests/ -m "not integration"

# Single file
pytest tests/test_phase3_qa.py -v
```

Integration tests (marked `@pytest.mark.integration`) require the production database and embedding model. They're skipped in fast local runs.

## Linting

```bash
ruff check src/           # Check for issues
ruff format src/          # Auto-format
```

Config is in `pyproject.toml` — line length 120, Python 3.11 target.

## Making Changes

1. **Branch from `main`**: `git checkout -b feature/your-feature`
2. **Write tests first** when adding new functionality
3. **Run the full suite** before pushing: `pytest tests/ -m "not integration" && ruff check src/`
4. **Keep commits focused** — one logical change per commit

## Pull Request Process

1. Create a PR against `main`
2. Tests must pass
3. CodeRabbit will review automatically — fix HIGH/CRITICAL issues, style-only comments can be skipped
4. PRs are squash-merged

## Key Patterns

- **Error handling in MCP tools**: Use `_error_result()` for user-facing errors (sets `isError=True`)
- **Database access**: Always use `VectorStore` class, never raw SQL outside of it
- **Logging**: Use `logging.getLogger(__name__)`, never `print()`
- **Env vars**: Prefix with `BRAINLAYER_` (see README for full list)
- **Schema constraints**: MCP tool input schemas should have `minimum`/`maximum` on numeric params, with matching server-side clamping in `call_tool`

## Adding an MCP Tool

1. Add the `Tool(...)` definition to the tools list in `mcp/__init__.py`
2. Add the handler function (prefix with `_`)
3. Add the routing in `call_tool`
4. Add `ToolAnnotations` (readOnlyHint, etc.)
5. Update the tool count in `test_think_recall_integration.py`
6. Write tests

## Reporting Issues

Open an issue with: what you expected, what happened, steps to reproduce, and your Python version + OS.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
