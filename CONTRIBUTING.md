# Contributing to BrainLayer

Thanks for your interest in contributing! BrainLayer is an open-source memory layer for AI agents, and contributions of all kinds are welcome.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/EtanHey/brainlayer.git
cd brainlayer

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Development Workflow

1. **Fork** the repository
2. **Create a feature branch** from `main`
3. **Write tests first** (TDD encouraged)
4. **Implement** your changes
5. **Run the test suite** to verify nothing breaks
6. **Submit a PR** with a clear description

## Code Structure

```
src/brainlayer/
  __init__.py          # Package entry + CLI
  vector_store.py      # Core: SQLite + sqlite-vec search
  engine.py            # Intelligence: think, recall, sessions, current_context
  chunker.py           # Text chunking with sentence boundaries
  enrichment.py        # LLM enrichment pipeline
  mcp/                 # MCP server (12 tools)
    __init__.py        # Tool definitions + handlers
  indexers/            # Source-specific indexers
    claude_code.py     # Claude Code conversation indexer
    whatsapp.py        # WhatsApp export indexer
    youtube.py         # YouTube transcript indexer
  paths.py             # Path resolution
  normalize.py         # Project name normalization
tests/
  test_engine.py       # Unit + integration tests for engine
  test_think_recall_integration.py  # Real-DB integration tests
  test_*.py            # Other test modules
```

## Testing

We use pytest. Tests are organized by speed:

- **Unit tests** (`test_engine.py` unit classes): Fast, no DB needed, run always
- **Integration tests** (classes with `Integration` or `Real` suffix): Use the production DB, may load embeddings

```bash
# Run just the fast unit tests
pytest tests/test_engine.py -v -k "not Integration"

# Run everything
pytest tests/ -v
```

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update the MCP tool count test if adding/removing tools
- Don't break existing tests

## Reporting Issues

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
