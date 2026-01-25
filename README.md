# Zikaron (זיכרון)

> Local knowledge pipeline for Claude Code conversations

**Zikaron** indexes your Claude Code conversation history into a searchable knowledge base. Query past solutions, code patterns, and debugging sessions through CLI or MCP.

## Features

- **AST-aware chunking** - Uses tree-sitter to split code at semantic boundaries
- **Content classification** - Preserves stack traces and AI-generated code verbatim
- **Local embeddings** - Uses Ollama (nomic-embed-text) for privacy
- **MCP integration** - Claude Code can query the knowledge base directly
- **Research-based** - Implements findings from Meta-RAG, cAST, and other 2024-2026 papers

## Quick Start

```bash
# Clone and setup
cd ~/Gits/zikaron
uv venv && source .venv/bin/activate
uv pip install -e .

# Pull embedding model
ollama pull nomic-embed-text

# Index your conversations
zikaron index

# Search
zikaron search "how did I implement authentication"
```

## Architecture

```
~/.claude/projects/     →  Pipeline  →  ChromaDB  →  CLI / MCP
   (JSONL files)           (5 stages)    (vectors)    (query)
```

See [CLAUDE.md](CLAUDE.md) for detailed documentation.

## License

MIT
