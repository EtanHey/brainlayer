# Local Environment Context

## Hardware
- **Machine:** MacBook Pro M1 Pro
- **RAM:** 32 GB
- **Storage:** 460GB total, ~28GB free

## Ollama Models Available
| Model | Size | Use Case |
|-------|------|----------|
| nomic-embed-text | 274 MB | Legacy embedding (768 dims) |
| mxbai-embed-large | 669 MB | Alternative embeddings |
| qwen3-coder-64k | 18 GB | Code generation |
| qwen2.5-coder:7b | 4.7 GB | Fast code tasks |
| llama3.1:8b | 4.9 GB | General purpose |

## Current Zikaron Setup (Feb 2026 - sqlite-vec)
- **Vector DB:** sqlite-vec at `~/.local/share/zikaron/zikaron.db`
- **Embedding Model:** bge-large-en-v1.5 via sentence-transformers (1024 dims)
- **Service:** FastAPI daemon on `/tmp/zikaron.sock`
- **Python:** 3.11+ with APSW for macOS SQLite extension support
- **Package Location:** `~/Gits/golems/packages/zikaron/`

## Data Sources Indexed
- Claude Code conversations (`~/.claude/projects/`)
- WhatsApp exports (style analysis)
- Gemini/YouTube exports

## Current Performance (Phase 1-2 Implemented)
- Cold start: **15s** (down from 180s)
- Warm search: **<2s** (with daemon running)
- Model load: **8s** (down from 30s via Ollama)
- Memory: **4GB** (down from 6GB+)

## Key Paths
```
~/Gits/golems/packages/zikaron/        # Source code
~/.local/share/zikaron/zikaron.db      # sqlite-vec database
~/.local/share/zikaron/chromadb.backup/# Old ChromaDB (after migration)
~/.claude/projects/                     # Claude Code conversations
/tmp/zikaron.sock                       # Daemon Unix socket
docs.local/research/                    # Research documents
```
