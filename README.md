# Zikaron

Zikaron is a local knowledge pipeline designed for Claude Code conversations. It helps you manage, search, and retrieve information from your AI interactions with a focus on privacy and local control.

## Features

*   **Local, privacy-first knowledge management** - All data stays on your machine
*   **Indexes and searches AI conversation history** - Never lose context from past conversations
*   **Leverages local LLMs via Ollama** - Powerful semantic search without cloud dependencies
*   **Communication pattern analysis** - Extract your writing style from WhatsApp, Claude, YouTube, and Gemini
*   **Personalized AI rules** - Generate Cursor rules, plus Claude.ai and Gemini casual-instruction sets for texts/DMs
*   **Command-line interface** - Easy interaction and automation

## The Memory Layer
Zikaron was built to serve as the long-term memory for [Claude-Golem](https://github.com/EtanHey/claude-golem) . While the Golem executes autonomous coding loops in the terminal, Zikaron ensures that the resulting conversation logs and architectural decisions are indexed locally via Ollama, preventing "context rot" and allowing you to clean up your workspace without losing insights.


## Getting Started

This guide will help you set up and run the Zikaron project locally.

### Prerequisites

*   **Python:** Version 3.11 or higher installed on your system.
*   **Ollama:** The Ollama application must be installed and running on your system to provide local LLM capabilities. Download from [ollama.com](https://ollama.com/).
    *   Ensure you have a suitable model pulled, such as `qwen3-coder` (recommended for its coding and reasoning capabilities). You can pull it using `ollama pull qwen3-coder`.

### Project Setup

Follow these steps to set up your local development environment:

1.  **Navigate to the Project Directory:**
    Ensure you are in the root of the Zikaron project (e.g., `/Users/{username}/Gits/zikaron`).
    ```bash
    cd /Users/{username}/Gits/zikaron
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies. This creates a `.venv` directory.
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    This command loads the virtual environment's Python interpreter and makes its packages available. You'll need to do this every time you open a new terminal to work on the project.
    ```bash
    source .venv/bin/activate
    ```
    *(Your terminal prompt should change to indicate the active environment, e.g., `(.venv) {username} ~/Gits/zikaron [master] $`)*

4.  **Install Project Dependencies:**
    This installs Zikaron and all its required libraries in "editable" mode, which is ideal for development.
    ```bash
    python3 -m pip install -e .
    ```

### Running Zikaron

To use Zikaron, you need to run two services concurrently: the Ollama AI model server and the Zikaron MCP server.

**Terminal 1: Start the Ollama Server**
Ensure Ollama is installed and running. If you haven't already, pull a model (e.g., `ollama pull qwen3-coder`). Then, start the Ollama server.
```bash
ollama serve
# Alternatively, if you want to run a specific model interactively and have it serve:
# ollama run qwen3-coder
```
*(Keep this terminal running; it serves the AI models.)*

**Terminal 2: Start the Zikaron MCP Server**
Make sure your virtual environment is activated (you should see `(.venv)` in your prompt). Then, run the Zikaron command-line interface for the MCP server.
```bash
zikaron-mcp
```
*(This server will also run continuously. Keep this terminal open.)*

### Usage Examples

Once both servers are running, you can interact with Zikaron using its CLI:

**Search your knowledge base:**
```bash
zikaron search "your query here"
```

**Index Claude Code conversations:**
```bash
zikaron index
```

**Analyze your communication style:**
```bash
# Extract patterns from WhatsApp and generate personalized rules
zikaron analyze-style

# With Claude chat export
zikaron analyze-style --claude-export ~/Downloads/claude-export.json

# Custom output location
zikaron analyze-style --output ~/.cursor/rules/my-style.md
```

**View statistics:**
```bash
zikaron stats
```

---

## Communication Pattern Analysis

Zikaron analyzes your communication patterns from WhatsApp, Claude, YouTube, and Gemini to generate personalized rules for AI assistants.

### Quick Start

```bash
# Basic style analysis (WhatsApp + optional Claude)
zikaron analyze-style

# Full longitudinal analysis with embeddings (recommended)
zikaron analyze-evolution --use-embeddings -c /path/to/claude-export/conversations.json -o data/archives/style-$(date +%Y-%m-%d-%H%M) -y
```

### What It Analyzes

**From WhatsApp:** Tone, length, emoji, common phrases, greetings (local SQLite).

**From Claude:** Response structures, clarifying questions (export from claude.ai).

**From YouTube:** Comments from Google Takeout (`data/youtube-comments/comments.csv`).

**From Gemini:** Chat history (when included in Google Takeout).

### Generated Output

- **Cursor rules** – For coding assistants
- **Claude.ai / Gemini instructions** – For casual texts and DMs; copy from `claude-ai-casual-instructions.md` and `gemini-casual-instructions.md` in the archive output into each app’s Settings → Personalization

### Privacy

All analysis happens locally on your machine. No data is sent anywhere.

**See [docs/communication-analysis.md](docs/communication-analysis.md) for detailed documentation.**

---

## Data Sources

| Source | Location | Use |
|--------|----------|-----|
| **Claude Code** | `~/.claude/projects/` | Indexing (JSONL) |
| **Claude.ai chats** | Export from claude.ai | Style analysis |
| **WhatsApp** | `~/Library/.../ChatStorage.sqlite` (macOS) | Style analysis |
| **YouTube** | `data/youtube-comments/comments.csv` or Takeout zip | Style analysis |
| **Gemini** | Google Takeout (My Activity/Gemini Apps) | Style analysis |

---

## CLI Commands

```bash
# Index conversations
zikaron index                          # Index all Claude Code conversations
zikaron index --project myproject      # Index specific project only

# Search
zikaron search "authentication"        # Semantic search
zikaron search "config.py" --text      # Text-based search
zikaron search "bug fix" --project app # Search within project

# Communication analysis
zikaron analyze-style                  # Analyze WhatsApp messages
zikaron analyze-evolution --use-embeddings -c ~/claude.json -y  # Full longitudinal analysis
zikaron list-chats -o chats.csv        # List chats for relationship tagging

# Statistics
zikaron stats                          # Show knowledge base stats

# Maintenance
zikaron clear --yes                    # Clear database
zikaron fix-projects                   # Fix project names
bash scripts/archive-style-analysis.sh # Archive /tmp runs to data/archives/
bash scripts/extract-youtube-and-cleanup.sh  # Extract YouTube from Takeout, delete zips

# MCP Server
zikaron serve                          # Start MCP server (for Claude Code)
```