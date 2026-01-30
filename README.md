# Zikaron

Zikaron is a local knowledge pipeline designed for Claude Code conversations. It helps you manage, search, and retrieve information from your AI interactions with a focus on privacy and local control.

## Features

*   **Local, privacy-first knowledge management** - All data stays on your machine
*   **Indexes and searches AI conversation history** - Never lose context from past conversations
*   **Leverages local LLMs via Ollama** - Powerful semantic search without cloud dependencies
*   **Communication pattern analysis** - Extract your writing style from WhatsApp and Claude chats
*   **Personalized AI rules** - Generate Cursor rules that match your communication preferences
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

Zikaron can analyze your communication patterns from WhatsApp and Claude chats to generate personalized rules for AI assistants.

### Quick Start

```bash
# Analyze your WhatsApp messages (most recent 1000)
zikaron analyze-style

# The tool will:
# 1. Extract messages from WhatsApp's local database
# 2. Analyze your writing style (tone, length, emoji usage, etc.)
# 3. Generate Cursor rules at ~/.cursor/rules/communication-style.md
```

### What It Analyzes

**From WhatsApp:**
- Your writing tone (casual vs formal)
- Average message length
- Emoji and punctuation patterns
- Common phrases you use
- Greeting patterns

**From Claude Chats (optional):**
- Response structures that work well for you
- Common clarifying questions
- Preferred explanation styles

### Generated Rules

The tool creates a Cursor rule file that helps AI assistants:
- Match your natural writing style
- Ask clarifying questions you typically need
- Structure responses the way you prefer
- Draft messages in your voice

### Privacy

All analysis happens locally on your machine. No data is sent anywhere.

**See [docs/communication-analysis.md](docs/communication-analysis.md) for detailed documentation.**

---

## Data Sources

### Claude Code Conversations

Location: `~/.claude/projects/`

Zikaron automatically indexes your Claude Code conversation history stored as JSONL files.

### WhatsApp Messages

Location: `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite`

WhatsApp messages are read directly from the local database (macOS only). Used for communication pattern analysis.

### Claude Desktop/Web Chats

Claude chat transcripts can be imported via manual export from claude.ai (Settings → Data & Privacy → Export my data).

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
zikaron analyze-style --whatsapp-limit 5000  # Analyze more messages
zikaron analyze-style --claude-export ~/claude.json  # Include Claude chats

# Statistics
zikaron stats                          # Show knowledge base stats

# Maintenance
zikaron clear --yes                    # Clear database
zikaron fix-projects                   # Fix project names

# MCP Server
zikaron serve                          # Start MCP server (for Claude Code)
```