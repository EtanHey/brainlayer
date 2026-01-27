# Zikaron

Zikaron is a local knowledge pipeline designed for Claude Code conversations. It helps you manage, search, and retrieve information from your AI interactions with a focus on privacy and local control.

## Features

*   Local, privacy-first knowledge management.
*   Indexes and searches AI conversation history.
*   Leverages local LLMs via Ollama for powerful semantic search.
*   Provides a command-line interface for easy interaction.

## The Memory Layer
Zikaron was built to serve as the long-term memory for [https://github.com/EtanHey/claude-golem](Claude Golem). While the Golem executes autonomous coding loops in the terminal, Zikaron ensures that the resulting conversation logs and architectural decisions are indexed locally via Ollama, preventing "context rot" and allowing you to clean up your workspace without losing insights.


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

### Usage Example

Once both servers are running, you can interact with Zikaron using its CLI. For example, to search your knowledge base:
```bash
zikaron search "your query here"
```