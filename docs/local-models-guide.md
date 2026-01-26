# Local LLM Models Guide

> Last updated: 2026-01-26
> Hardware: MacBook Pro M1 Pro, 32GB RAM

---

## Quick Reference: Models for Your Mac (32GB RAM)

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `qwen3-coder` | 19GB | Fast (MoE, 3.3B active) | Agentic coding, best overall |
| `qwen2.5-coder:32b` | 20GB | ~15-20 tok/s | High quality code gen |
| `qwen2.5-coder:14b` | 9GB | ~50 tok/s | Fast coding tasks |
| `devstral` | 14GB | ~30 tok/s | Codebase navigation, file ops |

---

## Model Details

### qwen3-coder (RECOMMENDED)

```bash
ollama pull qwen3-coder
```

- **Architecture**: MoE (Mixture of Experts)
- **Parameters**: 30.5B total, only 3.3B activated
- **Size**: ~19GB
- **RAM needed**: 32GB (perfect for your Mac)
- **Why it's good**: Smarter than dense models, runs fast due to MoE
- **Best for**: Agentic coding, multi-step tasks, code analysis

### qwen2.5-coder:32b

```bash
ollama pull qwen2.5-coder:32b
```

- **Architecture**: Dense transformer
- **Parameters**: 32.5B
- **Size**: ~20GB
- **RAM needed**: 24-32GB
- **Benchmark**: Best open-source on EvalPlus, LiveCodeBench, BigCodeBench
- **Best for**: High quality code generation, complex problems

### qwen2.5-coder:14b

```bash
ollama pull qwen2.5-coder:14b
```

- **Parameters**: 14B
- **Size**: ~9GB
- **Speed**: ~50+ tok/s on M1 Pro
- **Best for**: Fast iterations, simpler tasks, when speed matters

### devstral

```bash
ollama pull devstral
```

- **By**: Mistral AI
- **Size**: ~14GB
- **Specialty**: File system operations, code navigation, large codebases
- **Best for**: Codebase exploration, documentation audits, multi-file edits

---

## Benchmark Comparison (SWE-Bench Verified)

| Model | Score | Type |
|-------|-------|------|
| Claude 4 Sonnet | 72.7% | Closed-source |
| Claude 4 Opus | 72.5% | Closed-source |
| Qwen3-Coder | 69.6% | Open-source |
| DeepSeek-V3 | ~68% | Open-source |

**Takeaway**: Open-source is only ~3% behind Claude now.

---

## Models That WON'T Fit (32GB RAM)

| Model | Size | Issue |
|-------|------|-------|
| qwen3-coder:480b | 163-368GB | Way too big |
| deepseek-v3 | 400GB+ | Won't fit |
| Any 70B+ dense model | 40GB+ | Will swap, very slow |

---

## Usage Patterns

### For Code Research/Documentation
```bash
ollama run qwen3-coder
# or
ollama run devstral
```

### For Fast Code Generation
```bash
ollama run qwen2.5-coder:14b
```

### For Highest Quality (Slower)
```bash
ollama run qwen2.5-coder:32b
```

---

## Running with Context

### Pipe a file
```bash
cat README.md | ollama run qwen3-coder "Analyze this README and suggest improvements"
```

### Interactive session with system prompt
```bash
ollama run qwen3-coder --system "You are a senior software architect auditing a codebase."
```

---

## Free Cloud Alternatives (When Local Won't Cut It)

| Service | Models | Limit |
|---------|--------|-------|
| Gemini CLI | Gemini 2.5 Pro | 1000 req/day |
| Groq | Llama 3.3 70B | Very fast, free tier |
| Together.ai | All major models | $25 free credit |
| OpenRouter | Everything | Pay-per-use |

---

## Check What's Installed

```bash
ollama list
```

## Check Available Space

```bash
df -h /
du -sh ~/.ollama/models/
```

---

## Sources

- [Ollama qwen2.5-coder](https://ollama.com/library/qwen2.5-coder)
- [Ollama qwen3-coder](https://ollama.com/library/qwen3-coder)
- [Best Open Source LLMs for Coding 2026](https://www.siliconflow.com/articles/en/best-open-source-LLMs-for-coding)
- [Qwen AI Coding Review](https://www.index.dev/blog/qwen-ai-coding-review)
