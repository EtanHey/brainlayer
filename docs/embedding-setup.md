# Embedding Setup for Style Analysis

**StyleDistance** is the best model for style analysis. It clusters by *how* you write (formality, emoji, punctuation, phrasing)—not by topic. General embeddings (Qwen3, bge-m3) cluster by content; they're worse for style.

## Usage

```bash
brainlayer analyze-evolution --use-embeddings
```

## Setup

- **Install**: `sentence-transformers` (included in brainlayer deps)
- **First run**: Downloads `StyleDistance/mstyledistance` to `~/.cache/huggingface/` (~500MB)
- **Auth**: None for public models. If rate limited: `huggingface-cli login` (same as ml-training-pipeline)
- **Env**: `HF_TOKEN` used automatically if set

## Hardware

- **M1 Pro 32GB**: Runs comfortably
- **Storage**: ~500MB model
- **Embedding time**: ~30–60 min per 100K messages
