# AI Agent Instructions for Zikaron Data Quality PRD

## Overview

This PRD contains 95 stories:
- **75 RESEARCH stories**: Analyze 45 sample types, 5 researchers per type
- **20 AUDIT stories**: Build consensus on data handling recommendations

## Folder Structure

```
docs.local/
├── research/
│   ├── RESEARCH-001/
│   │   ├── paper-1-{type}.md
│   │   ├── paper-2-{type}.md
│   │   └── paper-3-{type}.md
│   └── ... (75 folders)
└── audit/
    ├── AUDIT-001-essay.md        # First auditor - fresh analysis
    ├── AUDIT-002-debate.md       # Debates 001
    ├── ...
    ├── AUDIT-020-debate.md       # Final debate
    └── consensus-tracker.md      # Vote counts for each type
```

## RESEARCH Story Guidelines

Each RESEARCH story covers 3 sample types with 3 criteria each (9 total):

1. **Extract samples**: Find 5+ examples from ~/.claude/projects/ JSONL files
2. **Analyze relevance**: Score each sample 1-5 for search usefulness
3. **Recommend handling**: Write to docs.local/research/{STORY-ID}/paper-{N}-{type}.md

### Paper Format

```markdown
# {Type} Analysis

## Samples Found

### Sample 1
- **Source**: {file path}
- **Content**: ```{actual content}```
- **Relevance Score**: X/5
- **Reasoning**: {why this score}

... (5+ samples)

## Recommendation

**Verdict**: KEEP | STRIP | MINIMIZE | TRANSFORM

**Reasoning**: {detailed explanation}

**If TRANSFORM**: {describe transformation}
```

## AUDIT Story Guidelines

### AUDIT-001 (First Auditor)
- Read ALL research papers
- Write fresh analysis to docs.local/audit/AUDIT-001-essay.md
- Cover all 45 types with recommendations

### AUDIT-002 to AUDIT-020 (Debaters)
- Read ALL previous audits
- Write debate to docs.local/audit/AUDIT-{N}-debate.md
- Agree or disagree for each of 45 types
- Update consensus-tracker.md

### Consensus Tracker Format

```markdown
# Consensus Tracker

| Type | KEEP | STRIP | MINIMIZE | TRANSFORM | Current Winner |
|------|------|-------|----------|-----------|----------------|
| Read-input | 2 | 15 | 3 | 0 | STRIP |
| ... | ... | ... | ... | ... | ... |
```

## The 45 Sample Types

### Tool Inputs (1-15)
1. Read-input, 2. Edit-input, 3. Write-input, 4. Bash-input, 5. Grep-input,
6. Glob-input, 7. Task-input, 8. WebFetch-input, 9. WebSearch-input, 10. LSP-input,
11. NotebookEdit-input, 12. TodoWrite-input, 13. AskUserQuestion-input, 14. Skill-input, 15. KillShell-input

### Tool Results (16-25)
16. Read-result, 17. Bash-result, 18. Grep-result, 19. Glob-result, 20. Task-result,
21. WebFetch-result, 22. WebSearch-result, 23. LSP-result, 24. Error-result, 25. Truncated-result

### Content Types (26-35)
26. assistant-explanation, 27. assistant-code, 28. assistant-json, 29. user-question,
30. user-code, 31. user-error, 32. system-prompt, 33. system-reminder, 34. progress-message, 35. summary-message

### Fringe Cases (36-45)
36. empty-content, 37. single-word, 38. very-long, 39. binary-encoded, 40. non-english,
41. emoji-heavy, 42. url-only, 43. path-only, 44. timestamp-only, 45. duplicate-content

## Data Sources

- **JSONL conversations**: ~/.claude/projects/
- **Current index backup**: ~/.local/share/zikaron-backup-*
- **Pipeline code**: src/zikaron/pipeline/ (extract.py, classify.py, chunk.py)

## DO NOT

- Do NOT edit index.json directly (use update.json)
- Do NOT mix research data between stories
- Do NOT skip samples - we need comprehensive coverage
- Do NOT commit after RESEARCH stories (only after AUDIT-020)
