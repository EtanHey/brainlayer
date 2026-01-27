#!/usr/bin/env python3
"""Generate 95 stories for zikaron data quality research PRD."""

import json
import os
from pathlib import Path

# All 45 sample types
SAMPLE_TYPES = [
    # Tool Inputs (1-15)
    ("Read-input", "File path inputs to Read tool"),
    ("Edit-input", "old_string/new_string inputs to Edit tool"),
    ("Write-input", "File path + content inputs to Write tool"),
    ("Bash-input", "Command string inputs to Bash tool"),
    ("Grep-input", "Pattern + path inputs to Grep tool"),
    ("Glob-input", "Pattern inputs to Glob tool"),
    ("Task-input", "Agent prompts to Task tool"),
    ("WebFetch-input", "URL + prompt inputs to WebFetch tool"),
    ("WebSearch-input", "Query inputs to WebSearch tool"),
    ("LSP-input", "Operation + position inputs to LSP tool"),
    ("NotebookEdit-input", "Cell edit inputs to NotebookEdit tool"),
    ("TodoWrite-input", "Todo list JSON to TodoWrite tool"),
    ("AskUserQuestion-input", "Question inputs to AskUserQuestion tool"),
    ("Skill-input", "Skill invocation inputs"),
    ("KillShell-input", "Shell ID inputs to KillShell tool"),
    # Tool Results (16-25)
    ("Read-result", "File contents returned from Read tool"),
    ("Bash-result", "Command output from Bash tool"),
    ("Grep-result", "Search matches from Grep tool"),
    ("Glob-result", "File path lists from Glob tool"),
    ("Task-result", "Agent responses from Task tool"),
    ("WebFetch-result", "Page content from WebFetch tool"),
    ("WebSearch-result", "Search results from WebSearch tool"),
    ("LSP-result", "Code intelligence data from LSP tool"),
    ("Error-result", "Failure/error messages from tools"),
    ("Truncated-result", "[truncated] markers in results"),
    # Content Types (26-35)
    ("assistant-explanation", "Natural language explanations from assistant"),
    ("assistant-code", "Code blocks in assistant responses"),
    ("assistant-json", "Structured JSON output from assistant"),
    ("user-question", "User questions and requests"),
    ("user-code", "Code pasted by user"),
    ("user-error", "Errors pasted by user"),
    ("system-prompt", "Initial system prompts"),
    ("system-reminder", "Mid-conversation system reminders"),
    ("progress-message", "Hook progress messages"),
    ("summary-message", "Conversation summaries"),
    # Fringe/Edge Cases (36-45)
    ("empty-content", "Empty or whitespace-only content"),
    ("single-word", "Single word responses"),
    ("very-long", "Content blocks >10KB"),
    ("binary-encoded", "Base64 or binary data"),
    ("non-english", "Hebrew/RTL content"),
    ("emoji-heavy", "Emoji-dense content"),
    ("url-only", "Content that is just URLs"),
    ("path-only", "Content that is just file paths"),
    ("timestamp-only", "Content that is just timestamps"),
    ("duplicate-content", "Repeated/duplicate content"),
]

RESEARCHERS_PER_TYPE = 5
PAPERS_PER_STORY = 3

def generate_research_stories(stories_dir: Path):
    """Generate 75 RESEARCH stories."""

    # Create assignment: each type needs 5 researchers
    # 45 types × 5 = 225 papers
    # 225 ÷ 3 = 75 stories

    # Interleave papers so each story covers DIFFERENT types
    # Create papers organized by researcher first, then type
    all_papers = []
    for researcher in range(1, RESEARCHERS_PER_TYPE + 1):
        for type_idx, (type_name, type_desc) in enumerate(SAMPLE_TYPES):
            all_papers.append({
                "type_idx": type_idx + 1,
                "type_name": type_name,
                "type_desc": type_desc,
                "researcher": researcher,
            })

    # Group into stories of 3 papers each
    story_num = 1
    for i in range(0, len(all_papers), PAPERS_PER_STORY):
        papers = all_papers[i:i + PAPERS_PER_STORY]
        if len(papers) < PAPERS_PER_STORY:
            break

        story_id = f"RESEARCH-{story_num:03d}"

        # Build criteria: 3 criteria per paper × 3 papers = 9 criteria
        criteria = []
        for paper in papers:
            paper_num = papers.index(paper) + 1
            t = paper["type_name"]
            criteria.extend([
                {"text": f"Paper {paper_num} ({t}): Extract 5+ samples of '{t}' from ~/.claude/projects/ JSONL files", "checked": False},
                {"text": f"Paper {paper_num} ({t}): Analyze each sample - is it useful for search? Score 1-5 relevance", "checked": False},
                {"text": f"Paper {paper_num} ({t}): Write recommendation to docs.local/research/{story_id}/paper-{paper_num}-{t}.md (keep/strip/minimize/transform)", "checked": False},
            ])

        title_types = ", ".join([p["type_name"] for p in papers])
        story = {
            "id": story_id,
            "title": f"Research data quality: {title_types}",
            "description": f"Analyze samples of {title_types}. For each type: extract samples, score relevance, recommend handling strategy.",
            "acceptanceCriteria": criteria,
            "storyType": "research"
        }

        story_path = stories_dir / f"{story_id}.json"
        with open(story_path, "w") as f:
            json.dump(story, f, indent=2)

        story_num += 1

    return story_num - 1

def generate_audit_stories(stories_dir: Path, num_audits: int = 20):
    """Generate 20 AUDIT stories for consensus building."""

    for i in range(1, num_audits + 1):
        story_id = f"AUDIT-{i:03d}"

        if i == 1:
            # First auditor - fresh take
            criteria = [
                {"text": "Read ALL docs.local/research/RESEARCH-*/paper-*.md files", "checked": False},
                {"text": "Create docs.local/audit/AUDIT-001-essay.md with your analysis", "checked": False},
                {"text": "For each of the 45 sample types, state your recommendation: KEEP/STRIP/MINIMIZE/TRANSFORM", "checked": False},
                {"text": "Provide reasoning for each recommendation", "checked": False},
                {"text": "Summarize overall data quality strategy", "checked": False},
            ]
            desc = "First auditor: Read all research papers and write initial essay with recommendations for all 45 sample types."
        else:
            # Subsequent auditors - debate previous
            prev_files = " + ".join([f"AUDIT-{j:03d}" for j in range(1, i)])
            criteria = [
                {"text": f"Read ALL previous audit essays: {prev_files}", "checked": False},
                {"text": f"Create docs.local/audit/AUDIT-{i:03d}-debate.md", "checked": False},
                {"text": "For EACH of the 45 sample types: agree or disagree with previous consensus", "checked": False},
                {"text": "If disagreeing, provide counter-arguments with evidence from research papers", "checked": False},
                {"text": "Update consensus tracker: docs.local/audit/consensus-tracker.md with current vote counts", "checked": False},
            ]
            desc = f"Auditor {i}: Read previous {i-1} audit(s), debate conclusions, update consensus tracker for all 45 types."

        story = {
            "id": story_id,
            "title": f"Audit {i}/20: {'Initial analysis' if i == 1 else 'Debate and consensus'}",
            "description": desc,
            "acceptanceCriteria": criteria,
            "storyType": "audit"
        }

        story_path = stories_dir / f"{story_id}.json"
        with open(story_path, "w") as f:
            json.dump(story, f, indent=2)

def generate_index(prd_dir: Path, num_research: int, num_audit: int):
    """Generate index.json."""

    research_ids = [f"RESEARCH-{i:03d}" for i in range(1, num_research + 1)]
    audit_ids = [f"AUDIT-{i:03d}" for i in range(1, num_audit + 1)]
    all_ids = research_ids + audit_ids

    index = {
        "$schema": "https://ralph.dev/schemas/prd-index.schema.json",
        "generatedAt": "2026-01-26T04:30:00Z",
        "nextStory": "RESEARCH-001",
        "storyOrder": all_ids,
        "pending": all_ids,
        "blocked": [],
        "completed": [],
        "stats": {
            "total": len(all_ids),
            "completed": 0,
            "pending": len(all_ids),
            "blocked": 0
        }
    }

    with open(prd_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

def generate_agents_md(prd_dir: Path):
    """Generate AGENTS.md."""

    content = """# AI Agent Instructions for Zikaron Data Quality PRD

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
"""

    with open(prd_dir / "AGENTS.md", "w") as f:
        f.write(content)

def main():
    prd_dir = Path("/Users/etanheyman/Gits/zikaron/prd-json")
    stories_dir = prd_dir / "stories"

    # Ensure directories exist
    stories_dir.mkdir(parents=True, exist_ok=True)

    # Generate stories
    num_research = generate_research_stories(stories_dir)
    print(f"Generated {num_research} RESEARCH stories")

    generate_audit_stories(stories_dir, 20)
    print("Generated 20 AUDIT stories")

    generate_index(prd_dir, num_research, 20)
    print("Generated index.json")

    generate_agents_md(prd_dir)
    print("Generated AGENTS.md")

    print(f"\nTotal: {num_research + 20} stories")

if __name__ == "__main__":
    main()
