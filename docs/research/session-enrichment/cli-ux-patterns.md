# CLI UX patterns for knowledge search tools

**The best CLI search tools converge on six proven display patterns, three levels of progressive disclosure, and a composable architecture that chains simple interactive elements rather than building monolithic TUI applications.** This research synthesizes patterns from ripgrep, fzf, GitHub CLI, kubectl, and dozens of knowledge management tools to provide a concrete blueprint for building a terminal-based knowledge search interface. The Python `rich` library combined with `click`/`typer`, `plotext`, and `iterfzf` forms the ideal stack. What follows is a comprehensive pattern guide with ASCII mockups, code examples, library recommendations, and prioritization frameworks.

---

## Six foundational display patterns power every great CLI search tool

Research across ripgrep, fzf, ag, gh, bat, fd, jq, httpie, and kubectl reveals **six primary layout patterns** that cover virtually all search result display needs:

**Pattern 1 — Grouped-by-file with context** (ripgrep, ag). Results cluster under filename headers with line numbers and highlighted matches. Blank lines separate file groups. This is the default TTY mode for both `rg` and `ag`, and the most natural pattern for code/content search:

```
src/components/SearchBar.tsx
 12:  function handleSearch(query: string) {
 13-    const results = performSearch(query);
 14-    setResults(results);

src/utils/api.ts
 45:  export async function handleSearch(q) {
 46-    const resp = await fetch(`/api?q=${q}`);
```

File paths render in **magenta bold**, line numbers in **green**, match text in **red bold**, and context lines appear dimmed with `-` separators. Match lines use `:` separators. The `--` divider separates non-adjacent context blocks within the same file.

**Pattern 2 — Aligned tabular** (kubectl, gh). Columnar data with aligned headers and whitespace-padded columns. GitHub CLI's `gh search repos` and kubectl's default output both use this for metadata-rich listings:

```
NAME                  DESCRIPTION                   ★      LANG   UPDATED
BurntSushi/ripgrep    Recursively search director…   48.2k  Rust   2d ago
sharkdp/fd            A simple, fast alternative…    35.1k  Rust   1w ago
sharkdp/bat           A cat clone with wings         51.2k  Rust   5d ago
```

**Pattern 3 — Interactive filtered list** (fzf). Real-time filtering with a prompt, scrollable list, match highlighting, and an optional preview pane. The user types to narrow results dynamically, with a `7/245` info line showing matches from total candidates.

**Pattern 4 — Rich metadata cards**. A composite pattern combining panels with inline badges, scores, tags, timestamps, and code snippets — ideal for knowledge search results:

```
╭─ 1. auth/middleware.ts ────────────────── Score: 0.95 ── 2 days ago ─╮
│  Tags: [auth] [middleware] [security]    Category: Backend            │
│                                                                       │
│  14 │ export function authMiddleware(req, res, next) {                │
│  15 │   const token = req.headers.authorization;                      │
│  16 │   if (!validateToken(token)) {                                  │
╰──────────────────────────────────────────────────────────────────────╯
```

**Pattern 5 — Compact table with expandable rows**. A hybrid that shows a scored, sortable table with the ability to expand individual results in-place — the best default for knowledge search:

```
 #  FILE                        LINE  SCORE  UPDATED     TAGS
 1  src/db/connection.ts          23  0.97   2d ago      db,core
 2  src/db/pool.ts                45  0.91   1w ago      db,pool
 3  tests/db/connection.test.ts   12  0.85   3d ago      db,test

 5 of 23 results | [Enter] expand | [/] filter | [n] next page
```

**Pattern 6 — Structured data output** (jq, httpie). Pretty-printed JSON/YAML with indentation and syntax coloring, used when piping to other tools. Every well-designed CLI should auto-detect TTY versus pipe and switch between human-readable and machine-readable output accordingly.

A universal design principle emerges: **all excellent CLI tools default to human-friendly output on TTY and machine-friendly output when piped**. The `NO_COLOR` env var and `--color=auto|always|never` flag are now standard conventions.

---

## Building with Rich: components, patterns, and responsive layouts

The Python **`rich` library** (55K+ GitHub stars) provides composable renderables that nest freely — the "Lego brick" approach. The key insight is matching Rich components to specific display needs.

**For list views**, `rich.table.Table` auto-resizes columns to fit terminal width. Use `ratio` for proportional columns and `no_wrap=True` for titles. Colored badges are created with console markup: `[bold white on blue] tag [/]`. Score badges can use conditional coloring — **green for high** (>0.8), **yellow for medium** (0.5–0.8), **red for low** (<0.5). Row styles like `row_styles=["", "dim"]` create zebra striping for scanability.

**For detail views**, `rich.panel.Panel` creates bordered cards with title and subtitle. Nest any renderable inside — tables, syntax-highlighted code via `rich.syntax.Syntax`, markdown via `rich.markdown.Markdown`, or compose multiple elements with `rich.console.group()`. The `Panel.fit()` method auto-sizes to content, while `Panel(expand=True)` fills the terminal width.

**For hierarchical/categorized results**, `rich.tree.Tree` provides visual hierarchy with guide lines — perfect for results grouped by project, tag, or category:

```
🔍 Search Results
├── Python (5 results)
│   ├── Getting Started with Rich — Score: 95%
│   └── Advanced CLI Patterns — Score: 82%
└── Design (3 results)
    └── Terminal UX Best Practices — Score: 78%
```

**For streaming results**, `rich.live.Live` provides flicker-free live updates. Combine with `Layout` for dashboard-style multi-pane displays. The `Live` context manager supports printing above the live display and alternate-screen mode for fullscreen applications.

A critical responsive pattern uses `Console.width` for adaptive layouts:

```python
console = Console()
if console.width >= 120:
    # Wide: multi-column card layout with Columns
    console.print(Columns(cards, equal=True))
elif console.width >= 80:
    # Medium: table layout with full metadata
    console.print(make_table(results))
else:
    # Narrow: stacked compact cards, heavy truncation
    for r in results:
        console.print(make_compact_card(r))
```

Rich auto-detects terminal capabilities (color depth, width, interactivity) and degrades gracefully. When piping to a file, it strips ANSI control codes automatically. Custom themes centralize styling decisions:

```python
from rich.theme import Theme
custom_theme = Theme({
    "result.title": "bold cyan",
    "result.score.high": "bold green",
    "result.score.low": "red",
    "tag": "bold white on dark_blue",
    "meta": "dim italic",
})
```

Notable open-source projects demonstrating Rich's capabilities include **rich-cli** (file viewing with syntax highlighting), **Memray** (Bloomberg's memory profiler), **Toolong** (log file viewer), and **ghtop** (GitHub activity dashboard).

---

## Command hierarchy: the search → filter → drill-down workflow

The most effective knowledge base CLI tools follow the **`APPNAME VERB NOUN --ADJECTIVE`** pattern (Cobra convention). Research across `nb`, `kb`, `tldr`, `cheat.sh`, and `howdoi` reveals a clear ideal command structure.

**Core subcommands** that every knowledge CLI needs: `search` (full-text with filters), `show`/`view` (single item detail), `list` (browse/filter), `tags` (list all tags with counts), `stats` (KB statistics and visualizations), and `explore` (interactive browsing mode). Single-letter aliases like `s`, `q`, `l` accelerate power users — `nb`'s approach of mapping `a`/`e`/`s`/`q`/`d` to add/edit/show/search/delete is proven effective.

The **progressive refinement** workflow is the heart of knowledge search:

```bash
# Stage 1: Broad search
kb search "python async"

# Stage 2: Filter results
kb search "python async" --tag tutorial --after 2024-01-01 --sort score

# Stage 3: Drill into specific result
kb show 42 --related

# Interactive mode (combines all stages)
kb explore "python async"
```

**Standard flags** should be consistent across subcommands. Filtering uses `--tag/-t` (repeatable), `--category/-c`, `--after`/`--before` for dates, `--sort` (relevance|date|title), and `--limit/-n`. Output control uses `--json/-j`, `--format` (human|json|csv|plain), `--verbose/-v`, and `--quiet/-q`. The `--interactive/-i` flag or dedicated `explore` command enables fzf-style fuzzy browsing.

**Output format switching** should be automatic: detect `sys.stdout.isatty()` and emit rich colored output for humans, JSON when piped. Explicit `--json` overrides this. This pattern, used by `gh`, `kubectl`, and `nb`, is the gold standard.

The **`nb` knowledge base tool** stands out as the most comprehensive reference implementation with **selector syntax** (`notebook:item`), hierarchical tags (`#project/design/ui`), flexible identifiers (by ID, filename, title, or `--last`), and piped content support (`echo "content" | nb add`).

---

## Interactive patterns that avoid full TUI complexity

The most effective interactive CLIs compose simple elements rather than building monolithic TUI applications. Three tools define the landscape.

**fzf** is the canonical interactive filter. Its genius lies in operating as a Unix filter — it takes input on stdin, provides fuzzy selection, and outputs the selected item on stdout. Key patterns include `--preview 'bat --color=always {}'` for live content preview, `--height 50%` for partial-screen operation (avoiding full TUI), and composable keybindings via `--bind 'ctrl-e:execute(edit {})'`. The `--layout=reverse` flag places the list at top with input at bottom, which feels more natural.

**gum** (Charmbracelet) provides atomic interactive elements that compose in shell scripts. `gum filter` provides fzf-like fuzzy filtering, `gum choose` offers selection lists, `gum input` handles text entry, `gum confirm` adds yes/no dialogs, and `gum spin` wraps long operations with spinners. The real power is composition — chaining these into drill-down workflows:

```bash
QUERY=$(gum input --placeholder "Search...")
SELECTED=$(kb search "$QUERY" --format plain | gum filter)
ACTION=$(gum choose "View" "Edit" "Copy" "Delete")
```

**For Python**, the recommended interactive libraries are:

- **`iterfzf`** — Bundles the fzf binary, streams items lazily, supports preview and multi-select. Best UX, but depends on fzf binary
- **`InquirerPy`** — Most actively maintained Inquirer port with a **fuzzy prompt** that mimics fzf. Built on prompt_toolkit. Pure Python
- **`questionary`** — Clean API for select/text/confirm/checkbox. Also built on prompt_toolkit
- **`prompt_toolkit`** — Low-level foundation with FuzzyCompleter, keybindings, and auto-suggestions

The **recommended stack** for a Python knowledge CLI combines `click` or `typer` for command structure, `rich` for output formatting (not interactive), and `iterfzf` or `InquirerPy` for interactive selection. This separation of concerns — framework for commands, library for display, tool for interaction — produces cleaner architecture than any monolithic TUI approach.

The drill-down pattern across all these tools follows three stages: **list view** (scored, filterable results) → **preview** (inline or side-pane content preview) → **detail view** (full content with metadata and actions). Standard keyboard conventions include `↑/↓` for navigation, `Enter` for selection, `Tab` for multi-select, `Ctrl-/` for toggling preview, and `Esc` for cancellation.

---

## Terminal visualization without a TUI framework

Activity graphs, score distributions, and per-project breakdowns are achievable in pure terminal output. **`plotext`** (2.1K GitHub stars) is the gold standard for Python terminal plotting — it supports bar charts, heatmaps, histograms, and scatter plots with matplotlib-like syntax, **zero dependencies**, and official Rich integration.

**Sparklines** are the highest-density visualization for inline use. A 10-line Python function using Unicode block characters (`▁▂▃▄▅▆▇█`) creates them:

```python
def sparkline(values):
    chars = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    return "".join(chars[min(7, int((v - mn) / rng * 7))] for v in values)
```

These embed directly in Rich tables for trend columns:

```
Project     Notes  7-day trend   Avg Score
research      42   ▃▅▇█▆▄▇      0.82
work          28   ▅▃▆▇▅▄▃      0.74
personal      15   ▁▃▂▅▇▃▆      0.69
```

**GitHub-style activity heatmaps** use Unicode block characters (`░▒▓█`) or colored spaces. `termgraph` has a built-in `--calendar` mode, and `plotext`'s `matrix_plot` handles arbitrary heatmap data. For embedding in Rich layouts, `termcharts` provides bar and pie charts that accept a `rich=True` parameter and return Rich-compatible renderables.

**Horizontal bar charts** are the most effective terminal visualization for category breakdowns:

```
Notes by Tag
  python     ████████████████████ 42
  research   █████████████▌       28
  terminal   ███████▌             15
  ux         ████▌                 9
```

The recommended visualization stack: **plotext** for full charts, **custom sparkline function** for inline trends, **termcharts** for Rich-integrated charts, and **Rich tables with embedded Unicode** for composite displays.

---

## Prioritizing metadata when every character counts

Terminal space is fundamentally constrained. Research from clig.dev, Nielsen Norman Group's progressive disclosure principles, and Algolia's search UX guidelines converge on a **four-level metadata priority framework**.

**Level 1 — Always visible** (works at 60 columns): title/name (truncated with ellipsis), relevance score (compact format like `0.92` or `★★★★☆`), and type indicator (single icon or character code).

**Level 2 — Important** (shown at 80+ columns): relative date (`2d ago`), primary tags (top 2–3, with `+N` overflow indicator), and source/project identifier.

**Level 3 — Contextual** (shown at 120+ columns or with `--verbose`): full tag list, summary snippet (first 80 characters), file path, and word count.

**Level 4 — Detail** (shown on item selection): full content preview, all metadata fields, related items, and edit history.

**Graceful degradation** follows specific truncation patterns. Dates compress from `2025-02-15 14:30:00` → `Feb 15, 2025` → `2d ago` → `2d`. Tags compress from full list → `#python #viz +3` → `#python +4`. Titles truncate with trailing ellipsis. Paths use middle truncation: `/home/user/.../project/file.md`.

Here is the same search displayed at three terminal widths:

**Wide (120 columns)** — full table with all metadata columns:
```
 Score  Title                                        Date       Tags                   Project
 0.95   Plotext terminal charts research notes        Feb 15     #python #plotext #cli   research
 0.87   Rich library visualization capabilities       Feb 12     #python #rich #tui      cli-tools
```

**Standard (80 columns)** — two-line format, tags truncated with overflow count:
```
 ★★★★★ Plotext terminal charts research notes
       Feb 15 · #python #plotext +1 · research
 ★★★★☆ Rich library visualization capabilities
       Feb 12 · #python #rich +1 · cli-tools
```

**Narrow (60 columns)** — single-line with heavy truncation:
```
.95 Plotext terminal charts rese…    2d ago
.87 Rich library visualization …    5d ago
```

Tools like `kubectl` use the `-o wide` flag for explicit width expansion. `eza` auto-calculates optimal grid columns from terminal width. `docker ps` truncates container IDs to 12 characters and commands with ellipsis. **The pattern is consistent: fixed-width left columns for identifiers, elastic right columns for descriptions, and explicit flags for progressively wider output.**

---

## Conclusion: a concrete blueprint

The optimal architecture for a CLI knowledge search tool combines **five layers**: `click`/`typer` for command structure with `VERB NOUN --ADJECTIVE` conventions; `rich` for all output formatting using Tables (list view), Panels (detail view), and Trees (categorized view); `iterfzf` or `InquirerPy` for interactive selection without TUI overhead; `plotext` with inline sparklines for terminal visualization; and width-aware rendering at three breakpoints (60/80/120 columns).

The most important design insight across all researched tools is that **composition beats monolithism**. ripgrep pipes to fzf which pipes to bat. gum chains atomic interactive elements. Rich nests renderables inside renderables. The tools that achieve the best UX are those that do one thing excellently and compose with others — and the best knowledge CLIs will follow the same principle, offering `--json` output for piping, `--interactive` mode for exploration, and rich formatted output as the human-friendly default.