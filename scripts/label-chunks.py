#!/usr/bin/env python3
"""Interactive Rich TUI for labeling chunks.

Shows chunk content + pre-labels. User confirms or corrects.
Saves progress after each chunk (resume-able).

Usage:
    python scripts/label-chunks.py
    python scripts/label-chunks.py --start-from 50  # Resume from chunk 50
"""

import json
import sys
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

DATA_FILE = Path(__file__).parent.parent / "data" / "labeled-samples.json"
TAXONOMY_FILE = Path(__file__).parent.parent / "src" / "zikaron" / "taxonomy.json"

console = Console()


def load_taxonomy() -> dict[str, str]:
    with open(TAXONOMY_FILE) as f:
        data = json.load(f)
    labels = {}
    for cat, info in data["categories"].items():
        for label, desc in info["labels"].items():
            labels[label] = desc
    return labels


def load_samples() -> list[dict]:
    with open(DATA_FILE) as f:
        return json.load(f)


def save_samples(samples: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def render_chunk(sample: dict, idx: int, total: int, taxonomy: dict):
    """Render a chunk for review."""
    console.clear()

    # Header
    reviewed = sum(1 for s in load_samples() if s.get("reviewed"))
    console.print(f"[bold blue]Zikaron Label Studio[/] — Chunk {idx+1}/{total} ({reviewed} reviewed)")
    console.print()

    # Chunk metadata
    meta_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    meta_table.add_column("Key", style="bold cyan")
    meta_table.add_column("Value")
    meta_table.add_row("Project", sample.get("project", "unknown"))
    meta_table.add_row("Type", sample.get("content_type", "unknown"))
    meta_table.add_row("Source", str(sample.get("source_file", ""))[-60:])
    console.print(Panel(meta_table, title="Metadata", box=box.ROUNDED))

    # Content
    content = sample.get("content", "")[:1200]
    if not content.strip():
        console.print(Panel("[bold red]⚠ EMPTY CONTENT[/] — use 'meta/empty' or 's' to skip",
                            title="Content", box=box.ROUNDED, expand=True))
    elif len(content.strip()) < 20:
        console.print(Panel(content + "\n\n[dim](very short — consider meta/empty or meta/noise)[/]",
                            title="Content", box=box.ROUNDED, expand=True))
    else:
        console.print(Panel(content, title="Content", box=box.ROUNDED, expand=True))

    # Pre-labels
    pre_labels = sample.get("pre_labels", [])
    if pre_labels:
        label_text = Text()
        for pl in pre_labels:
            conf = pl["confidence"]
            style = "green" if conf >= 0.8 else "yellow" if conf >= 0.6 else "red"
            label_text.append(f"  {pl['label']}", style=style)
            desc = taxonomy.get(pl["label"], "")
            label_text.append(f" ({conf:.0%}) — {desc}\n", style="dim")
        console.print(Panel(label_text, title="Pre-labels", box=box.ROUNDED))
    else:
        console.print("[dim]No pre-labels for this chunk[/]")

    # Human labels (if already reviewed)
    human = sample.get("human_labels", [])
    if human:
        console.print(f"[green]Current human labels: {', '.join(human)}[/]")


def get_label_input(taxonomy: dict, pre_labels: list[dict]) -> list[str]:
    """Get label input from user with fuzzy completion."""
    all_labels = list(taxonomy.keys())
    completer = FuzzyWordCompleter(all_labels)

    console.print()
    console.print("[bold]Commands:[/]")
    console.print("  [cyan]Enter[/] = accept pre-labels as-is")
    console.print("  [cyan]label1 label2 ...[/] = set labels (Tab=accept autocomplete, Enter=submit)")
    console.print("  [cyan]+ label[/] = add to pre-labels")
    console.print("  [cyan]- label[/] = remove from pre-labels")
    console.print("  [cyan]s[/] = skip (mark as reviewed without labels)")
    console.print("  [cyan]b[/] = back (go to previous chunk)")
    console.print("  [cyan]q[/] = quit and save")
    console.print()

    pre_label_names = [pl["label"] for pl in pre_labels]

    try:
        user_input = prompt(
            "Labels> ",
            completer=completer,
            complete_while_typing=True,
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None  # Signal to quit

    if not user_input:
        # Accept pre-labels
        return pre_label_names if pre_label_names else []

    if user_input.lower() == "q":
        return None  # Quit

    if user_input.lower() == "s":
        return []  # Skip

    if user_input.lower() == "b":
        return "BACK"  # Go back

    # Parse additions/removals
    labels = list(pre_label_names)

    if user_input.startswith("+"):
        # Add labels
        new_labels = user_input[1:].strip().split()
        for nl in new_labels:
            # Fuzzy match
            matched = _fuzzy_match(nl, all_labels)
            if matched and matched not in labels:
                labels.append(matched)
                console.print(f"  [green]+ {matched}[/]")
            elif not matched:
                console.print(f"  [red]? {nl} (not found)[/]")
        return labels

    if user_input.startswith("-"):
        # Remove labels
        rm_labels = user_input[1:].strip().split()
        for rl in rm_labels:
            matched = _fuzzy_match(rl, labels)
            if matched:
                labels.remove(matched)
                console.print(f"  [red]- {matched}[/]")
        return labels

    # Full replacement
    new_labels = []
    for token in user_input.split():
        matched = _fuzzy_match(token, all_labels)
        if matched:
            new_labels.append(matched)
        else:
            console.print(f"  [red]? {token} (not found)[/]")
    return new_labels


def _fuzzy_match(query: str, options: list[str]) -> str | None:
    """Simple fuzzy match — exact, prefix, or substring."""
    q = query.lower()
    # Exact
    for opt in options:
        if opt.lower() == q:
            return opt
    # Prefix
    for opt in options:
        if opt.lower().startswith(q):
            return opt
    # Substring
    for opt in options:
        if q in opt.lower():
            return opt
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0, help="Resume from chunk N")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        console.print("[red]No labeled-samples.json found. Run pre-label.py first.[/]")
        sys.exit(1)

    taxonomy = load_taxonomy()
    samples = load_samples()

    total = len(samples)
    reviewed = sum(1 for s in samples if s.get("reviewed"))
    console.print(f"[bold blue]Zikaron Label Studio[/]")
    console.print(f"Samples: {total} | Reviewed: {reviewed} | Remaining: {total - reviewed}")
    console.print(f"Taxonomy: {len(taxonomy)} labels\n")

    # Find start position
    start = args.start_from
    if start == 0 and reviewed > 0:
        # Auto-resume from first unreviewed
        for i, s in enumerate(samples):
            if not s.get("reviewed"):
                start = i
                break
        console.print(f"[dim]Resuming from chunk {start + 1}[/]")

    idx = start
    while idx < total:
        sample = samples[idx]

        if sample.get("reviewed") and args.start_from == 0:
            idx += 1
            continue  # Skip already reviewed unless explicit start

        render_chunk(sample, idx, total, taxonomy)
        labels = get_label_input(taxonomy, sample.get("pre_labels", []))

        if labels is None:
            # Quit
            save_samples(samples)
            console.print(f"\n[bold green]Saved.[/] {sum(1 for s in samples if s.get('reviewed'))}/{total} reviewed.")
            return

        if labels == "BACK":
            # Go back to previous chunk
            if idx > 0:
                # Find previous reviewed chunk to re-do
                idx -= 1
                while idx > 0 and not samples[idx].get("reviewed"):
                    idx -= 1
                samples[idx].pop("reviewed", None)
                samples[idx].pop("human_labels", None)
                save_samples(samples)
                console.print(f"[yellow]Going back to chunk {idx + 1}[/]")
            continue

        sample["human_labels"] = labels
        sample["reviewed"] = True

        # Show what was saved
        if labels:
            console.print(f"  [green]Saved: {', '.join(labels)}[/]")
        else:
            console.print(f"  [dim]Saved: (no labels)[/]")

        idx += 1

        # Save after each chunk
        save_samples(samples)

    console.print(f"\n[bold green]All {total} chunks reviewed![/]")
    console.print(f"Next: python scripts/train-setfit.py")


if __name__ == "__main__":
    main()
