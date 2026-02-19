#!/usr/bin/env python3
"""Generate a communication style card from BrainLayer conversation data.

Pulls user messages from the database, samples across all sources,
runs semantic style analysis, and saves the results.

Usage:
    python3 scripts/generate_style_card.py [--db PATH] [--output-dir DIR] [--sample-size N]

Output is saved to the specified directory (default: ~/.local/share/brainlayer/storage/).
The output contains personal communication patterns — do NOT commit it.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development installs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brainlayer.vector_store import VectorStore


def get_user_messages(db_path: Path, min_chars: int = 10, sample_size: int = 20000) -> list[str]:
    """Pull user messages from DB, filter noise, sample."""
    import apsw

    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    cursor = conn.cursor()

    # Get user messages (sender = 'human' or content_type = 'user_message')
    rows = list(cursor.execute("""
        SELECT content, source, char_count
        FROM chunks
        WHERE (sender = 'human' OR content_type = 'user_message')
          AND char_count >= ?
        ORDER BY RANDOM()
        LIMIT ?
    """, [min_chars, sample_size * 2]))  # Over-sample, then dedupe

    conn.close()

    # Dedupe and clean
    seen = set()
    messages = []
    for content, source, char_count in rows:
        text = content.strip()
        if text and text not in seen:
            seen.add(text)
            messages.append(text)
            if len(messages) >= sample_size:
                break

    return messages


def main():
    parser = argparse.ArgumentParser(description="Generate communication style card from BrainLayer data")
    parser.add_argument("--db", type=Path, default=Path.home() / ".local/share/brainlayer/brainlayer.db",
                        help="Path to BrainLayer database")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: ~/.local/share/brainlayer/storage/style/)")
    parser.add_argument("--sample-size", type=int, default=20000,
                        help="Number of messages to sample (default: 20000)")
    parser.add_argument("--min-chars", type=int, default=10,
                        help="Minimum message length in chars (default: 10)")
    args = parser.parse_args()

    if not args.db.exists():
        # Try legacy path
        legacy_db = Path.home() / ".local/share/zikaron/zikaron.db"
        if legacy_db.exists():
            args.db = legacy_db
            print(f"Using legacy database: {args.db}")
        else:
            print(f"Database not found: {args.db}")
            sys.exit(1)

    output_dir = args.output_dir or Path.home() / ".local/share/brainlayer/storage/style"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Database: {args.db}")
    print(f"Output: {output_dir}")
    print(f"Sample size: {args.sample_size}")

    # Pull messages
    print("\nPulling user messages...")
    messages = get_user_messages(args.db, min_chars=args.min_chars, sample_size=args.sample_size)
    print(f"  Got {len(messages)} unique messages")

    if len(messages) < 100:
        print("Too few messages for meaningful analysis. Need at least 100.")
        sys.exit(1)

    # Source distribution
    import apsw
    conn = apsw.Connection(str(args.db), flags=apsw.SQLITE_OPEN_READONLY)
    source_counts = dict(conn.cursor().execute("""
        SELECT COALESCE(source, 'unknown'), COUNT(*)
        FROM chunks
        WHERE sender = 'human' OR content_type = 'user_message'
        GROUP BY source
        ORDER BY COUNT(*) DESC
    """))
    conn.close()
    print("\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    # Run analysis
    print("\nRunning semantic style analysis...")
    try:
        from brainlayer.pipeline.semantic_style import analyze_semantic_style
        analysis = analyze_semantic_style(messages, output_dir=output_dir)
        print(f"\nStyle card saved to: {output_dir}")
        print(f"Topics found: {len(analysis.topic_clusters)}")
        for name, cluster in analysis.topic_clusters.items():
            print(f"  {name}: {cluster.message_count} msgs, formality={cluster.formality:.2f}")
    except ImportError as e:
        print(f"\nCould not import style analyzer: {e}")
        print("Make sure brainlayer is installed with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        print("This usually means Ollama isn't running (needed for embeddings).")
        sys.exit(1)


if __name__ == "__main__":
    main()
