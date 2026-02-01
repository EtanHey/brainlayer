#!/usr/bin/env python3
"""
Zikaron Watcher - Auto-index new Claude conversations.

Watches ~/.claude/projects/ for new .jsonl files and triggers indexing.
Run as launchd agent for always-on indexing.
"""

import subprocess
import sys
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
DEBOUNCE_SECONDS = 30  # Wait before indexing to let conversation finish
INDEX_COMMAND = ["zikaron", "index"]

# Track pending files to debounce
pending_files: dict[Path, float] = {}


class ConversationHandler(FileSystemEventHandler):
    """Handle new/modified conversation files."""

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
        if event.src_path.endswith(".jsonl"):
            self.queue_index(Path(event.src_path))

    def on_modified(self, event: FileModifiedEvent):
        if event.is_directory:
            return
        if event.src_path.endswith(".jsonl"):
            self.queue_index(Path(event.src_path))

    def queue_index(self, path: Path):
        """Queue file for indexing with debounce."""
        pending_files[path] = time.time()
        print(f"[Watcher] Queued: {path.name}")


def process_pending():
    """Process files that have been stable for DEBOUNCE_SECONDS."""
    now = time.time()
    ready = [p for p, t in pending_files.items() if now - t >= DEBOUNCE_SECONDS]

    if ready:
        print(f"[Watcher] Indexing {len(ready)} conversation(s)...")
        for path in ready:
            del pending_files[path]

        try:
            result = subprocess.run(
                INDEX_COMMAND,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
            )
            if result.returncode == 0:
                print(f"[Watcher] ✓ Indexed successfully")
            else:
                print(f"[Watcher] ✗ Index failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("[Watcher] ✗ Index timed out")
        except Exception as e:
            print(f"[Watcher] ✗ Error: {e}")


def main():
    """Start the watcher."""
    if not CLAUDE_PROJECTS.exists():
        print(f"[Watcher] Error: {CLAUDE_PROJECTS} not found")
        sys.exit(1)

    print(f"[Watcher] Watching: {CLAUDE_PROJECTS}")
    print(f"[Watcher] Debounce: {DEBOUNCE_SECONDS}s")
    print(f"[Watcher] Press Ctrl+C to stop\n")

    handler = ConversationHandler()
    observer = Observer()
    observer.schedule(handler, str(CLAUDE_PROJECTS), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(5)
            process_pending()
    except KeyboardInterrupt:
        print("\n[Watcher] Stopping...")
        observer.stop()

    observer.join()
    print("[Watcher] Done.")


if __name__ == "__main__":
    main()
