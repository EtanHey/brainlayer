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

# Force unbuffered output for launchd logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
DEBOUNCE_SECONDS = 30  # Wait before indexing to let conversation finish

# Use full path to venv zikaron CLI (launchd has minimal PATH)
ZIKARON_VENV = Path.home() / "Gits" / "golems" / "packages" / "zikaron" / ".venv"
INDEX_COMMAND = [str(ZIKARON_VENV / "bin" / "zikaron"), "index-fast"]

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
            print(f"[Watcher] Running: {' '.join(INDEX_COMMAND)}")
            result = subprocess.run(
                INDEX_COMMAND,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
                env={**dict(__import__('os').environ), "PYTHONUNBUFFERED": "1"},
            )
            if result.returncode == 0:
                # Extract chunk count from output if present
                output = result.stdout.strip()
                if "chunks" in output.lower():
                    print(f"[Watcher] ✓ {output.split(chr(10))[-1]}")
                else:
                    print(f"[Watcher] ✓ Indexed successfully")
            else:
                print(f"[Watcher] ✗ Index failed (exit {result.returncode})")
                if result.stderr:
                    print(f"[Watcher] stderr: {result.stderr[:500]}")
                if result.stdout:
                    print(f"[Watcher] stdout: {result.stdout[:500]}")
        except subprocess.TimeoutExpired:
            print("[Watcher] ✗ Index timed out (5 min)")
        except FileNotFoundError as e:
            print(f"[Watcher] ✗ Command not found: {e}")
        except Exception as e:
            print(f"[Watcher] ✗ Error: {type(e).__name__}: {e}")


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
