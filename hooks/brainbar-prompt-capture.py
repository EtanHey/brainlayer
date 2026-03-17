#!/usr/bin/env python3
"""UserPromptSubmit hook — captures user prompt for pairing with assistant response.

Reads prompt from stdin (Claude Code hook payload), stores in a lightweight
temp file keyed by session_id. The Stop hook reads this file to pair
prompt + response before indexing.

Target latency: <5ms.
"""

import json
import sys
from pathlib import Path

PENDING_DIR = Path.home() / ".brainlayer" / "pending"


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    session_id = payload.get("session_id", "")
    # UserPromptSubmit provides the user's input in the payload
    user_input = payload.get("user_message") or payload.get("prompt", "")
    if not session_id or not user_input:
        return

    # Write to pending dir (fast filesystem write)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    pending_file = PENDING_DIR / f"{session_id}.txt"
    pending_file.write_text(user_input[:10000])  # Cap at 10K chars


if __name__ == "__main__":
    main()
