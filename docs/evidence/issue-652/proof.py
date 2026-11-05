#!/usr/bin/env python3
"""Issue #652 source-boundary proof (read-only, synthetic, no DB, no Swift toolchain).

Run from the repo root:  python3 -I docs/evidence/issue-652/proof.py
Reads Swift source text only. Prints PASS/FAIL per claim and exits 1 if any
claim about current source no longer holds (meaning this report is stale).
"""
import pathlib
import re
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "brain-bar" / "Sources"
CONT = SRC / "BrainBar" / "InjectionContinuation.swift"
DB = SRC / "BrainBar" / "BrainDatabase.swift"
results = []


def claim(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))


# BrainBarDaemon/ symlinks many files into BrainBar/; dedupe by resolved path.
swift = {}
for _p in sorted(SRC.rglob("*.swift")):
    if not _p.is_symlink():
        swift[_p] = _p.read_text()

# C1: resumeCommand has no production caller. The "copy to continue" UI action is absent.
callers = [str(p.relative_to(ROOT)) for p, t in swift.items()
           if "resumeCommand(" in t and p != CONT]
claim("C1 no production caller of InjectionContinuation.resumeCommand", callers == [], f"callers={callers}")

# C2: the only pasteboard writer is BrainBarWorkspace.copy, and nothing passes it a resume command.
pb = [str(p.relative_to(ROOT)) for p, t in swift.items() if "NSPasteboard.general.setString" in t]
claim("C2 the only pasteboard writer is BrainBarWorkspace.swift", pb == ["brain-bar/Sources/BrainBar/BrainBarWorkspace.swift"], f"writers={pb}")

# C3: no --teleport, cse_ or T3 thread-link emitter anywhere in BrainBar.
tele = [str(p.relative_to(ROOT)) for p, t in swift.items() if re.search(r"--teleport|cse_|t3://|T3 thread", t)]
claim("C3 no cloud --teleport / cse_ / T3 link emitter in BrainBar sources", tele == [], f"hits={tele}")

# C4: when the conversation ID is not a UUID, resumeCommand's fallback emits `claude --resume <fallbackSessionID>`
# (a BrainLayer-namespace ID) unless it is empty or unsafe.
ct = CONT.read_text()
claim("C4 latent fallback emits `claude --resume <fallback>` for a non-UUID ID",
      'command: "claude --resume \\(fallback)"' in ct)

# C5: the injection_events SELECT does not project claude_conversation_id, so event.claudeConversationID
# is always "", and the resumable ID always comes from the INJECTED CHUNKS.
dt = DB.read_text()
m = re.search(r"SELECT id, session_id, timestamp, query, chunk_ids, token_count,(.*?)FROM injection_events", dt, re.S)
claim("C5 injection_events SELECT omits claude_conversation_id", bool(m) and "claude_conversation_id" not in m.group(1))

# C6: resumableClaudeConversationID falls back to the chunks' own conversation and source_file UUIDs.
fn = re.search(r"func resumableClaudeConversationID\(.*?\n    \}\n", dt, re.S)
body = fn.group(0) if fn else ""
claim("C6 resumable ID is taken from chunk.claudeConversationID or chunk.sourceFile",
      "chunk.claudeConversationID" in body and "extractClaudeConversationID(fromSourceFile: chunk.sourceFile)" in body)


# Synthetic fixture: a Python model of the Swift selection logic (C5 + C6), fed invented IDs only.
def is_uuid(v):
    try:
        uuid.UUID(v)
        return len(v) == 36
    except ValueError:
        return False


def resumable(event_cid, chunks):
    if is_uuid(event_cid):
        return event_cid
    for c in chunks:
        if is_uuid(c["claude_conversation_id"]):
            return c["claude_conversation_id"]
    for c in chunks:
        stem = pathlib.PurePosixPath(c["source_file"]).stem
        if is_uuid(stem):
            return stem
    return ""


RECIPIENT = "11111111-1111-4111-8111-111111111111"   # session that received the injection
MEMORY_SRC = "22222222-2222-4222-8222-222222222222"  # conversation the injected memory came from
event_cid = ""  # C5: never projected, so always empty
chunks = [{"claude_conversation_id": "",
           "source_file": f"/Users/x/.claude/projects/-Users-x-Gits-other/{MEMORY_SRC}.jsonl"}]
got = resumable(event_cid, chunks)
claim("C7 synthetic: resumable ID is the MEMORY SOURCE conversation, not the recipient session",
      got == MEMORY_SRC and got != RECIPIENT, f"got={got} recipient={RECIPIENT}")

print(f"\n{sum(results)}/{len(results)} claims hold on this tree")
sys.exit(0 if all(results) else 1)
