"""Latch the real cache reader and check the actual owner's close/borrow methods."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def method(source, name):
    start = source.index("    func " + name)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--before-ref", required=True)
a = p.parse_args()
assert re.fullmatch("[0-9a-f]{40}", a.before_ref)
root = Path(__file__).resolve().parents[4]
out = root / ".build.noindex/entity-cache-lifetime-proof"
out.mkdir(parents=True, exist_ok=True)
for case in ("before", "after"):

    def source(path):
        if case == "before":
            return subprocess.check_output(["git", "show", a.before_ref + ":" + path], cwd=root, text=True)
        return (root / path).read_text()

    owner = source("brain-bar/Sources/BrainBar/BrainDatabase.swift")
    cache = source("brain-bar/Sources/BrainBar/KnowledgeGraph/EntityCache.swift")
    anchor = "        let columns = Self.tableColumns(db)"
    assert cache.count(anchor) == 1
    cache = cache.replace(anchor, "        probeEntered.signal(); probeRelease.wait()\n" + anchor)
    borrow = method(owner, "withSQLiteHandle<") if case == "after" else ""
    call = "cache.load(from: db" + (".dbHandle" if case == "before" else "") + ")"
    program = (
        """
import Foundation
import SQLite3
let probeEntered = DispatchSemaphore(value: 0)
let probeRelease = DispatchSemaphore(value: 0)
enum DBError: Error { case notOpen }
final class BrainDatabase: @unchecked Sendable {
    private var db: OpaquePointer?
    private let transactionLock = NSRecursiveLock()
    var openedFileIdentity: Int?
    var isOpen = true
    var dbHandle: OpaquePointer? { db }
    init() {
        precondition(sqlite3_open(":memory:", &db) == SQLITE_OK)
        precondition(sqlite3_exec(db, "CREATE TABLE kg_entities(id TEXT,name TEXT,entity_type TEXT,status TEXT,user_verified INTEGER)", nil,nil,nil) == SQLITE_OK)
    }
"""
        + method(owner, "close()")
        + "\n"
        + borrow
        + "\n}\n"
        + cache
        + """
let db = BrainDatabase()
let cache = EntityCache()
let readFinished = DispatchSemaphore(value: 0)
let closeStarted = DispatchSemaphore(value: 0)
let closeFinished = DispatchSemaphore(value: 0)
DispatchQueue.global().async { """
        + call
        + """; readFinished.signal() }
precondition(probeEntered.wait(timeout: .now()+5) == .success)
DispatchQueue.global().async { closeStarted.signal(); db.close(); closeFinished.signal() }
precondition(closeStarted.wait(timeout: .now()+5) == .success)
"""
    )
    if case == "before":
        program += """
precondition(closeFinished.wait(timeout: .now()+5) == .success)
print("owner_closed_while_reader_latched=true")
exit(1)
"""
    else:
        program += """
precondition(closeFinished.wait(timeout: .now()+0.1) == .timedOut)
probeRelease.signal()
precondition(readFinished.wait(timeout: .now()+5) == .success)
precondition(closeFinished.wait(timeout: .now()+5) == .success)
print("owner_closed_while_reader_latched=false")
exit(0)
"""
    path = out / (case + ".swift")
    path.write_text(program)
    binary = path.with_suffix("")
    subprocess.run(["swiftc", str(path), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    receipt = {"case": case, "rc": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr}
    print(json.dumps(receipt), flush=True)
    assert result.returncode == (1 if case == "before" else 0), receipt
    assert result.stdout.strip() == "owner_closed_while_reader_latched=" + str(case == "before").lower(), receipt
