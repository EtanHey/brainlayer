import Darwin
import Foundation
import XCTest
@testable import BrainBar

final class DeferredStoreTelemetryTests: XCTestCase {
    private func withLog(_ body: (URL) throws -> Void) throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent("brainbar-telemetry-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let old = ProcessInfo.processInfo.environment["BRAINLAYER_DEFERRED_TELEMETRY_PATH"]
        setenv("BRAINLAYER_DEFERRED_TELEMETRY_PATH", root.appendingPathComponent("events.jsonl").path, 1)
        defer {
            DeferredStoreTelemetry.waitForWritesForTesting()
            if let old { setenv("BRAINLAYER_DEFERRED_TELEMETRY_PATH", old, 1) }
            else { unsetenv("BRAINLAYER_DEFERRED_TELEMETRY_PATH") }
            try? FileManager.default.removeItem(at: root)
        }
        try body(root)
    }

    func testContendedSinkDropsEventAndWorkerRecoversBeforeLockRelease() throws {
        try withLog { root in
            let log = root.appendingPathComponent("events.jsonl")
            let fd = Darwin.open(log.path + ".lock", O_CREAT | O_RDWR, 0o600)
            XCTAssertGreaterThanOrEqual(fd, 0)
            guard fd >= 0 else { return }
            defer { Darwin.close(fd) }
            XCTAssertEqual(flock(fd, LOCK_EX | LOCK_NB), 0)
            DeferredStoreTelemetry.record("queued", dbPath: "unused", queueID: "blocked", chunkID: "blocked",
                                          origin: nil, queuedAt: nil)
            let drained = DispatchSemaphore(value: 0)
            DispatchQueue.global().async {
                DeferredStoreTelemetry.waitForWritesForTesting()
                drained.signal()
            }
            let result = drained.wait(timeout: .now() + 2)
            flock(fd, LOCK_UN)
            DeferredStoreTelemetry.waitForWritesForTesting()
            XCTAssertEqual(result, .success, "sink worker must finish while another owner holds its lock")
            DeferredStoreTelemetry.record("queued", dbPath: "unused", queueID: "recovered", chunkID: "recovered",
                                          origin: nil, queuedAt: nil)
            DeferredStoreTelemetry.waitForWritesForTesting()
            let text = try String(contentsOf: log, encoding: .utf8)
            XCTAssertFalse(text.contains("blocked"))
            XCTAssertTrue(text.contains("recovered"))
        }
    }

    func testDuplicateReplayReturnsCanonicalChunkWithoutClaimingNewCommit() throws {
        try withLog { root in
            let db = BrainDatabase(path: root.appendingPathComponent("test.db").path)
            defer { db.close() }
            let content = "Only the canonical existing memory should survive a duplicate deferred replay."
            let existing = try db.store(content: content, tags: [], importance: 5, source: "mcp",
                                        chunkID: "canonical", conversationID: "same-conversation")
            let queued = try db.queuePendingStore(content: content, tags: [], importance: 5, source: "mcp",
                                                  chunkID: "never-inserted", conversationID: "same-conversation")
            let flushed = db.flushPendingStores()
            XCTAssertEqual(flushed.count, 1)
            XCTAssertEqual(flushed.first?.storedChunk.chunkID, existing.chunkID)
            XCTAssertEqual(flushed.first?.storedChunk.outcome, .duplicate)
            XCTAssertTrue(db.flushPendingStores().isEmpty, "duplicate replay must still clear its queue")
            DeferredStoreTelemetry.waitForWritesForTesting()
            let events = try String(contentsOf: root.appendingPathComponent("events.jsonl"), encoding: .utf8)
                .split(separator: "\n").map {
                    try XCTUnwrap(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
                }
            XCTAssertEqual(events.compactMap { $0["event"] as? String }, ["queued"])
            XCTAssertEqual(events.first?["queue_id"] as? String, queued.queueID)
        }
    }
}
