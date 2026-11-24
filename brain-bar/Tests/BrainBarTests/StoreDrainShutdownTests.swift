import SQLite3
import XCTest
@testable import BrainBar

private final class ShutdownReadGate: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
}

final class StoreDrainShutdownTests: XCTestCase {
    func testShutdownReleasesCompletedBackupGateBeforeJoiningDrain() {
        let router = MCPRouter()
        let waitingAtGate = DispatchSemaphore(value: 0)
        let drained = DispatchSemaphore(value: 0)
        let stopped = DispatchSemaphore(value: 0)
        router.backupSnapshotStarted()
        router.enqueueStoreDrainForTesting(beforeDrain: { waitingAtGate.signal() }) {
            drained.signal()
        }
        XCTAssertEqual(waitingAtGate.wait(timeout: .now() + 5), .success)
        // The backup worker has joined, but its server-queue completion callback
        // cannot run while cleanup occupies that queue.
        DispatchQueue.global().async { router.stopPendingStoreDrains(); stopped.signal() }
        let result = stopped.wait(timeout: .now() + 5)
        XCTAssertEqual(result, .success, "Shutdown must release the completed backup gate before joining")
        // Also releases a broken implementation so a fail-first run cannot hang.
        router.backupSnapshotFinished()
        if result == .timedOut { XCTAssertEqual(stopped.wait(timeout: .now() + 5), .success) }
        XCTAssertEqual(drained.wait(timeout: .now() + 5), .success)
    }

    func testQueuedItemReadHoldsCloseLockAndRejectsClosedDatabase() throws {
        let path = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString + ".db").path
        let database = BrainDatabase(path: path)
        defer { database.close(); removeBrainBarTestDatabase(path) }
        let gate = ShutdownReadGate()
        try database.withSQLiteHandleForTesting { handle in
            sqlite3_progress_handler(handle, 1, { context in
                let gate = Unmanaged<ShutdownReadGate>.fromOpaque(context!).takeUnretainedValue()
                gate.entered.signal()
                gate.release.wait()
                return 1 // End the query after observing its handle lifetime protection.
            }, Unmanaged.passUnretained(gate).toOpaque())
        }
        let finished = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            _ = try? database.hasStoredQueuedItemForTesting(queueID: "absent")
            finished.signal()
        }
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 5), .success)
        XCTAssertTrue(database.transactionLockIsHeldElsewhereForTesting(),
                      "Queue dedupe must hold the lock close() uses while SQLite is reading")
        gate.release.signal()
        XCTAssertEqual(finished.wait(timeout: .now() + 5), .success)
        try database.withSQLiteHandleForTesting { sqlite3_progress_handler($0, 0, nil, nil) }
        database.close()
        XCTAssertThrowsError(try database.hasStoredQueuedItemForTesting(queueID: "absent")) {
            guard case BrainDatabase.DBError.notOpen = $0 else { return XCTFail("Expected notOpen, got \($0)") }
        }
    }

    func testProductionStopJoinsDrainBeforeClosingOwnedDatabase() throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let path = root.appendingPathComponent("fixture.db").path
        // A relative fixture socket stays below Darwin's 104-byte Unix socket limit
        // even when the harness worktree path is long.
        let socketPath = "sd-\(UUID().uuidString).sock"
        let server = BrainBarServer(socketPath: socketPath,
                                    dbPath: path, enableHybridSearchHelper: false)
        let ready = expectation(description: "database ready")
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let stopped = DispatchSemaphore(value: 0)
        let drained = DispatchSemaphore(value: 0)
        server.onDatabaseReady = { database in
            // Runs on the production drain queue, with a real, server-owned SQLite handle.
            DispatchQueue.global().async {
                server.enqueueStoreDrainForTesting {
                    entered.signal()
                    release.wait()
                    do { _ = try database.pragma("page_count") }
                    catch { XCTFail("Drain lost its open database: \(error)") }
                    drained.signal()
                }
                ready.fulfill()
            }
        }
        server.start()
        wait(for: [ready], timeout: 5)
        server.onDatabaseReady = nil
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
        DispatchQueue.global().async { server.stop(); stopped.signal() }
        let prematureStop = stopped.wait(timeout: .now() + 0.2)
        XCTAssertEqual(prematureStop, .timedOut, "Production cleanup must join the held drain")
        release.signal()
        XCTAssertEqual(drained.wait(timeout: .now() + 5), .success)
        if prematureStop == .timedOut {
            XCTAssertEqual(stopped.wait(timeout: .now() + 5), .success)
        }
        removeBrainBarTestDatabase(path)
        try FileManager.default.removeItem(atPath: socketPath + ".lock")
        try FileManager.default.removeItem(at: root)
    }
}
