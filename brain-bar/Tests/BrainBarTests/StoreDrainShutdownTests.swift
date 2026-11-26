import SQLite3
import XCTest
@testable import BrainBar

/// Parks the first SQLite progress callback only. SQLite can call the handler
/// again after it asked for an interrupt (CI's macOS 15 SQLite did, 24 Nov). A
/// second park would wait on a consumed release while holding transactionLock,
/// and the test's own cleanup would then wait on that lock forever.
private final class ShutdownReadGate: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var shouldPark = true

    func step() -> Int32 {
        lock.lock()
        let park = shouldPark
        shouldPark = false
        lock.unlock()
        if park {
            entered.signal()
            release.wait()
        }
        return 1 // End the query after observing its handle lifetime protection.
    }
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
        var readerFinished = true
        // close() takes transactionLock, so a reader that never finished keeps its database.
        defer { if readerFinished { database.close(); removeBrainBarTestDatabase(path) } }
        let gate = ShutdownReadGate()
        try database.withSQLiteHandleForTesting { handle in
            sqlite3_progress_handler(handle, 1, { context in
                Unmanaged<ShutdownReadGate>.fromOpaque(context!).takeUnretainedValue().step()
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
        // Every wait below is bounded: a reader still holding transactionLock
        // fails this test instead of hanging the whole suite.
        guard finished.wait(timeout: .now() + 5) == .success else {
            readerFinished = false
            return XCTFail("Queue dedupe read did not finish; leaving its database open")
        }
        try database.withSQLiteHandleForTesting { sqlite3_progress_handler($0, 0, nil, nil) }
        database.close()
        XCTAssertThrowsError(try database.hasStoredQueuedItemForTesting(queueID: "absent")) {
            guard case BrainDatabase.DBError.notOpen = $0 else { return XCTFail("Expected notOpen, got \($0)") }
        }
    }

    func testProductionStopJoinsDrainBeforeClosingOwnedDatabase() throws {
        let fixture = try HeldDrainServerFixture(test: self)
        defer { fixture.removeFiles() }
        DispatchQueue.global().async { fixture.server.stop(); fixture.stopped.signal() }
        let prematureStop = fixture.stopped.wait(timeout: .now() + 0.2)
        XCTAssertEqual(prematureStop, .timedOut, "Production cleanup must join the held drain")
        fixture.release.signal()
        XCTAssertEqual(fixture.drained.wait(timeout: .now() + 5), .success)
        if prematureStop == .timedOut {
            XCTAssertEqual(fixture.stopped.wait(timeout: .now() + 5), .success)
        }
    }

    /// A drain can block without limit in the pending-store flock (the Python side
    /// takes the same lock file). Shutdown must give up on it after its bound and
    /// must not close the handle under it.
    func testProductionStopBoundsDrainJoinAndKeepsHandleOpenForLiveDrain() throws {
        executionTimeAllowance = 60
        let fixture = try HeldDrainServerFixture(test: self, joinTimeout: 0.3)
        defer { fixture.removeFiles() }
        DispatchQueue.global().async { fixture.server.stop(); fixture.stopped.signal() }
        let stop = fixture.stopped.wait(timeout: .now() + 3)
        XCTAssertEqual(stop, .success, "stop() must return within the drain-join bound")
        XCTAssertTrue(fixture.database.isOpen, "A timed-out join must leave the live drain's handle open")
        // The drain body then runs its query on that handle (XCTFail inside if closed).
        fixture.release.signal()
        XCTAssertEqual(fixture.drained.wait(timeout: .now() + 5), .success)
        if stop == .timedOut { XCTAssertEqual(fixture.stopped.wait(timeout: .now() + 5), .success) }
    }
}

/// A started production server with a real, server-owned SQLite handle and one
/// drain held on the production drain queue until `release` is signalled.
private final class HeldDrainServerFixture: @unchecked Sendable {
    let root: URL
    let path: String
    let server: BrainBarServer
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let stopped = DispatchSemaphore(value: 0)
    let drained = DispatchSemaphore(value: 0)
    private(set) var database: BrainDatabase!

    init(test: XCTestCase, joinTimeout: TimeInterval? = nil) throws {
        // A short directory keeps the socket below Darwin's 104-byte Unix socket limit.
        root = brainBarTestTemporaryDirectory()
            .appendingPathComponent("sd-" + UUID().uuidString.prefix(8))
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        path = root.appendingPathComponent("fixture.db").path
        let socketPath = root.appendingPathComponent("s.sock").path
        XCTAssertLessThan(socketPath.utf8.count, 104, "Socket path exceeds Darwin's sun_path limit")
        server = BrainBarServer(socketPath: socketPath, dbPath: path, enableHybridSearchHelper: false)
        if let joinTimeout { server.pendingStoreDrainJoinTimeout = joinTimeout }
        let ready = test.expectation(description: "database ready")
        server.onDatabaseReady = { [self] database in
            self.database = database
            DispatchQueue.global().async { [self] in
                server.enqueueStoreDrainForTesting { [self] in
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
        test.wait(for: [ready], timeout: 5)
        server.onDatabaseReady = nil
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
    }

    /// Runs after the drain finished, so closing a handle a timed-out stop left open is safe.
    func removeFiles() {
        database?.close()
        removeBrainBarTestDatabase(path)
        // The socket's .lock file lives under root; tolerate a run that never made it.
        if FileManager.default.fileExists(atPath: root.path) {
            do { try FileManager.default.removeItem(at: root) }
            catch { XCTFail("Unable to remove fixture root \(root.path): \(error)") }
        }
    }
}
