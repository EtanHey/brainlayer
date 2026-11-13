import Foundation
import SQLite3
import XCTest
@testable import BrainBar

final class EntityCacheLifetimeTests: XCTestCase {
    func testCloseWaitsUntilBorrowedSQLiteReadFinishes() throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent("brainbar-entity-lifetime-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let db = BrainDatabase(path: root.appendingPathComponent("test.db").path)
        defer { db.close() }
        try db.insertEntity(id: "race-person", type: "person", name: "Race Person")
        let cache = EntityCache()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let readFinished = DispatchSemaphore(value: 0)
        let closeStarted = DispatchSemaphore(value: 0)
        let closeFinished = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let gate = ReadGate(entered: entered, release: release)
        defer { withExtendedLifetime(gate) {} }
        try db.withSQLiteHandleForTesting { handle in
            sqlite3_set_authorizer(handle, { context, _, _, _, _, _ in
                let gate = Unmanaged<ReadGate>.fromOpaque(context!).takeUnretainedValue()
                if !gate.didEnter {
                    gate.didEnter = true
                    gate.entered.signal()
                    _ = gate.release.wait(timeout: .now() + 5)
                }
                return SQLITE_OK
            }, Unmanaged.passUnretained(gate).toOpaque())
        }
        DispatchQueue.global().async {
            defer { readFinished.signal() }
            cache.load(from: db)
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
        DispatchQueue.global().async {
            closeStarted.signal()
            db.close()
            closeFinished.signal()
        }
        XCTAssertEqual(closeStarted.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(closeFinished.wait(timeout: .now() + 0.05), .timedOut)
        release.signal()
        XCTAssertEqual(readFinished.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(closeFinished.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(cache.detectEntities(in: "Race Person").map(\.id), ["race-person"])
    }

    func testCacheRefreshAfterOwnerCloseRetainsSafeSnapshot() throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent("brainbar-entity-lifetime-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let db = BrainDatabase(path: root.appendingPathComponent("test.db").path)
        defer { db.close() }
        try db.insertEntity(id: "lifetime-person", type: "person", name: "Fixture Person")
        let cache = EntityCache()
        defer { XCTAssertEqual(cache.stopAndDrainForTesting(deadline: .now() + 5), .finished) }
        cache.load(from: db)
        XCTAssertEqual(cache.detectEntities(in: "Fixture Person").map(\.id), ["lifetime-person"])
        cache.startRefreshTimer(db: db)
        db.close()
        cache.fireRefreshTimerForTesting()
        XCTAssertEqual(cache.detectEntities(in: "Fixture Person").map(\.id), ["lifetime-person"])
    }
    func testTimerDoesNotReadRetiredHandleEvenWhenSQLiteCloseIsBusy() throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent("entity-timer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let db = BrainDatabase(path: root.appendingPathComponent("test.db").path)
        try db.insertEntity(id: "timer-person", type: "person", name: "Original Person")
        let cache = EntityCache()
        defer { XCTAssertEqual(cache.stopAndDrainForTesting(deadline: .now() + 5), .finished) }
        cache.load(from: db)
        XCTAssertEqual(cache.detectEntities(in: "Original Person").map(\.id), ["timer-person"])
        cache.startRefreshTimer(db: db)

        // A pending statement keeps the retired handle allocated: the old timer then
        // deterministically reads the changed row instead of relying on a use-after-free crash.
        let handle = try XCTUnwrap(db.dbHandle)
        var pending: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(handle, "SELECT 1", -1, &pending, nil), SQLITE_OK)
        defer {
            sqlite3_finalize(pending)
            sqlite3_close(handle)
        }
        XCTAssertEqual(sqlite3_exec(handle, "UPDATE kg_entities SET name = 'Changed Person'", nil, nil, nil), SQLITE_OK)
        db.close()
        cache.fireRefreshTimerForTesting()
        XCTAssertEqual(cache.detectEntities(in: "Original Person").map(\.id), ["timer-person"])
        XCTAssertTrue(cache.detectEntities(in: "Changed Person").isEmpty)
    }
}

private final class ReadGate: @unchecked Sendable {
    var didEnter = false
    let entered: DispatchSemaphore
    let release: DispatchSemaphore

    init(entered: DispatchSemaphore, release: DispatchSemaphore) {
        self.entered = entered
        self.release = release
    }
}
