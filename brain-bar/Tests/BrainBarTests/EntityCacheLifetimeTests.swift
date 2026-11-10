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
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let readFinished = DispatchSemaphore(value: 0)
        let closeStarted = DispatchSemaphore(value: 0)
        let closeFinished = DispatchSemaphore(value: 0)
        defer { release.signal() }
        DispatchQueue.global().async {
            defer { readFinished.signal() }
            do {
                try db.withSQLiteHandle { handle in
                    var statement: OpaquePointer?
                    XCTAssertEqual(sqlite3_prepare_v2(handle, "SELECT 1", -1, &statement, nil), SQLITE_OK)
                    defer { sqlite3_finalize(statement) }
                    entered.signal()
                    XCTAssertEqual(release.wait(timeout: .now() + 5), .success)
                    XCTAssertEqual(sqlite3_step(statement), SQLITE_ROW)
                }
            } catch { XCTFail("borrowed read failed: \(error)") }
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
        XCTAssertThrowsError(try db.withSQLiteHandle { _ in XCTFail("closed handle escaped") })
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
        db.close()
        cache.load(from: db)
        XCTAssertEqual(cache.detectEntities(in: "Fixture Person").map(\.id), ["lifetime-person"])
    }
}
