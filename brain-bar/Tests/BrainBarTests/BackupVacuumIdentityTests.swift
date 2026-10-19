import Foundation
import SQLite3
import XCTest

@testable import BrainBar

/// 2026-10-19: a backup aimed at a temp DB reached BrainBar, which VACUUMed the production DB
/// into the caller's folder. `expected_db_path` must name the served DB, or nothing is written.
final class BackupVacuumIdentityTests: XCTestCase {
    private var directory: URL!
    private var servedPath: String!
    private var db: BrainDatabase!
    private var router: MCPRouter!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: brainBarTestTemporaryPath() + "vacuum-identity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("live"), withIntermediateDirectories: true
        )
        servedPath = directory.appendingPathComponent("live/served.db").path
        db = BrainDatabase(path: servedPath)
        router = MCPRouter(profile: "full")
        router.setDatabase(db)
    }

    override func tearDownWithError() throws {
        db.close()
        try? FileManager.default.removeItem(at: directory)
    }

    private func vacuum(target: URL, expected: String?) -> [String: Any] {
        var arguments: [String: Any] = ["target_path": target.path]
        if let expected { arguments["expected_db_path"] = expected }
        let response = router.handle([
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": "brain_backup_vacuum_into", "arguments": arguments] as [String: Any],
        ])
        return response["result"] as? [String: Any] ?? [:]
    }

    private func text(_ result: [String: Any]) -> String {
        ((result["content"] as? [[String: Any]])?.first?["text"] as? String) ?? ""
    }

    func testMismatchedExpectedDatabaseRefusesBeforeVacuum() throws {
        let other = directory.appendingPathComponent("other.db").path
        let otherDB = BrainDatabase(path: other)
        otherDB.close()
        let target = directory.appendingPathComponent("backups/2026-10-18.db")

        let result = vacuum(target: target, expected: other)

        XCTAssertEqual(result["isError"] as? Bool, true)
        XCTAssertTrue(text(result).contains("BrainBar database identity mismatch"), text(result))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path + ".complete"))
    }

    func testMissingExpectedDatabaseFileRefusesBeforeVacuum() throws {
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: directory.appendingPathComponent("absent.db").path)

        XCTAssertEqual(result["isError"] as? Bool, true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
    }

    func testSymlinkToServedDatabaseIsTheSameIdentityAndEchoesDBPath() throws {
        let link = directory.appendingPathComponent("link.db").path
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: servedPath)
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: link)

        XCTAssertNil(result["isError"], text(result))
        XCTAssertTrue(FileManager.default.fileExists(atPath: target.path))
        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text(result).utf8)) as? [String: Any]
        )
        XCTAssertEqual(payload["db_path"] as? String, servedPath)
    }

    private func inode(_ path: String) throws -> (dev: Int64, ino: Int64) {
        var info = stat()
        guard stat(path, &info) == 0 else { throw CocoaError(.fileNoSuchFile) }
        return (Int64(info.st_dev), Int64(info.st_ino))
    }

    private func marker(_ path: String) -> String? {
        var handle: OpaquePointer?
        defer { sqlite3_close(handle) }
        guard sqlite3_open_v2(path, &handle, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return nil }
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(handle, "SELECT v FROM identity_marker", -1, &stmt, nil) == SQLITE_OK,
              sqlite3_step(stmt) == SQLITE_ROW, let text = sqlite3_column_text(stmt, 0) else { return nil }
        return String(cString: text)
    }

    private func writeMarker(_ path: String, _ value: String) throws {
        var handle: OpaquePointer?
        defer { sqlite3_close(handle) }
        guard sqlite3_open(path, &handle) == SQLITE_OK,
              sqlite3_exec(handle, "CREATE TABLE IF NOT EXISTS identity_marker(v TEXT); DELETE FROM identity_marker;"
                  + "INSERT INTO identity_marker VALUES('\(value)');", nil, nil, nil) == SQLITE_OK
        else { throw CocoaError(.fileWriteUnknown) }
    }

    /// Sol, PR #1414 round 1: move the live DB's directory aside and recreate the path with a
    /// different DB. A pathname check passes while the open connection still reads the OLD file.
    func testReplacedServedPathRefusesBecauseTheOpenHandleIsADifferentFile() throws {
        try writeMarker(servedPath, "OLD_OPEN_HANDLE")
        let live = directory.appendingPathComponent("live")
        try FileManager.default.moveItem(at: live, to: directory.appendingPathComponent("moved"))
        try FileManager.default.createDirectory(at: live, withIntermediateDirectories: true)
        try writeMarker(servedPath, "NEW_EXPECTED_DB")
        XCTAssertEqual(marker(servedPath), "NEW_EXPECTED_DB")
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: servedPath)

        XCTAssertEqual(result["isError"] as? Bool, true, text(result))
        XCTAssertTrue(text(result).contains("BrainBar database identity mismatch"), text(result))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
    }

    func testMovedServedDatabaseRefusesEvenWhenExpectedNamesItsNewPath() throws {
        let live = directory.appendingPathComponent("live")
        let moved = directory.appendingPathComponent("moved")
        try FileManager.default.moveItem(at: live, to: moved)
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: moved.appendingPathComponent("served.db").path)

        XCTAssertEqual(result["isError"] as? Bool, true, text(result))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
    }

    func testUnlinkedServedDatabaseRefuses() throws {
        try FileManager.default.removeItem(atPath: servedPath)
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: servedPath)

        XCTAssertEqual(result["isError"] as? Bool, true, text(result))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
    }

    func testEchoCarriesTheOpenDatabaseInode() throws {
        let target = directory.appendingPathComponent("snap.db")
        let opened = try inode(servedPath)

        let result = vacuum(target: target, expected: servedPath)

        XCTAssertNil(result["isError"], text(result))
        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text(result).utf8)) as? [String: Any]
        )
        XCTAssertEqual((payload["db_dev"] as? NSNumber)?.int64Value, opened.dev)
        XCTAssertEqual((payload["db_ino"] as? NSNumber)?.int64Value, opened.ino)
    }

    func testRequestWithoutExpectedDatabaseStillServesAndEchoesDBPath() throws {
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: nil)

        XCTAssertNil(result["isError"], text(result))
        XCTAssertTrue(text(result).contains("\"db_path\""), text(result))
    }
}
