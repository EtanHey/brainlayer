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

    func testReplacementBetweenSQLiteOpenAndInitialIdentityCaptureRefuses() throws {
        db.close()
        let actualA = directory.appendingPathComponent("opening-window-source.db")
        try writeMarker(actualA.path, "A_HELD_BY_SQLITE")
        servedPath = directory.appendingPathComponent("opening-window-alias.db").path
        try FileManager.default.createSymbolicLink(atPath: servedPath, withDestinationPath: actualA.path)
        let replacement = directory.appendingPathComponent("replacement.db")
        try writeMarker(replacement.path, "B_AT_EXPECTED_PATH")
        let moved = directory.appendingPathComponent("opened-A.db")
        var seamRan = false
        db = BrainDatabase(
            path: servedPath,
            openConfiguration: .init(),
            onConnectionOpenedForTesting: { handle in
                do {
                    try FileManager.default.moveItem(atPath: self.servedPath, toPath: moved.path)
                    try FileManager.default.moveItem(at: replacement, to: URL(fileURLWithPath: self.servedPath))
                    seamRan = true
                    var moved: Int32 = -1
                    let rc = sqlite3_file_control(handle, "main", SQLITE_FCNTL_HAS_MOVED, &moved)
                    print("OPEN_WINDOW_VFS filename=\(String(cString: sqlite3_db_filename(handle, "main"))) rc=\(rc) moved=\(moved)")
                } catch { XCTFail("opening-window fixture failed: \(error)") }
            }
        )
        router.setDatabase(db)
        XCTAssertTrue(seamRan)
        XCTAssertTrue(db.isOpen, "scratch connection must remain usable for the identity assertion")
        XCTAssertEqual(marker(moved.path), "A_HELD_BY_SQLITE")
        XCTAssertEqual(marker(servedPath), "B_AT_EXPECTED_PATH")
        let mismatch = db.openDatabaseIdentityMismatch(expectedPath: servedPath)
        let target = directory.appendingPathComponent("opening-window.db")
        let result = vacuum(target: target, expected: servedPath)
        print("OPEN_WINDOW_OUTCOME mismatch=\(String(describing: mismatch)) result=\(text(result)) raw=\(String(describing: marker(target.path)))")
        XCTAssertNotNil(mismatch, "pathname B must not be accepted as the SQLite connection still holding A")
        XCTAssertEqual(result["isError"] as? Bool, true, text(result))
        XCTAssertTrue(text(result).contains("BrainBar database identity mismatch"), text(result))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path + ".complete"))
    }

    func testOpeningThroughUnchangedAliasBacksUpTheActualSQLiteData() throws {
        try writeMarker(servedPath, "VALID_A")
        db.close()
        let actual = servedPath!
        let alias = directory.appendingPathComponent("valid-alias.db").path
        try FileManager.default.createSymbolicLink(atPath: alias, withDestinationPath: actual)
        db = BrainDatabase(path: alias)
        router.setDatabase(db)
        let target = directory.appendingPathComponent("valid.db")
        let result = vacuum(target: target, expected: actual)
        XCTAssertNil(result["isError"], text(result))
        XCTAssertEqual(marker(target.path), "VALID_A")
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text(result).utf8)) as? [String: Any])
        let source = try inode(actual)
        XCTAssertEqual((payload["db_dev"] as? NSNumber)?.int64Value, source.dev)
        XCTAssertEqual((payload["db_ino"] as? NSNumber)?.int64Value, source.ino)
    }

    func testUnsupportedFailedOrUnwrittenVFSValidationNeverAuthorizesBackup() throws {
        for (index, control) in [(SQLITE_NOTFOUND, Int32(0)), (SQLITE_IOERR, Int32(0)), (SQLITE_OK, Int32(-1))].enumerated() {
            db.backupHasMovedResultForTesting = control
            for expected in [servedPath, nil] as [String?] {
                let target = directory.appendingPathComponent("unknown-\(index)-\(expected == nil).db")
                let result = vacuum(target: target, expected: expected)
                XCTAssertEqual(result["isError"] as? Bool, true, text(result))
                XCTAssertTrue(text(result).contains("BrainBar database identity mismatch"), text(result))
                XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
                XCTAssertFalse(FileManager.default.fileExists(atPath: target.path + ".complete"))
            }
        }
    }

    func testUnknownInitialBindingIsNeverAdoptedLater() throws {
        db.close()
        db = BrainDatabase(path: servedPath, openConfiguration: .init(),
                           onConnectionOpenedForTesting: { _ in },
                           hasMovedResultForTesting: (SQLITE_NOTFOUND, 0))
        XCTAssertTrue(db.isOpen)
        XCTAssertNil(db.openedFileIdentity)
        db.backupHasMovedResultForTesting = nil
        router.setDatabase(db)
        let target = directory.appendingPathComponent("initially-unknown.db")
        let result = vacuum(target: target, expected: nil)
        XCTAssertEqual(result["isError"] as? Bool, true, text(result))
        XCTAssertTrue(text(result).contains("identity is unknown"), text(result))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
    }

    func testInMemoryConnectionCannotPublishAnUnboundBackup() throws {
        db.close()
        db = BrainDatabase(path: ":memory:")
        let target = directory.appendingPathComponent("in-memory.db")
        XCTAssertNil(db.openedFileIdentity)
        XCTAssertThrowsError(try db.vacuumInto(targetPath: target.path)) { error in
            XCTAssertTrue(String(describing: error).contains("BrainBar database identity mismatch"), "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path + ".complete"))
    }

    private final class VacuumReplacement {
        let alias: String
        let replacement: URL
        let aside: URL
        var ran = false
        var error: Error?
        init(alias: String, replacement: URL, aside: URL) {
            self.alias = alias
            self.replacement = replacement
            self.aside = aside
        }
        func replaceOnce() {
            guard !ran else { return }
            ran = true
            do {
                try FileManager.default.moveItem(atPath: alias, toPath: aside.path)
                try FileManager.default.moveItem(at: replacement, to: URL(fileURLWithPath: alias))
            } catch { self.error = error }
        }
    }

    func testAliasReplacementDuringSQLiteVacuumRefusesBeforePublishingRawOrReceipt() throws {
        try writeMarker(servedPath, "A_DURING_VACUUM")
        db.close()
        let alias = directory.appendingPathComponent("during-vacuum-alias.db").path
        try FileManager.default.createSymbolicLink(atPath: alias, withDestinationPath: servedPath)
        let replacement = directory.appendingPathComponent("during-vacuum-B.db")
        try writeMarker(replacement.path, "B_DURING_VACUUM")
        db = BrainDatabase(path: alias)
        router.setDatabase(db)
        let fixture = VacuumReplacement(alias: alias, replacement: replacement,
                                        aside: directory.appendingPathComponent("during-vacuum-old-alias"))
        try db.withSQLiteHandleForTesting { handle in
            sqlite3_progress_handler(handle, 1, { context in
                guard let context else { return 0 }
                Unmanaged<VacuumReplacement>.fromOpaque(context).takeUnretainedValue().replaceOnce()
                return 0
            }, Unmanaged.passUnretained(fixture).toOpaque())
        }
        defer { try? db.withSQLiteHandleForTesting { sqlite3_progress_handler($0, 0, nil, nil) } }
        let target = directory.appendingPathComponent("during-vacuum.db")
        let result = vacuum(target: target, expected: alias)
        XCTAssertTrue(fixture.ran, "fixture must execute inside SQLite after backup preflight")
        XCTAssertNil(fixture.error)
        XCTAssertEqual(result["isError"] as? Bool, true, text(result))
        XCTAssertTrue(text(result).contains("BrainBar database identity mismatch"), text(result))
        XCTAssertEqual(marker(alias), "B_DURING_VACUUM")
        XCTAssertEqual(marker(servedPath), "A_DURING_VACUUM")
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path + ".complete"))
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

    func testCompletionReceiptBindsTheSyncedRawIdentity() throws {
        let target = directory.appendingPathComponent("receipt.db")
        let result = vacuum(target: target, expected: servedPath)
        XCTAssertNil(result["isError"], text(result))
        let data = try Data(contentsOf: URL(fileURLWithPath: target.path + ".complete"))
        let receipt = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        XCTAssertNotNil(receipt, "completion marker must bind the writer's synced raw identity")
        let raw = try inode(target.path)
        XCTAssertEqual((receipt?["schema_version"] as? NSNumber)?.intValue, 1)
        XCTAssertEqual(receipt?["target_path"] as? String, target.path)
        XCTAssertEqual((receipt?["target_dev"] as? NSNumber)?.int64Value, raw.dev)
        XCTAssertEqual((receipt?["target_ino"] as? NSNumber)?.int64Value, raw.ino)
    }

    func testPeerAtPublicTargetBeforeWriterCompletionIsPreserved() throws {
        let target = directory.appendingPathComponent("peer.db")
        let peer = Data("peer owns this pathname".utf8)
        db.onBackupStagingCreatedForTesting = {
            do { try peer.write(to: target, options: .withoutOverwriting) }
            catch { XCTFail("could not create peer fixture: \(error)") }
        }
        let result = vacuum(target: target, expected: servedPath)
        XCTAssertEqual(result["isError"] as? Bool, true)
        XCTAssertEqual(try Data(contentsOf: target), peer)
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path + ".complete"))
        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertFalse(entries.contains { $0.hasPrefix(".brainbar-vacuum-") })
    }
}
