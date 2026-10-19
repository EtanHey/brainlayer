import Foundation
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
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        servedPath = directory.appendingPathComponent("served.db").path
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

    func testRequestWithoutExpectedDatabaseStillServesAndEchoesDBPath() throws {
        let target = directory.appendingPathComponent("snap.db")

        let result = vacuum(target: target, expected: nil)

        XCTAssertNil(result["isError"], text(result))
        XCTAssertTrue(text(result).contains("\"db_path\""), text(result))
    }
}
