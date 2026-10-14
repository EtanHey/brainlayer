@testable import BrainBarLifecycle
import XCTest

final class MCPTransportHealthTests: XCTestCase {
    func testReceiptFileRejectsForeignOwnerAndNonRegularFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("owned.json")
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: Data("{}".utf8), attributes: [.posixPermissions: 0o600]))
        XCTAssertEqual(try TransportReceiptFile.read(file.path), Data("{}".utf8))
        XCTAssertThrowsError(try TransportReceiptFile.read(file.path, ownerUID: getuid() + 1))
        XCTAssertThrowsError(try TransportReceiptFile.read(root.path))
        let fifo = root.appendingPathComponent("fifo")
        XCTAssertEqual(mkfifo(fifo.path, 0o600), 0)
        XCTAssertThrowsError(try TransportReceiptFile.read(fifo.path))
    }
    func testUnsafePriorReceiptsAreRefusedWithoutRestoringCounts() throws {
        for defect in ["symlink", "hardlink", "loose-mode", "dangling-symlink"] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let path = root.appendingPathComponent("transport-health.json")
            let db = root.appendingPathComponent("fixture.db").path
            let producer = MCPTransportHealth(path: path.path)
            producer.record(.requestsDropped, count: 3)
            producer.publish(dbPath: db, inFlight: 0)
            let original = try Data(contentsOf: path)
            let target = root.appendingPathComponent("other.json")
            if defect.contains("symlink") {
                try FileManager.default.moveItem(at: path, to: target)
                try FileManager.default.createSymbolicLink(at: path, withDestinationURL: defect == "symlink" ? target : root.appendingPathComponent("missing.json"))
            } else if defect == "hardlink" {
                XCTAssertEqual(link(path.path, target.path), 0)
            } else { XCTAssertEqual(chmod(path.path, 0o644), 0) }
            let next = MCPTransportHealth(path: path.path)
            next.publish(dbPath: db, inFlight: 0)
            let value = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
            let incident = try XCTUnwrap(value["incident"] as? [String: Any], defect)
            XCTAssertEqual(incident["state"] as? String, "unmeasurable", defect)
            XCTAssertNil(incident["counts"], defect)
            XCTAssertEqual(value["state"] as? String, "unmeasurable", defect)
            if FileManager.default.fileExists(atPath: target.path) {
                XCTAssertEqual(try Data(contentsOf: target), original, "Refused links must not change their other target")
            }
        }
    }
    func testFailureEpisodeSurvivesProducerRestartBeforeScheduledConsumer() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("transport-health.json")
        let db = root.appendingPathComponent("fixture.db").path
        let first = MCPTransportHealth(path: path.path)
        first.record(.requestsDropped, count: 3)
        first.publish(dbPath: db, inFlight: 0, stopped: true)
        let restarted = MCPTransportHealth(path: path.path)
        restarted.publish(dbPath: db, inFlight: 0)
        let value = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        let episode = try XCTUnwrap(value["incident"] as? [String: Any], "A restart before the Python poll must not erase a known episode")
        let counts = try XCTUnwrap(episode["counts"] as? [String: Int])
        XCTAssertEqual(counts["requests_dropped"], 3)
        XCTAssertEqual(value["observed_seconds"] as? Int, 0)
    }
    func testRetainedEpisodeClearsOnlyAfterAFullCleanWindow() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("transport-health.json")
        let db = root.appendingPathComponent("fixture.db").path
        let start = Date()
        var elapsed: TimeInterval = 0
        let first = MCPTransportHealth(path: path.path, now: { start.addingTimeInterval(elapsed) }, uptime: { elapsed })
        first.record(.requestsDropped, count: 3)
        first.publish(dbPath: db, inFlight: 0)
        elapsed = 100
        let next = MCPTransportHealth(path: path.path, now: { start.addingTimeInterval(elapsed) }, uptime: { elapsed })
        elapsed = 999
        next.publish(dbPath: db, inFlight: 0)
        let partial = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        XCTAssertNotNil(partial["incident"] as? [String: Any])
        elapsed = 1000
        next.publish(dbPath: db, inFlight: 0)
        let clean = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        XCTAssertTrue(clean["incident"] is NSNull)
    }

    func testMalformedPriorReceiptPreservesUnknownWithoutCopyingPayload() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("transport-health.json")
        try Data("{\"private_payload\":\"DO_NOT_KEEP\", ".utf8).write(to: path)
        let next = MCPTransportHealth(path: path.path)
        next.publish(dbPath: root.appendingPathComponent("fixture.db").path, inFlight: 0)
        let data = try Data(contentsOf: path)
        let value = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let incident = try XCTUnwrap(value["incident"] as? [String: Any])
        XCTAssertEqual(incident["state"] as? String, "unmeasurable")
        XCTAssertNil(incident["counts"])
        XCTAssertFalse(String(data: data, encoding: .utf8)!.contains("DO_NOT_KEEP"))
    }

}
