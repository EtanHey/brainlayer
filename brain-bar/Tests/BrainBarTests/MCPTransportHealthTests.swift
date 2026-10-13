import BrainBarLifecycle
import XCTest

final class MCPTransportHealthTests: XCTestCase {
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
