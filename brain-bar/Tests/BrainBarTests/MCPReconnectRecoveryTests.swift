@testable import BrainBarLifecycle
import XCTest

final class MCPReconnectRecoveryTests: XCTestCase {
    func testAllowedTwoSocketChurnDoesNotAccumulateAcrossRestarts() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("transport.json"), db = root.appendingPathComponent("private.db").path
        let start = Date(); var elapsed = 0.0
        MCPTransportHealth(path: path.path, now: { start }, uptime: { 0 }).publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 12)
        let next = MCPTransportHealth(path: path.path, now: { start.addingTimeInterval(elapsed) }, uptime: { elapsed })
        elapsed = 60
        next.publish(dbPath: db, inFlight: 0, persistentConnections: 10)
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        XCTAssertTrue(raw["reconnect"] is NSNull, "Below-tripwire churn is resolved, not an accumulating deficit")
        next.publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 10)
        raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        XCTAssertEqual((raw["reconnect"] as? [String: Any])?["severed_persistent"] as? Int, 10)
    }

    func testDeficitSurvivesPartialGraceAndAnotherRestart() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("transport.json").path, db = root.appendingPathComponent("private.db").path
        let start = Date(); var elapsed = 0.0
        func producer() -> MCPTransportHealth {
            MCPTransportHealth(path: path, now: { start.addingTimeInterval(elapsed) }, uptime: { elapsed })
        }
        func snapshot() throws -> [String: Any] {
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any])
        }
        producer().publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 12)
        elapsed = 100
        let next = producer()
        elapsed = 159
        next.publish(dbPath: db, inFlight: 0, persistentConnections: 2)
        var recovery = try XCTUnwrap(snapshot()["reconnect"] as? [String: Any])
        XCTAssertEqual(recovery["observed_seconds"] as? Int, 59)
        XCTAssertEqual(recovery["severed_persistent"] as? Int, 12)
        next.publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 2)
        let third = producer()
        elapsed += 60
        third.publish(dbPath: db, inFlight: 0, persistentConnections: 0)
        recovery = try XCTUnwrap(snapshot()["reconnect"] as? [String: Any])
        XCTAssertEqual(recovery["observed_seconds"] as? Int, 60)
        XCTAssertEqual(recovery["severed_persistent"] as? Int, 12, "A second restart cannot shrink the unresolved baseline to2")
        XCTAssertEqual(recovery["recovered_persistent"] as? Int, 0)
        third.publish(dbPath: db, inFlight: 0, persistentConnections: 12)
        XCTAssertTrue(try snapshot()["reconnect"] is NSNull)
    }

    func testAgeQualifiedRecoveryMustRemainLiveAtGrace() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("transport.json").path, db = root.appendingPathComponent("private.db").path
        let start = Date(); var elapsed = 0.0
        let first = MCPTransportHealth(path: path, now: { start.addingTimeInterval(elapsed) }, uptime: { elapsed })
        first.publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 12)
        let next = MCPTransportHealth(path: path, now: { start.addingTimeInterval(elapsed) }, uptime: { elapsed })
        elapsed = 10
        next.publish(dbPath: db, inFlight: 0, persistentConnections: 12)
        elapsed = 60
        next.publish(dbPath: db, inFlight: 0, persistentConnections: 0)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any])
        let recovery = try XCTUnwrap(raw["reconnect"] as? [String: Any])
        XCTAssertEqual(recovery["severed_persistent"] as? Int, 12)
        XCTAssertEqual(recovery["recovered_persistent"] as? Int, 0, "Repeated closed sockets cannot accumulate recovery credit")
    }

    func testUnsafeOrMalformedRecoveryCannotBecomeCleanZero() throws {
        for defect in ["loose-mode", "boolean", "future", "foreign", "unknown"] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let path = root.appendingPathComponent("transport.json"), db = root.appendingPathComponent("private.db").path
            MCPTransportHealth(path: path.path).publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 12)
            var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
            var saved = try XCTUnwrap(raw["reconnect"] as? [String: Any])
            if defect == "boolean" { saved["severed_persistent"] = true }
            if defect == "future" { saved["severed_at"] = "2099-01-01T00:00:00Z" }
            if defect == "foreign" { saved["db_path"] = db + "-other" }
            if defect == "unknown" { saved = ["state": "unmeasurable", "reason": "unknown", "private_payload": "DO_NOT_COPY"] }
            raw["reconnect"] = saved
            try JSONSerialization.data(withJSONObject: raw).write(to: path)
            XCTAssertEqual(chmod(path.path, defect == "loose-mode" ? 0o644 : 0o600), 0)
            var elapsed = 0.0
            let next = MCPTransportHealth(path: path.path, now: { Date().addingTimeInterval(elapsed) }, uptime: { elapsed })
            elapsed = 900
            next.publish(dbPath: db, inFlight: 0)
            let data = try Data(contentsOf: path)
            raw = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(raw["state"] as? String, "unmeasurable", defect)
            XCTAssertNil(raw["counts"], defect)
            XCTAssertEqual((raw["reconnect"] as? [String: Any])?["state"] as? String, "unmeasurable", defect)
            XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("DO_NOT_COPY"))
        }
    }
    func testPreviousBootAndExpiredBaselineAreNonnumericAtExactBoundary() throws {
        for (bootChanged, gap, measured) in [(true, 60.0, false), (false, 172800.0, false), (false, 300.0, true), (false, 301.0, false)] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let path = root.appendingPathComponent("transport.json"), db = root.appendingPathComponent("private.db").path
            let start = Date(timeIntervalSince1970: floor(Date().timeIntervalSince1970))
            MCPTransportHealth(path: path.path, now: { start }).publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 12)
            var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
            var saved = try XCTUnwrap(raw["reconnect"] as? [String: Any])
            XCTAssertNotNil(saved["boot_time"])
            if bootChanged { saved["boot_time"] = "previous-boot" }
            raw["reconnect"] = saved
            try JSONSerialization.data(withJSONObject: raw).write(to: path)
            var elapsed = 0.0
            let next = MCPTransportHealth(path: path.path, now: { start.addingTimeInterval(gap + elapsed) }, uptime: { elapsed })
            next.publish(dbPath: db, inFlight: 0)
            raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
            XCTAssertEqual(raw["state"] as? String, measured ? "measured" : "unmeasurable")
            if !measured { XCTAssertNil(raw["counts"]); XCTAssertNil((raw["reconnect"] as? [String: Any])?["severed_persistent"]) }
            if measured {
                elapsed = 1; next.publish(dbPath: db, inFlight: 0)
                raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
                XCTAssertNil(raw["counts"], "The consumer and running producer must expire the same boundary")
            }
            let fresh = MCPTransportHealth(path: path.path, now: { start.addingTimeInterval(gap + elapsed) })
            fresh.publish(dbPath: db, inFlight: 0, stopped: true, persistentConnections: 12)
            raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
            XCTAssertEqual((raw["reconnect"] as? [String: Any])?["severed_persistent"] as? Int, 12, "Discarding stale evidence must allow a new shutdown baseline")
        }
    }

}
