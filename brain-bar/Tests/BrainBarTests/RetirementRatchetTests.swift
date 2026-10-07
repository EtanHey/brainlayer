import Darwin
import Foundation
import XCTest
@testable import BrainBar

final class RetirementRatchetTests: XCTestCase {
    private func call(_ router: MCPRouter, _ name: String, _ arguments: [String: Any]) -> [String: Any] {
        router.handle(["jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": ["name": name, "arguments": arguments]])
    }

    func testNativeNetworkBoundaryIsArmed() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["RETIREMENT_NATIVE_BOUNDARY"] == "1",
                          "Requires the retirement native boundary harness")
        XCTAssertEqual(ProcessInfo.processInfo.environment["RETIREMENT_NATIVE_BOUNDARY"], "1")
        setenv("RETIREMENT_PHASE", "control", 1)
        defer { setenv("RETIREMENT_PHASE", "candidate", 1) }
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { Darwin.close(fd) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = UInt16(9).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(result, -1)
        XCTAssertEqual(errno, EPERM)
    }

    func testEveryPaletteRejectsRetiredDispatch() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["RETIREMENT_NATIVE_BOUNDARY"] == "1",
                          "Requires the retirement native boundary harness")
        for profile in ["core", "full", "operator"] {
            let router = MCPRouter(profile: profile)
            _ = call(router, "expand_palette", [:])
            let listed = router.handle(["jsonrpc": "2.0", "id": 1, "method": "tools/list"])
            let tools = try XCTUnwrap((listed["result"] as? [String: Any])?["tools"] as? [[String: Any]])
            XCTAssertFalse(tools.contains { $0["name"] as? String == "brain_enrich" })
            XCTAssertTrue(tools.contains { $0["name"] as? String == "brain_store" })
            for arguments: [String: Any] in [["mode": "realtime"], ["mode": "batch", "phase": "submit"], ["stats": true]] {
                let rejected = call(router, "brain_enrich", arguments)
                XCTAssertEqual((rejected["error"] as? [String: Any])?["code"] as? Int, -32601)
                XCTAssertNil(rejected["result"])
            }
        }
    }

    func testActualTemporaryStoreDigestAndSearchRemainLocal() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("native.db").path
        let db = BrainDatabase(path: path)
        defer { db.close() }
        let router = MCPRouter(profile: "full", dbPath: path)
        router.setDatabase(db)
        let stored = call(router, "brain_store", ["content": "NATIVE-R11 synthetic local decision",
                                                  "project": "fixture", "type": "note"])
        XCTAssertNil(stored["error"])
        let receipt = try XCTUnwrap(stored["result"] as? [String: Any])
        let storedChunk = try XCTUnwrap(receipt["_brainbarStoredChunk"] as? [String: Any])
        XCTAssertNotNil(storedChunk["chunk_id"] as? String)
        let digest = call(router, "brain_digest", ["content": "NATIVE-R11-DIGEST synthetic historical content", "project": "fixture"])
        XCTAssertNil(digest["error"])
        XCTAssertEqual((digest["result"] as? [String: Any])?["content_integrity"] as? String, "verified")
        XCTAssertFalse(try db.search(query: "NATIVE-R11", limit: 5).isEmpty)
        let retired = call(router, "brain_enrich", ["mode": "realtime"])
        XCTAssertEqual((retired["error"] as? [String: Any])?["code"] as? Int, -32601)
    }
}
