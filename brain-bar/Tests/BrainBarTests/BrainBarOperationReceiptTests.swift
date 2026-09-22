import XCTest
@testable import BrainBar

final class BrainBarOperationReceiptTests: XCTestCase {
    func testDaemonReceiptIsReadableBySeparateUIStore() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-receipts-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let daemon = BrainBarOperationReceipts(url: url)
        let ui = BrainBarOperationReceipts(url: url)
        let recordedAt = Date(timeIntervalSince1970: 1_000)
        daemon.record(BrainBarOperationReceipt(kind: .search, durationMillis: 142, count: 10, recordedAt: recordedAt))
        daemon.waitForWritesForTesting()
        XCTAssertNil(ui.search)
        ui.reload()
        XCTAssertEqual(ui.search?.count, 10)
        XCTAssertEqual(ui.search?.durationMillis, 142)
        XCTAssertEqual(ui.search?.value(now: recordedAt.addingTimeInterval(180)), "142 ms · 10 results · 3 min ago")
    }

    func testSearchCountUsesOnlyRecognizedResultHeaders() {
        XCTAssertEqual(BrainBarOperationReceipt.searchCount(in: "## Search results for \"x\" - 3 of 8 shown\n"), 3)
        XCTAssertNil(BrainBarOperationReceipt.searchCount(in: "some search text with 10 results"))
    }

    func testReceiptFormatsMeasuredAndUnavailableFields() {
        let now = Date(timeIntervalSince1970: 1_000)
        XCTAssertEqual(
            BrainBarOperationReceipt(kind: .search, durationMillis: 142, count: 10,
                                     recordedAt: now.addingTimeInterval(-180)).value(now: now),
            "142 ms · 10 results · 3 min ago"
        )
        XCTAssertEqual(
            BrainBarOperationReceipt(kind: .ingest, durationMillis: 1_200, count: nil,
                                     recordedAt: now).value(now: now),
            "1.2 s · chunks unavailable · just now"
        )
    }

    func testRouterPublishesSearchAndStoreReceipts() throws {
        let dbPath = NSTemporaryDirectory() + "brainbar-receipt-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let receipts = BrainBarOperationReceipts()
        let router = MCPRouter(profile: "full", receiptStore: receipts)
        router.setDatabase(db)

        _ = router.handle([
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": "brain_store", "arguments": ["content": "Fixture receipt memory"]],
        ])
        XCTAssertEqual(receipts.ingest?.count, 1)
        XCTAssertNotNil(receipts.ingest?.durationMillis)

        _ = router.handle([
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": ["name": "brain_store", "arguments": ["content": "Fixture receipt memory"]],
        ])
        XCTAssertEqual(receipts.ingest?.count, 0, "a duplicate must not claim a new chunk")

        _ = router.handle([
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": ["name": "brain_search", "arguments": ["query": "Fixture receipt memory"]],
        ])
        XCTAssertEqual(receipts.search?.count, 1)
        XCTAssertNotNil(receipts.search?.durationMillis)

        _ = router.handle([
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": ["name": "brain_search", "arguments": [:] as [String: Any]],
        ])
        XCTAssertEqual(receipts.search?.value, "\(receipts.search?.durationMillis ?? -1) ms · failed · results unavailable · just now")
    }

    func testDefaultRouterIsMemoryOnlyAndLeavesNoSidecar() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("brainbar-no-receipt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let dbPath = directory.appendingPathComponent("brainlayer.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = MCPRouter(profile: "full", dbPath: dbPath)
        router.setDatabase(db)
        _ = router.handle([
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": ["name": "brain_store", "arguments": ["content": "Fixture only"]],
        ])
        XCTAssertFalse(router.receiptPersistenceEnabledForTesting)
        XCTAssertFalse(FileManager.default.fileExists(atPath:
            URL(fileURLWithPath: dbPath).deletingLastPathComponent().appendingPathComponent("operation-receipts.json").path))
    }

    func testSlowFailingReceiptWriteDoesNotBlockOrChangeMCPResponse() throws {
        let dbPath = NSTemporaryDirectory() + "brainbar-slow-receipt-\(UUID().uuidString).db"
        let db = BrainDatabase(path: dbPath)
        defer { db.close(); try? FileManager.default.removeItem(atPath: dbPath) }
        let writerEntered = expectation(description: "writer entered")
        let responseReady = expectation(description: "MCP response ready")
        let release = DispatchSemaphore(value: 0)
        let url = URL(fileURLWithPath: dbPath + ".receipts")
        let receipts = BrainBarOperationReceipts(url: url, persist: { _, _ in
            writerEntered.fulfill()
            _ = release.wait(timeout: .now() + 5)
            throw NSError(domain: "fixture", code: 1)
        })
        let router = MCPRouter(profile: "full", receiptStore: receipts)
        router.setDatabase(db)
        let baseline = MCPRouter(profile: "full")
        baseline.setDatabase(db)
        let request: [String: Any] = [
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": ["name": "brain_search", "arguments": ["query": "fixture"]],
        ]
        let expected = try JSONSerialization.data(withJSONObject: baseline.handle(request), options: .sortedKeys)
        DispatchQueue.global().async {
            let actualRequest: [String: Any] = [
                "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": ["name": "brain_search", "arguments": ["query": "fixture"]],
            ]
            let actual = try? JSONSerialization.data(withJSONObject: router.handle(actualRequest), options: .sortedKeys)
            XCTAssertEqual(actual, expected)
            responseReady.fulfill()
        }
        defer { release.signal(); receipts.waitForWritesForTesting() }
        wait(for: [writerEntered, responseReady], timeout: 2)
    }
}
