import XCTest
@testable import BrainBar

final class BrainBarOperationReceiptTests: XCTestCase {
    func testDaemonReceiptIsReadableBySeparateUIStore() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-receipts-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let daemon = BrainBarOperationReceipts(url: url)
        let ui = BrainBarOperationReceipts(url: url)
        daemon.record(BrainBarOperationReceipt(kind: .search, durationMillis: 142, count: 10))
        XCTAssertNil(ui.search)
        ui.reload()
        XCTAssertEqual(ui.search?.count, 10)
        XCTAssertEqual(ui.search?.durationMillis, 142)
    }

    func testSearchCountUsesOnlyRecognizedResultHeaders() {
        XCTAssertEqual(BrainBarOperationReceipt.searchCount(in: "## Search results for \"x\" - 3 of 8 shown\n"), 3)
        XCTAssertEqual(BrainBarOperationReceipt.searchCount(in: "┌─ brain_search: \"x\" ─ 1 result\n"), 1)
        XCTAssertNil(BrainBarOperationReceipt.searchCount(in: "some search text with 10 results"))
    }

    func testReceiptFormatsMeasuredAndUnavailableFields() {
        XCTAssertEqual(
            BrainBarOperationReceipt(kind: .search, durationMillis: 142, count: 10).value,
            "142 ms · 10 results"
        )
        XCTAssertEqual(
            BrainBarOperationReceipt(kind: .ingest, durationMillis: 1_200, count: nil).value,
            "1.2 s · chunks unavailable"
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
        XCTAssertEqual(receipts.search?.value, "\(receipts.search?.durationMillis ?? -1) ms · failed · results unavailable")
    }
}
