import Foundation
import XCTest
@testable import BrainBar

final class BrainBarHybridSingleFlightTests: XCTestCase {
    final class ParkedExecutor: @unchecked Sendable {
        let lock = NSLock()
        var work: [@Sendable () -> Void] = []
        var submissions = 0
        func submit(_ action: @escaping @Sendable () -> Void) {
            lock.lock(); defer { lock.unlock() }
            submissions += 1; work.append(action)
        }
        func drain() {
            lock.lock(); let pending = work; work = []; lock.unlock()
            pending.forEach { $0() }
        }
    }
    final class Helper: HybridSearchClientProtocol, @unchecked Sendable {
        func search(arguments: [String: Any]) throws -> HybridSearchResponse { .init(text: "helper") }
    }
    func testDefaultBudgetAdmitsHelperBeyondLegacy800Milliseconds() throws {
        let router = MCPRouter(profile: "full", hybridSearchClient: Helper(),
            hybridSearchExecutor: { action in
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.9, execute: action)
            })
        XCTAssertEqual(try router.hybridSearchWithinBudget(arguments: [:])?.text, "helper")
    }

    func testHealthyConcurrentSearchesBothReachHybridClient() throws {
        let entered = DispatchSemaphore(value: 0)
        let executor = ParkedExecutor()
        let router = MCPRouter(profile: "full", hybridSearchClient: Helper(), hybridSearchBudget: 5,
            hybridSearchExecutor: { action in
                executor.submit(action)
                entered.signal()
            })
        let finished = expectation(description: "both callers finish")
        finished.expectedFulfillmentCount = 2
        for _ in 0..<2 {
            DispatchQueue.global().async {
                defer { finished.fulfill() }
                do {
                    XCTAssertEqual(try router.hybridSearchWithinBudget(arguments: [:])?.text, "helper")
                } catch { XCTFail("Unexpected error: \(error)") }
            }
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 1), .success)
        XCTAssertEqual(entered.wait(timeout: .now() + 1), .success)
        executor.drain()
        wait(for: [finished], timeout: 6)
        XCTAssertEqual(executor.submissions, 2)
    }

    func testTimedOutSearchKeepsAdmissionUntilWorkerActuallyFinishes() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build.noindex/search-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let db = BrainDatabase(path: root.appendingPathComponent("test.db").path)
        defer { db.close() }
        let executor = ParkedExecutor()
        defer { executor.drain() }
        let router = MCPRouter(profile: "full", hybridSearchClient: Helper(), hybridSearchBudget: 0.001,
            hybridSearchExecutor: { executor.submit($0) })
        router.setDatabases(write: db, read: db)
        for id in 1...3 {
            let response = router.handle(["jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": ["name": "brain_search", "arguments": ["query": "fixture"]]])
            XCTAssertTrue(String(describing: response).contains("Search mode: BrainBar database fallback"))
            let result = try XCTUnwrap(response["result"] as? [String: Any])
            let meta = try XCTUnwrap(result["_meta"] as? [String: Any])
            let route = try XCTUnwrap(meta["brainlayer/search"] as? [String: String])
            XCTAssertEqual(route["route"], "lexical_fallback")
            XCTAssertEqual(route["reason"], id == 1 ? "budget_exceeded" : "helper_busy")
            if id > 1 { XCTAssertTrue(String(describing: response).contains("hybrid helper busy past budget")) }
        }
        XCTAssertEqual(executor.submissions, 1, "Timeout must not admit another blocked worker")
        executor.drain()
        _ = router.handle(["jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": ["name": "brain_search", "arguments": ["query": "fixture"]]])
        XCTAssertEqual(executor.submissions, 2, "Actual completion releases admission")
    }
}
