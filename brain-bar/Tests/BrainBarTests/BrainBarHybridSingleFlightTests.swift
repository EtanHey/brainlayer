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
            _ = router.handle(["jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": ["name": "brain_search", "arguments": ["query": "fixture"]]])
        }
        XCTAssertEqual(executor.submissions, 1, "Timeout must not admit another blocked worker")
        executor.drain()
        _ = router.handle(["jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": ["name": "brain_search", "arguments": ["query": "fixture"]]])
        XCTAssertEqual(executor.submissions, 2, "Actual completion releases admission")
    }
}
