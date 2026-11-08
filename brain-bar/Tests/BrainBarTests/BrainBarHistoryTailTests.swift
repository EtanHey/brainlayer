import Foundation
import XCTest
@testable import BrainBar

final class BrainBarHistoryTailTests: XCTestCase {
    func testTornLinesKeepValidResultsAndLargeHistoryReadsTail() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build.noindex/history-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("runs.jsonl")
        let valid = "{\"event\":\"finished\",\"step\":\"index\",\"ts\":123,\"status\":\"ok\",\"returncode\":0}\n"
        for prefix in ["", String(repeating: "x", count: 4 * 1024 * 1024 + 100) + "\n"] {
            try Data((prefix + "{torn\n" + valid + "{torn").utf8).write(to: file)
            let history = BrainBarCalendarHistory.read(directory: root)
            XCTAssertEqual(history.skippedLineCount, 2)
            XCTAssertNil(history.error)
            XCTAssertEqual(history.runs["index"]?.code, 0)
        }
    }
}
