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
            try Data((prefix + "{torn\n" + "{\"event\":\"runner\",\"scheduled_date\":\"2026-11-08\"}\n" + valid + "{torn").utf8).write(to: file)
            let history = BrainBarCalendarHistory.read(directory: root)
            XCTAssertEqual(history.skippedLineCount, 2)
            XCTAssertNil(history.error)
            XCTAssertEqual(history.runs["index"]?.code, 0)
        }
    }
    private func fixture(_ text: String, check: (URL) throws -> Void) throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build.noindex/history-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(text.utf8).write(to: root.appendingPathComponent("runs.jsonl"))
        try check(root)
    }

    func testSkippedRowsReachObservations() throws {
        try fixture("{torn\n" + finished()) { root in
            let history = BrainBarCalendarHistory.read(directory: root)
            let observations = BrainBarSupervisorStatusTests.supervised(history: history).sampleActivity()
            XCTAssertTrue(observations[.maintenanceNightly]?.calendarScheduleUncertainty?.contains("skipped 1") == true)
        }
    }

    private func finished(_ code: Int = 0) -> String {
        "{\"event\":\"finished\",\"step\":\"index\",\"ts\":123,\"status\":\"ok\",\"returncode\":\(code)}\n"
    }

    func testRejectedFinishedPreservesSpawn() throws {
        let spawn = "{\"event\":\"spawn\",\"step\":\"index\",\"ts\":100}\n"
        try fixture(spawn + finished().replacingOccurrences(of: "\"returncode\":0", with: "\"returncode\":\"bad\"") + finished()) { root in
            let history = BrainBarCalendarHistory.read(directory: root)
            XCTAssertEqual(history.skippedLineCount, 1)
            XCTAssertEqual(history.runs["index"]?.startedAt, Date(timeIntervalSince1970: 100))
        }
    }

    func testRejectedSummaryPreservesPendingRunnerError() throws {
        let error = "{\"event\":\"runner-error\",\"ts\":100}\n"
        let summary = "{\"event\":\"summary\",\"kind\":\"nightly\",\"scheduled_date\":\"2026-11-08\",\"ts\":123,\"failed\":1}\n"
        try fixture(error + summary.replacingOccurrences(of: "\"failed\":1", with: "\"failed\":-1").replacingOccurrences(of: "2026-11-08", with: "2026-11-07") + summary) { root in
            let history = BrainBarCalendarHistory.read(directory: root)
            XCTAssertEqual(history.skippedLineCount, 1)
            XCTAssertEqual(history.maintenanceNight?.runnerErrorAt, Date(timeIntervalSince1970: 100))
        }
    }

    func testTruncatedAttemptDoesNotInventNightFailureOrStreak() throws {
        let prefix = "{\"event\":\"runner\",\"kind\":\"nightly\",\"scheduled_date\":\"2026-11-08\"}\n"
        let summary = "{\"event\":\"summary\",\"kind\":\"nightly\",\"scheduled_date\":\"2026-11-08\",\"ts\":123,\"failed\":1}\n"
        try fixture(prefix + String(repeating: " ", count: 4 * 1024 * 1024) + "\n" + summary + finished()) { root in
            let history = BrainBarCalendarHistory.read(directory: root)
            XCTAssertNil(history.maintenanceNight)
            XCTAssertNil(history.runs["index"])
        }
    }

    func testBoundaryAndLastRunsMatchWholeReader() throws {
        let rows = (0..<20).map { n in finished(n).replacingOccurrences(of: "\"index\"", with: "\"step-\(n)\"") }
        let suffix = rows.suffix(5).joined()
        try fixture(rows.joined()) { root in
            let tail = try XCTUnwrap(BrainBarJobOwnership.readOwnedFile(root.appendingPathComponent("runs.jsonl"), privateMode: false, limit: suffix.utf8.count, tail: true))
            XCTAssertEqual(String(decoding: tail, as: UTF8.self), suffix)
            let whole = BrainBarCalendarHistory.read(directory: root)
            try fixture(suffix) { shortRoot in
                let short = BrainBarCalendarHistory.read(directory: shortRoot)
                for n in 15..<20 {
                    XCTAssertEqual(short.runs["step-\(n)"]?.code, whole.runs["step-\(n)"]?.code)
                    XCTAssertEqual(short.runs["step-\(n)"]?.at, whole.runs["step-\(n)"]?.at)
                    XCTAssertEqual(short.runs["step-\(n)"]?.consecutiveFailures, whole.runs["step-\(n)"]?.consecutiveFailures)
                }
            }
        }
    }
    func testShrinkAfterStatStillDropsPartialFirstLine() throws {
        let text = "1234567890partial\nkeep\ntrailing"
        try fixture(text) { root in
            let url = root.appendingPathComponent("runs.jsonl")
            let writer = try FileHandle(forWritingTo: url)
            defer { try? writer.close() }
            let data = try BrainBarJobOwnership.readOwnedFile(url, privateMode: false,
                limit: text.utf8.count - 11, tail: true, onTruncated: {
                    try? writer.truncate(atOffset: UInt64("1234567890partial\nkeep\n".utf8.count))
                })
            XCTAssertEqual(data.map { String(decoding: $0, as: UTF8.self) }, "keep\n")
        }
    }
}
