import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusSnapshotVectorTests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests
    let old = "missing embeddings partial: scanned 1864 rows (instruction_limit); not an exact count"

    func testVectorSampleAndCoverageAcceptance() throws {
        for missing in [0, 7] {
            var input = T.healthy()
            input.badge = .init(badgeOn: false, reason: old, activeCodes: ["missing_embeddings_scan_incomplete"],
                                activeMessages: [old], activeSeverities: ["warning"])
            input.badge?.vectorScan = .init(completeness: "partial", scanned: 1864, missing: missing, reason: "instruction_limit")
            if missing > 0 {
                input.badge?.activeMessages.append("At least 7 checked chunks are waiting for vectors. See Vector backlog for the latest full count.")
                input.badge = .init(badgeOn: false, reason: old,
                    activeCodes: ["missing_embeddings_scan_incomplete", "missing_embeddings_backlog"],
                    activeMessages: input.badge!.activeMessages, vectorScan: input.badge!.vectorScan)
            }
            input.vectorCheck = .init(count: 5, checkedAt: T.now, error: nil)
            let snapshot = BrainBarStatusSnapshot.derive(input)
            XCTAssertFalse(snapshot.issues.contains { $0.id == "health.missing_embeddings_scan_incomplete" })
            let detail = try XCTUnwrap(input.badge?.vectorScan?.detail)
            XCTAssertEqual(detail, "Background vector check in progress. Checked 1,864 chunks; \(missing == 0 ? "none" : "7") in this sample are waiting for vectors.")
            XCTAssertEqual(input.vectorCheck?.summary(now: T.now), "5 chunks waiting for vectors · checked 14:10 UTC")
            XCTAssertFalse(snapshot.issues.map(\.text).joined().contains("instruction_limit"))
            XCTAssertEqual(snapshot.issues.contains { $0.text.hasPrefix("At least 7 checked chunks") }, missing > 0)
            input.vectorCheck = .init(count: 5, checkedAt: T.now.addingTimeInterval(-601), error: nil)
            XCTAssertEqual(BrainBarStatusSnapshot.derive(input).issues.filter { $0.text.contains("not verified") || $0.text.contains("could not be verified") }.count, 1)
        }
    }

    func testUnverifiedCoverageAndNonroutineScansStayVisibleWithoutJargon() {
        for scan in [BrainBarVectorScan(completeness: "partial", scanned: 0, missing: 0, reason: "instruction_limit"),
                     .init(completeness: "unknown", scanned: 1864, missing: 0, reason: "unexpected"),
                     .init(completeness: "partial", scanned: 1864, missing: 0, reason: "unexpected")] {
            var input = T.healthy()
            input.badge = .init(badgeOn: false, reason: old, activeCodes: ["missing_embeddings_scan_incomplete"], activeMessages: [old])
            input.badge?.vectorScan = scan
            input.vectorCheck = .init(count: 5, checkedAt: T.now, error: nil)
            let texts = BrainBarStatusSnapshot.derive(input).issues.map(\.text)
            XCTAssertTrue(texts.contains { $0.hasPrefix("The vector check could not finish. The current backlog is not verified by this check.") })
            XCTAssertFalse(texts.joined().contains("instruction_limit"))
        }
        for (at, error) in [(T.now.addingTimeInterval(-601), nil), (nil, "query failed"), (T.now, "query failed")] as [(Date?, String?)] {
            var input = T.healthy()
            input.vectorCheck = .init(count: at == nil ? nil : 5, checkedAt: at, error: error)
            input.badge = .init(badgeOn: false, reason: "SQL check failed", activeCodes: ["missing_embeddings_count_failed"], activeMessages: ["SQL check failed"])
            let texts = BrainBarStatusSnapshot.derive(input).issues.map(\.text)
            XCTAssertTrue(texts.contains("SQL check failed"))
            XCTAssertEqual(texts.filter { $0 == BrainBarVectorCheck.unverified }.count, 1)
            XCTAssertTrue(input.vectorCheck!.summary(now: T.now).hasPrefix(at == nil ? "Vector backlog could not" : "Last measured backlog: 5"))
        }
        XCTAssertTrue(BrainBarVectorCheck(count: 5, checkedAt: T.now.addingTimeInterval(-600), error: nil).isFresh(now: T.now))
        XCTAssertFalse(BrainBarVectorCheck(count: 5, checkedAt: T.now.addingTimeInterval(1), error: nil).isFresh(now: T.now))
        XCTAssertEqual(BrainBarVectorScan(completeness: "complete", scanned: 1864, missing: 7, reason: "").detail,
                       "Background vector check complete. 7 chunks are waiting for vectors.")
    }

    func testIncompleteScanAndCoverageNoticesShareIdentityWithoutHidingMissingRows() throws {
        for error in [nil, "query failed"] as [String?] {
            for reason in ["sql_error", "unexpected"] {
                for stale in [false, true] {
                    for scanPresent in [false, true] where stale || scanPresent {
                        var input = T.healthy()
                        input.vectorCheck = .init(count: 5, checkedAt: stale ? T.now.addingTimeInterval(-601) : T.now,
                                                  error: stale ? error : nil)
                        if scanPresent {
                            input.badge = .init(badgeOn: false, reason: old,
                                activeCodes: ["missing_embeddings_scan_incomplete"], activeMessages: [old],
                                vectorScan: .init(completeness: "unknown", scanned: 10, missing: 0, reason: reason))
                        }
                        let issues = BrainBarStatusSnapshot.derive(input).issues
                        XCTAssertEqual(issues.count, 1)
                        XCTAssertEqual(issues.first?.id, "health.vector-coverage")
                        if scanPresent {
                            XCTAssertTrue(issues[0].text.contains(reason == "sql_error" ? "database query failed" : "stopped unexpectedly"))
                        }
                    }
                }
                var input = T.healthy()
                input.vectorCheck = .init(count: 5, checkedAt: T.now.addingTimeInterval(-601), error: error)
                input.badge = .init(badgeOn: false, reason: old,
                    activeCodes: ["missing_embeddings_scan_incomplete", "missing_embeddings_backlog"],
                    activeMessages: [old, "At least 7 checked chunks are waiting for vectors."],
                    vectorScan: .init(completeness: "partial", scanned: 10, missing: 7, reason: reason))
                let issues = BrainBarStatusSnapshot.derive(input).issues
                XCTAssertEqual(issues.count, 2)
                XCTAssertTrue(issues.contains { $0.id == "health.vector-coverage" })
                XCTAssertTrue(issues.contains { $0.id == "health.missing_embeddings_backlog" && $0.text.contains("At least 7") })
            }
        }
    }

    func testFirstCoverageAttemptDoesNotFlashAnUnverifiedNotice() {
        var input = T.healthy()
        input.vectorCheck = .init(count: nil, checkedAt: nil, error: nil)
        for badge in [nil, BadgeStatePresentation(badgeOn: false, reason: old,
            activeCodes: ["missing_embeddings_scan_incomplete"], activeMessages: [old],
            vectorScan: .init(completeness: "partial", scanned: 1864, missing: 0, reason: "instruction_limit"))] {
            input.badge = badge
            XCTAssertTrue(BrainBarStatusSnapshot.derive(input).issues.isEmpty)
            XCTAssertEqual(input.vectorCheck?.summary(now: T.now), "Checking vector backlog…")
        }
        input.vectorCheck = .init(count: nil, checkedAt: nil, error: "query failed")
        XCTAssertEqual(BrainBarStatusSnapshot.derive(input).issues.filter { $0.text == BrainBarVectorCheck.unverified }.count, 1)
        input.vectorCheck = nil
        XCTAssertTrue(BrainBarStatusSnapshot.derive(input).issues.contains { $0.text.hasPrefix("The vector check could not finish.") })
    }

    func testVectorCopyUsesSingularAndPluralChunks() {
        for count in [0, 1, 2] {
            let noun = count == 1 ? "chunk" : "chunks"
            let verb = count == 1 ? "is" : "are"
            XCTAssertEqual(BrainBarVectorCheck(count: count, checkedAt: T.now, error: nil).summary(now: T.now),
                           "\(count) \(noun) waiting for vectors · checked 14:10 UTC")
            XCTAssertEqual(BrainBarVectorScan(completeness: "complete", scanned: 2, missing: count, reason: "").detail,
                           "Background vector check complete. \(count) \(noun) \(verb) waiting for vectors.")
        }
        XCTAssertEqual(BrainBarVectorScan(completeness: "partial", scanned: 1, missing: 1, reason: "page_limit").detail,
                       "Background vector check in progress. Checked 1 chunk; 1 in this sample is waiting for vectors.")
    }

    func testStructuredScanReadAndReconciliation() throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let badgeURL = root.appendingPathComponent("badge-state.json")
        let badge: [String: Any] = ["schema_version": 1, "generated_at": ISO8601DateFormatter().string(from: T.now),
            "alerts": ["state": "measured", "reason": "", "inputs": [], "badge_on": true, "active": [
                ["code": "missing_embeddings_scan_incomplete", "message": old, "severity": "warning"],
                ["code": "job_alert_index", "message": "Index failed", "severity": "critical"]]]]
        try JSONSerialization.data(withJSONObject: badge).write(to: badgeURL)
        try Data(#"{"missing_embedding_scan":{"completeness":"partial","scanned":1864,"missing":0,"reason":"instruction_limit"},"missing_vectors":99999}"#.utf8)
            .write(to: root.appendingPathComponent("health-check-state.json"))
        let read = BadgeStateReader.read(url: badgeURL, now: T.now, cadence: .known(300)).reconciled(with: .init(active: [:]))
        XCTAssertEqual(read.vectorScan?.scanned, 1864)
        XCTAssertTrue(read.vectorScan?.routinePartial == true)
    }
}
