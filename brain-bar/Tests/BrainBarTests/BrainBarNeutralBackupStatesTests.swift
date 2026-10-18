import AppKit
import Foundation
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarNeutralBackupStatesTests: XCTestCase {
    let now = Date(timeIntervalSince1970: 1_790_769_600)

    func document(
        attempt: Bool = false, pending: String? = nil, error: String? = nil,
        freshness: String = "fresh", retention: String = "PASS", attemptAge: Double = 1,
        localVerified: Bool = true, snapshotAge: Double = 2, historicalUpload: Bool = false
    ) throws -> ObservabilityDocument {
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else {
            throw NSError(domain: "fixture", code: 1)
        }
        var root = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
        root["generatedAt"] = now.timeIntervalSinceReferenceDate
        root["schemaVersion"] = 2
        var backups = try XCTUnwrap(root["backups"] as? [String: Any])
        backups["freshness"] = freshness
        backups["retentionInvariant"] = retention
        backups["errorType"] = error as Any? ?? NSNull()
        backups["lastVerifiedUpload"] = historicalUpload ? [
            "at": now.addingTimeInterval(-40 * 86400).timeIntervalSinceReferenceDate,
            "ageHours": 960, "archiveId": "real-old-archive", "verified": true,
        ] : NSNull() as Any
        backups["survivingArchives30D"] = 0
        backups["thresholdHours"] = 36
        backups["launchd"] = ["label": "com.brainlayer.jsonl-backup", "bootstrapped": true, "disabledDirPresent": false]
        if attempt {
            backups["transcriptAttempt"] = [
                "at": now.addingTimeInterval(-attemptAge * 3600).timeIntervalSinceReferenceDate,
                "status": "no-op", "verified": true, "uploaded": false,
            ]
        }
        backups["dbSnapshot"] = [
            "lastAt": now.addingTimeInterval(-snapshotAge * 3600).timeIntervalSinceReferenceDate,
            "destination": "local.db.gz", "verified": pending == nil,
        ]
        if let pending {
            backups["localSnapshot"] = [
                "lastAt": now.addingTimeInterval(-snapshotAge * 3600).timeIntervalSinceReferenceDate,
                "destination": "local.db.gz", "localVerified": localVerified, "uploaded": false,
                "uploadState": pending,
                "uploadError": ["failed", "abandoned"].contains(pending) ? "Drive rejected upload" : NSNull() as Any,
                "localIntegrity": "ok", "localIntegrityPragma": "quick_check",
            ]
            backups["dbSnapshot"] = [
                "lastAt": now.addingTimeInterval(-86400).timeIntervalSinceReferenceDate,
                "destination": "previous.db.gz", "verified": true,
            ]
        }
        root["backups"] = backups
        return try JSONDecoder().decode(ObservabilityDocument.self, from: JSONSerialization.data(withJSONObject: root))
    }

    func testVerifiedIdleNoOpIsNeutralWithoutInventingCoverage() throws {
        for history in [false, true] {
            let b = try document(attempt: true, historicalUpload: history).backups
            let checks = BrainBarBackupChecks.derive(b, now: now)
            XCTAssertEqual(checks.checks[0].tone, .neutral)
            XCTAssertTrue(checks.checks[0].value.contains("Idle"))
            XCTAssertEqual(checks.checks[5].tone, .neutral)
            XCTAssertEqual(checks.checks[5].value, "None")
            XCTAssertFalse(checks.checks[3].value.contains("Both copies"))
            XCTAssertNil(checks.attentionSentence)
            if history { XCTAssertTrue(checks.details.contains { $0.value == "real-old-archive" }) }
            else { XCTAssertFalse(checks.details.contains { $0.label == "Last transcript upload" }) }
        }
    }

    func testPendingAndInProgressLocalCopiesAreNeutralAndKeepOffsiteHistory() throws {
        for state in ["pending", "in_progress"] {
            let b = try document(attempt: true, pending: state).backups
            let checks = BrainBarBackupChecks.derive(b, now: now)
            XCTAssertEqual(checks.checks[1].tone, .green)
            XCTAssertEqual(checks.checks[2].tone, .neutral)
            XCTAssertEqual(checks.checks[2].value, "Local snapshot verified, upload in progress")
            XCTAssertNil(checks.attentionSentence)
            XCTAssertTrue(checks.details.contains { $0.value.contains("previous.db.gz") })
        }
    }

    func testFailedUploadShowsActualReason() throws {
        let b = try document(attempt: true, pending: "failed", error: "DriveUploadError").backups
        let checks = BrainBarBackupChecks.derive(b, now: now)
        XCTAssertEqual(checks.checks[1].tone, .green)
        XCTAssertEqual(checks.checks[2].tone, .red)
        XCTAssertTrue(checks.checks[2].value.contains("Drive rejected upload"))
        XCTAssertNotNil(checks.attentionSentence)
    }

    func testAbandonedAndNothingPendingNeverClaimFreshLocalProof() throws {
        let abandoned = BrainBarBackupChecks.derive(try document(attempt: true, pending: "abandoned", localVerified: false).backups, now: now)
        XCTAssertEqual(abandoned.checks[2].tone, .red)
        XCTAssertTrue(abandoned.checks[2].value.contains("abandoned"))
        XCTAssertTrue(abandoned.checks[2].value.contains("Drive rejected upload"))
        let nothing = BrainBarBackupChecks.derive(try document(attempt: true, pending: "nothing_pending").backups, now: now)
        XCTAssertFalse(nothing.summary.contains("Upload in progress"))
        XCTAssertEqual(nothing.checks.first { $0.id == "freshness" }?.tone, .red)
    }

    func testMissingStaleFailedAndOldProducerEvidenceStaysRed() throws {
        let states = try [
            document(), document(attempt: true, attemptAge: 40),
            document(attempt: true, attemptAge: -1), document(attempt: true, freshness: "stale"),
            document(attempt: true, retention: "FAIL"),
            document(attempt: true, error: "job_alert:Weekly maintenance missed its quiet window"),
            document(attempt: true, pending: "pending", localVerified: false),
            document(attempt: true, pending: "pending", snapshotAge: 40),
        ]
        for document in states {
            XCTAssertNotNil(BrainBarBackupChecks.derive(document.backups, now: now).attentionSentence)
        }
    }
    func testV2OptionalEvidenceSurvivesSnakeCaseWireAndOldDocumentsStayRed() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("observability.json")
        for attempt in [false, true] {
            let doc = try document(attempt: attempt, pending: "pending")
            try encoder.encode(doc).write(to: url)
            guard case let .readable(decoded) = ObservabilityReader.read(url: url) else {
                return XCTFail("Wire document must decode")
            }
            let checks = BrainBarBackupChecks.derive(decoded.backups, now: now)
            XCTAssertEqual(checks.checks[0].tone, attempt ? .neutral : .red)
            XCTAssertEqual(decoded.backups.transcriptAttempt != nil, attempt)
            XCTAssertEqual(decoded.backups.localSnapshot?.uploadState, "pending")
            XCTAssertEqual(checks.checks[1].tone, .green)
            XCTAssertEqual(checks.checks[2].tone, .neutral)
        }
    }

    func testRealPythonProducerFixturesDecodeAndKeepOffsiteHistory() throws {
        let fixture = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures/neutral-backup-states-v2.json")
        let cases = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [[String: Any]])
        XCTAssertEqual(cases.count, 9)
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("observability.json")
        for item in cases {
            let name = try XCTUnwrap(item["name"] as? String)
            try JSONSerialization.data(withJSONObject: XCTUnwrap(item["document"])).write(to: url)
            guard case let .readable(doc) = ObservabilityReader.read(url: url) else {
                return XCTFail("Real producer JSON failed to decode: \(name)")
            }
            let checks = BrainBarBackupChecks.derive(doc.backups, now: doc.generatedAt)
            let status = ObservabilityPresentation.backupStatus(for: doc.backups, now: doc.generatedAt)
            XCTAssertEqual(status.snapshot.tone, .green, name)
            switch name {
            case "pending", "in_progress":
                XCTAssertEqual(status.snapshotUpload?.tone, .neutral, name)
                XCTAssertEqual(status.snapshotUpload?.text, "Local snapshot verified, upload in progress")
                XCTAssertEqual(doc.backups.dbSnapshot?.destination, "previous.db.gz")
                XCTAssertNil(checks.attentionSentence)
            case "failed", "abandoned":
                XCTAssertEqual(status.snapshotUpload?.tone, .red, name)
                XCTAssertTrue(status.snapshotUpload?.text.contains("Synthetic Drive rejected upload") == true)
                XCTAssertEqual(doc.backups.errorType, "SyntheticUploadFailure")
                XCTAssertNotNil(checks.attentionSentence)
            case "verified":
                XCTAssertNil(status.snapshotUpload)
                XCTAssertEqual(doc.backups.dbSnapshot?.destination, "local.db.gz")
            case "legacy":
                XCTAssertEqual(status.upload.tone, .red)
                XCTAssertNil(doc.backups.transcriptAttempt)
            default:
                XCTAssertEqual(status.upload.tone, .neutral, name)
                XCTAssertEqual(doc.backups.survivingArchives30D, 0)
            }
        }
    }

    func testDashboardAndJobsShareNeutralStatesAndRealFailuresWin() throws {
        for state in [nil, "pending", "in_progress"] as [String?] {
            let doc = try document(attempt: true, pending: state)
            let lines = ObservabilityPresentation.backupStatus(for: doc.backups, now: now)
            let healthy = BrainLayerLaunchdGroupStatus(health: .healthy, attentionReason: nil, lastRunText: "", nextRunText: "")
            let badge = BrainBarBackupsHealth.derive(job: healthy, drive: nil, status: lines, statusUnavailableReason: "missing")
            XCTAssertEqual(badge.badge, state == nil ? .idle : .uploading)
            let failed = BrainLayerLaunchdGroupStatus(health: .unhealthy,
                attentionReason: "Weekly maintenance missed its quiet window", lastRunText: "", nextRunText: "")
            XCTAssertEqual(BrainBarBackupsHealth.derive(job: failed, drive: nil, status: lines, statusUnavailableReason: "").badge, .attention)
            let drive = DriveAuthPresentation.derive(status: .init(state: .invalid, reason: "Expired", expiresAt: nil),
                isReconnecting: false, lastOutcome: nil, now: now, formatDate: { _ in "" })
            XCTAssertEqual(BrainBarBackupsHealth.derive(job: healthy, drive: drive, status: lines, statusUnavailableReason: "").badge, .attention)
            let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: now)
            let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now)
            let hero = BrainBarHeroPresentation.derive(flow: flow, stats: collector.stats,
                backupTruth: BrainBarHeroBackupTruth.derive(from: .readable(doc), now: now, cadence: .known(300)))
            let dashboard = BrainBarOnePagePresentation.derive(snapshotFreshness: collector.snapshotFreshnessState,
                hero: hero, observability: .readable(doc), stats: collector.stats, agentActivity: collector.agentActivity,
                now: now, observabilityCadence: .known(300))
            XCTAssertEqual(dashboard.backupLines.map(\.tone), state == nil ? [.green, .neutral] : [.green, .neutral, .neutral])
            XCTAssertTrue(dashboard.backupLines.last?.text.contains("Idle") == true)
            // Job-alert reconciliation must retain the new optional evidence.
            XCTAssertNotNil(doc.replacingBackupsErrorType(nil).backups.transcriptAttempt)
        }
    }

    func testNeutralChecksRenderInIsolatedNativeView() throws {
        if ProcessInfo.processInfo.environment["CI"] == "true" { throw XCTSkip("Native render runs locally") }
        for state in [nil, "pending", "in_progress", "failed", "abandoned", "nothing_pending"] as [String?] {
            let doc = try document(attempt: true, pending: state, error: state == "failed" ? "DriveUploadError" : nil, historicalUpload: true)
            let checks = BrainBarBackupChecks.derive(doc.backups, now: now, locale: Locale(identifier: "en_US"))
            let lines = ObservabilityPresentation.backupStatus(for: doc.backups, now: now)
            let view = VStack(alignment: .leading, spacing: 18) {
                Text(state ?? "Verified idle no-op").font(.title2)
                BrainBarBackupChecksCard(checks: checks, hiddenAlert: nil, detailsExpanded: .constant(true), copy: { _ in })
                ObservabilityStatusRows(lines: lines.lines)
            }.padding(24).frame(width: 900).background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, .dark)
            let host = NSHostingView(rootView: view)
            host.frame = NSRect(x: 0, y: 0, width: 900, height: host.fittingSize.height)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
            host.layoutSubtreeIfNeeded()
            let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            XCTAssertGreaterThan(png.count, 10_000)
            let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("docs.local/brainbar-render")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try png.write(to: root.appendingPathComponent("neutral-\(state ?? "idle").png"))
            if state == nil || state == "pending" {
                let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: now)
                let dashboard = NSHostingView(rootView: BrainBarDashboardPreview.make(collector: collector,
                    observabilityResult: .readable(doc), now: now, observabilityCadence: .known(300)))
                dashboard.frame = NSRect(x: 0, y: 0, width: 960, height: 640)
                dashboard.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
                dashboard.layoutSubtreeIfNeeded()
                let image = try XCTUnwrap(dashboard.bitmapImageRepForCachingDisplay(in: dashboard.bounds))
                dashboard.cacheDisplay(in: dashboard.bounds, to: image)
                let data = try XCTUnwrap(image.representation(using: .png, properties: [:]))
                XCTAssertGreaterThan(data.count, 10_000)
                try data.write(to: root.appendingPathComponent("neutral-dashboard-\(state ?? "idle").png"))
            }
        }
    }

}
