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
        localVerified: Bool = true, snapshotAge: Double = 2, historicalUpload: Bool = false, offsiteAge: Double = 24
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
                "lastAt": now.addingTimeInterval(-offsiteAge * 3600).timeIntervalSinceReferenceDate,
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
            XCTAssertTrue(checks.checks[0].value.contains("No new transcripts"))
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
            XCTAssertEqual(checks.checks[2].value, state == "pending" ? "Upload queued" : "Upload in progress")
            XCTAssertNil(checks.attentionSentence)
            XCTAssertTrue(checks.details.contains { $0.value.contains("previous.db.gz") })
        }
    }

    func testFailedUploadShowsActualReason() throws {
        let b = try document(attempt: true, pending: "failed", error: "DriveUploadError").backups
        let checks = BrainBarBackupChecks.derive(b, now: now)
        XCTAssertEqual(checks.checks[0].tone, .neutral) // PROBE2: DB failure cannot blame transcripts.
        XCTAssertEqual(checks.checks.first { $0.id == "archives" }?.tone, .neutral)
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
        // Defensive decoder state: real producers do not currently emit nothing_pending.
        let nothing = BrainBarBackupChecks.derive(try document(attempt: true, pending: "nothing_pending").backups, now: now)
        XCTAssertFalse(nothing.summary.contains("Upload in progress"))
        XCTAssertEqual(nothing.checks.first { $0.id == "freshness" }?.tone, .red)
        XCTAssertEqual(nothing.checks[2].tone, .red)
        XCTAssertEqual(nothing.checks[2].value, "No local snapshot found to verify. Run backup-daily or check the snapshot job.")
        XCTAssertEqual(nothing.checks[2].value, nothing.checks.first { $0.id == "freshness" }?.value)
    }

    func testMissingStaleFailedAndOldProducerEvidenceStaysRed() throws {
        let states = try [
            document(), document(attempt: true, attemptAge: 40),
            document(attempt: true, attemptAge: -1), document(attempt: true, freshness: "stale"),
            document(attempt: true, retention: "FAIL"),
            document(attempt: true, error: "job_alert:Weekly maintenance missed its quiet window"),
            document(attempt: true, pending: "pending", localVerified: false),
            document(attempt: true, pending: "pending", snapshotAge: 40),
            document(attempt: true, pending: "in_progress", offsiteAge: 30 * 24),
        ]
        for document in states {
            XCTAssertNotNil(BrainBarBackupChecks.derive(document.backups, now: now).attentionSentence)
        }
    }
    func testUnrelatedErrorsKeepTranscriptCoverageNeutralAndTranscriptErrorsDoNot() throws {
        for error in ["DriveUploadError", "drive_credentials_restored_backup_pending", "job_alert:Weekly maintenance missed its quiet window"] {
            let status = ObservabilityPresentation.backupStatus(for: try document(attempt: true, error: error).backups, now: now)
            XCTAssertEqual(status.upload.tone, .neutral, error)
            XCTAssertEqual(status.archives.tone, .neutral, error)
            XCTAssertEqual(status.freshness.tone, .neutral, error)
            if error.hasPrefix("job_alert:") { XCTAssertEqual(status.lines.filter { $0.tone == .red }.count, 1) }
        }
        let status = ObservabilityPresentation.backupStatus(for: try document(attempt: true, error: "jsonl_backup_attempt_failed").backups, now: now)
        XCTAssertEqual(status.upload.tone, .red)
        XCTAssertEqual(status.archives.tone, .red)
    }

    func testIdleCopyUsesLastUploadAndInjectedLocaleAndCalendar() throws {
        let doc = try document(attempt: true, historicalUpload: true)
        for locale in [Locale(identifier: "en_US"), Locale(identifier: "fr_FR")] {
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = TimeZone(secondsFromGMT: -12 * 3600)!
            let checks = BrainBarBackupChecks.derive(doc.backups, now: now, locale: locale, calendar: calendar)
            let formatter = DateFormatter()
            formatter.calendar = calendar
            formatter.timeZone = calendar.timeZone
            formatter.locale = locale
            formatter.dateFormat = "MMM d"
            XCTAssertEqual(checks.checks[0].value, "No new transcripts since \(formatter.string(from: doc.backups.lastVerifiedUpload!.at)) · checked \(BrainBarBackupChecks.relative(doc.backups.transcriptAttempt!.at, now: now, locale: locale))")
            XCTAssertEqual(checks.checks.first { $0.id == "freshness" }?.title, "Transcript coverage")
            XCTAssertFalse(checks.checks.map(\.value).joined().contains("no-op"))
            let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: now)
            let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now)
            let hero = BrainBarHeroPresentation.derive(flow: flow, stats: collector.stats,
                backupTruth: BrainBarHeroBackupTruth.derive(from: .readable(doc), now: now, cadence: .known(300)))
            let dashboard = BrainBarOnePagePresentation.derive(snapshotFreshness: collector.snapshotFreshnessState,
                hero: hero, observability: .readable(doc), stats: collector.stats, agentActivity: collector.agentActivity,
                now: now, calendar: calendar, locale: locale, observabilityCadence: .known(300))
            formatter.dateFormat = "MMM d 'at' HH:mm"
            XCTAssertEqual(dashboard.backupLines.last?.text, "Transcripts · Drive · no changes since \(formatter.string(from: doc.backups.lastVerifiedUpload!.at))")
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
        XCTAssertEqual(cases.count, 13)
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
            XCTAssertEqual(status.snapshot.tone, name == "stale_offsite" ? .red : .green, name)
            switch name {
            case "stopped":
                XCTAssertEqual(doc.backups.localSnapshot?.uploadState, "failed")
                XCTAssertTrue(status.snapshotUpload?.text.contains("stopped, exit 143") == true)
                XCTAssertEqual(status.snapshotUpload?.tone, .red)
                XCTAssertFalse(status.uploadInProgress)
                XCTAssertEqual(doc.backups.dbSnapshot?.destination, "previous.db.gz")
                XCTAssertNotNil(checks.attentionSentence)
            case "pending", "in_progress":
                XCTAssertEqual(status.snapshotUpload?.tone, .neutral, name)
                XCTAssertEqual(status.snapshotUpload?.text, name == "pending" ? "Upload queued" : "Upload in progress")
                XCTAssertEqual(doc.backups.dbSnapshot?.destination, "previous.db.gz")
                XCTAssertNil(checks.attentionSentence)
            case "failed", "abandoned":
                XCTAssertEqual(status.snapshotUpload?.tone, .red, name)
                XCTAssertTrue(status.snapshotUpload?.text.contains("Synthetic Drive rejected upload") == true)
                XCTAssertEqual(doc.backups.errorType, "SyntheticUploadFailure")
                XCTAssertNotNil(checks.attentionSentence)
            case "stale_offsite":
                XCTAssertEqual(status.snapshotUpload?.tone, .red)
                XCTAssertEqual(status.snapshotUpload?.text, "Last offsite DB copy is 30 days old; upload still pending")
                XCTAssertFalse(status.uploadInProgress)
                let job = BrainLayerLaunchdGroupStatus(health: .healthy, attentionReason: nil, lastRunText: "", nextRunText: "")
                XCTAssertEqual(BrainBarBackupsHealth.derive(job: job, drive: nil, status: status, statusUnavailableReason: "").badge, .attention)
                XCTAssertNotNil(checks.attentionSentence)
            case "m1_weekly_alert":
                XCTAssertEqual(status.lines.filter { $0.tone == .red }.map(\.text), ["Weekly maintenance missed its quiet window"])
                XCTAssertEqual(status.upload.tone, .neutral)
                XCTAssertEqual(status.archives.tone, .neutral)
            case "verified_bundle":
                XCTAssertEqual(status.freshness.tone, .green)
                XCTAssertFalse(checks.checks.first { $0.id == "freshness" }!.value.contains("attempt"))
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
            XCTAssertEqual(badge.badge, state == nil ? .idle : state == "pending" ? .queued : .uploading)
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
            XCTAssertTrue(dashboard.backupLines.last?.text.contains("Transcripts · Drive · no changes") == true)
            // Job-alert reconciliation must retain the new optional evidence.
            XCTAssertNotNil(doc.replacingBackupsErrorType(nil).backups.transcriptAttempt)
        }
    }

    func testNeutralChecksRenderInIsolatedNativeView() throws {
        if ProcessInfo.processInfo.environment["CI"] == "true" { throw XCTSkip("Native render runs locally") }
        let fixture = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures/neutral-backup-states-v2.json")
        let cases = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [[String: Any]])
        let temporary = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let url = temporary.appendingPathComponent("observability.json")
        var documents: [(String, ObservabilityDocument)] = []
        for item in cases {
            try JSONSerialization.data(withJSONObject: XCTUnwrap(item["document"])).write(to: url)
            guard case let .readable(doc) = ObservabilityReader.read(url: url) else { return XCTFail("Render fixture failed to decode") }
            let name = try XCTUnwrap(item["name"] as? String)
            documents.append((name, doc))
        }
        // No real producer emits this state yet; exercise the defensive decoder branch directly.
        documents.append(("nothing_pending", try document(attempt: true, pending: "nothing_pending", historicalUpload: true)))
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(".build.noindex/brainbar-render")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for (name, original) in documents {
            let doc = name == "verified" ? try XCTUnwrap(documents.first { $0.0 == "verified_bundle" }?.1) : original
            for scheme in name == "idle" || name == "pending" ? [ColorScheme.dark, .light] : [.dark] {
                let suffix = scheme == .light ? "-light" : ""
                let checks = BrainBarBackupChecks.derive(doc.backups, now: doc.generatedAt, locale: BrainBarDashboardPreview.goldenLocale,
                    calendar: BrainBarDashboardPreview.goldenCalendar)
                let lines = ObservabilityPresentation.backupStatus(for: doc.backups, now: doc.generatedAt)
                let view = VStack(alignment: .leading, spacing: 18) {
                    Text(name).font(.title2)
                    BrainBarBackupChecksCard(checks: checks, hiddenAlert: nil, detailsExpanded: .constant(true), copy: { _ in })
                    ObservabilityStatusRows(lines: lines.lines)
                }.padding(24).frame(width: 900).background(Color(nsColor: .windowBackgroundColor))
                    .environment(\.colorScheme, scheme)
                try render(view, to: root.appendingPathComponent("neutral-\(name)\(suffix).png"), width: 900)
                if ["idle", "idle_old_upload", "pending"].contains(name) {
                    let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: doc.generatedAt)
                    let dashboard = BrainBarDashboardPreview.make(collector: collector,
                        observabilityResult: .readable(doc), now: doc.generatedAt, observabilityCadence: .known(300), colorScheme: scheme)
                    try render(dashboard, to: root.appendingPathComponent("neutral-dashboard-\(name)\(suffix).png"), width: 960, height: 640)
                }
                if scheme == .dark, ["idle", "in_progress", "m1_weekly_alert", "stale_offsite"].contains(name) {
                    let health = BrainBarBackupsHealth.derive(job: .init(health: .healthy, attentionReason: nil, lastRunText: "", nextRunText: ""),
                        drive: nil, status: lines, statusUnavailableReason: "Missing")
                    let viewModel = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
                        store: BrainLayerConfigStore(configURL: temporary.appendingPathComponent("synthetic.env")),
                        initialLaunchdStates: [.backupDaily: .loaded, .jsonlBackup: .loaded], refreshStatusOnLoad: false,
                        now: { doc.generatedAt }, initialObservabilityResult: .readable(doc))
                    let card = BrainBarJobGroupCard(group: .backups, viewModel: viewModel, backupsHealth: health, backupsReason: health.reason)
                        .padding(24).frame(width: 600).background(Color(nsColor: .windowBackgroundColor)).environment(\.colorScheme, .dark)
                    try render(card, to: root.appendingPathComponent("neutral-jobs-\(name).png"), width: 600)
                }
            }
        }
    }

    private func render<V: View>(_ view: V, to url: URL, width: CGFloat, height: CGFloat? = nil) throws {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: width, height: height ?? host.fittingSize.height)
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(png.count, 10_000)
        try png.write(to: url)
    }
}
