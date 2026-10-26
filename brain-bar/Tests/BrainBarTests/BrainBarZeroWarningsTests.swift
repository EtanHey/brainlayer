import Darwin
import Foundation
import XCTest
@testable import BrainBar

/// The calendar reader opens every path component with O_NOFOLLOW, so tests hand it a realpath.
private func canonicalTemporaryDirectory(_ prefix: String, _ test: XCTestCase) throws -> URL {
    let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
    defer { free(canonical) }
    let root = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent("\(prefix)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                           attributes: [.posixPermissions: 0o700])
    test.addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    return root
}

/// Serves one fixed sample, so a view refresh can never reach the real launchctl.
private struct FixedActivity: BrainLayerLaunchdStatusSampling {
    let observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]
    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] { observations.mapValues(\.loadState) }
    func sampleActivity() -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] { observations }
}

/// A healthy system shows zero warnings, and a real warning still shows (2026-10-21).
final class BrainBarZeroWarningsTests: XCTestCase {
    private static let base: TimeInterval = 1_784_466_000

    /// The MBP's 10-21 calendar run: every step ok, and the weekly replaced the nightly.
    private static let healthyWeeklyNight = """
    {"event":"runner","ts":\(base - 900),"scheduled_date":"2026-10-21"}
    {"event":"spawn","step":"index","ts":\(base - 899),"pid":4241}
    {"event":"finished","step":"index","ts":\(base - 800),"status":"ok","returncode":0}
    {"event":"spawn","step":"maintenance-weekly","ts":\(base - 700),"pid":4242}
    {"event":"finished","step":"maintenance-weekly","ts":\(base - 600),"status":"ok","returncode":0,"replaces":"maintenance-nightly"}
    {"event":"spawn","step":"jsonl-backup","ts":\(base - 500),"pid":4243}
    {"event":"finished","step":"jsonl-backup","ts":\(base - 400),"status":"ok","returncode":0}
    {"event":"spawn","step":"backup-daily","ts":\(base - 300),"pid":4244}
    {"event":"finished","step":"backup-daily","ts":\(base - 200),"status":"ok","returncode":0}

    """

    /// The 10-20 run Etan's screenshot still showed: daily upload exit 1, weekly VACUUM exit 76.
    private static let failedNight = """
    {"event":"runner","ts":\(base - 90_000),"scheduled_date":"2026-10-20"}
    {"event":"spawn","step":"maintenance-weekly","ts":\(base - 89_000),"pid":4142}
    {"event":"finished","step":"maintenance-weekly","ts":\(base - 88_000),"status":"failed","returncode":76,"replaces":"maintenance-nightly"}
    {"event":"spawn","step":"backup-daily","ts":\(base - 87_000),"pid":4144}
    {"event":"finished","step":"backup-daily","ts":\(base - 86_000),"status":"failed","returncode":1}

    """

    private func history(_ jsonl: String) throws -> BrainBarCalendarHistory {
        let directory = try canonicalTemporaryDirectory("zero-warnings", self)
        try Data(jsonl.utf8).write(to: directory.appendingPathComponent("runs.jsonl"))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(history.error)
        return history
    }

    private func status(_ group: BrainLayerLaunchdJobGroup,
                        _ observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]) -> BrainLayerLaunchdGroupStatus {
        group.status(settings: BrainLayerConfig.defaultConfig.launchdJobs, observations: observations,
                     formatDate: { _ in "Oct 21, 04:17" })
    }

    // MARK: 1. Healthy notes are neutral; amber stays reserved for real warnings.

    func testHealthyOwnerNotesAreInformationalOnEveryGroup() throws {
        let observations = BrainBarSupervisorStatusTests.supervised(history: try history(Self.healthyWeeklyNight)).sampleActivity()
        for group in BrainLayerLaunchdJobGroup.allCases {
            let value = status(group, observations)
            XCTAssertNotNil(value.note, group.rawValue)
            XCTAssertEqual(value.noteTone, .info, "\(group.rawValue): \(value.note ?? "")")
        }
        XCTAssertEqual(observations[.watch]?.ownerNoteTone, .info)
        XCTAssertEqual(observations[.backupDaily]?.ownerNoteTone, .info)
    }

    func testFailedOrStoppedCalendarResultsKeepAnAmberNote() throws {
        for (status, code) in [("failed", Int32(1)), ("stopped", -15), ("ok", 9)] {
            let history = BrainBarCalendarHistory(runs: [
                "backup-daily": .init(at: Date(timeIntervalSince1970: Self.base), code: code, status: status),
            ])
            let observations = BrainBarSupervisorStatusTests.supervised(history: history).sampleActivity()
            XCTAssertEqual(observations[.backupDaily]?.ownerNoteTone, .warning, status)
            let backups = self.status(.backups, observations)
            XCTAssertEqual(backups.health, .unhealthy)
            XCTAssertEqual(backups.noteTone, .warning)
        }
    }

    func testFailedSupervisorComponentKeepsAnAmberNote() {
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { command in
            if command.first == "/bin/launchctl" { return .init(terminationStatus: 0, output: "runs = 0") }
            return .init(terminationStatus: 0, output: """
            {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":null,
            "starts":3,"failures":1,"last_exit_code":9,"last_run_at":1784466000}}
            """)
        }, now: { Date(timeIntervalSince1970: Self.base) }, ownership: { .supervisor },
        calendarHistory: { .empty }, calendarSchedule: { DateComponents(hour: 3, minute: 0) },
        calendarIsDisabled: { _ in false })
        let observations = provider.sampleActivity()
        XCTAssertEqual(observations[.watch]?.ownerNoteTone, .warning)
        XCTAssertEqual(status(.ingest, observations).noteTone, .warning)
    }

    func testRunWarningsAndConfiguredOffStayWarnings() {
        let configured = BrainLayerLaunchdJobGroup.ingest.status(
            settings: [.watch: .init(enabled: false, loadState: .loaded), .index: .init(enabled: false, loadState: .loaded)],
            observations: [:], formatDate: { _ in "" })
        XCTAssertEqual(configured.note, "Configured OFF: Watcher, Index.")
        XCTAssertEqual(configured.noteTone, .warning)
    }

    // MARK: 2. A weekly that replaced the nightly covers it.

    @MainActor
    func testWeeklyThatReplacedNightlyCoversItOnJobsAndBackups() throws {
        let observations = BrainBarSupervisorStatusTests.supervised(history: try history(Self.healthyWeeklyNight)).sampleActivity()
        let weeklyAt = Date(timeIntervalSince1970: Self.base - 600)
        XCTAssertEqual(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt, weeklyAt)
        let maintenance = status(.maintenance, observations)
        XCTAssertEqual(maintenance.health, .healthy)
        XCTAssertNil(maintenance.attentionReason)
        XCTAssertEqual(maintenance.noteTone, .info)
        let note = try XCTUnwrap(maintenance.note)
        XCTAssertFalse(note.contains("Nightly skipped"), note)
        XCTAssertFalse(note.contains("Scheduled, not yet run"), note)
        XCTAssertTrue(note.contains("Nightly covered by weekly"), note)
        XCTAssertTrue(maintenance.lastRunText.hasPrefix("Nightly Covered by weekly · Oct 21, 04:17"), maintenance.lastRunText)
        XCTAssertFalse(maintenance.lastRunText.contains("Not yet run"), maintenance.lastRunText)

        let model = try settingsModel(observations)
        let nightly = try XCTUnwrap(model.presentedBackupSchedules.first { $0.title == "Nightly maintenance" })
        XCTAssertEqual(nightly.lastRun, "Covered by weekly · " + DashboardMetricFormatter.jobDateTimeString(weeklyAt))
        XCTAssertFalse(nightly.cadence.contains("Scheduled, not yet run"), nightly.cadence)
        let weekly = try XCTUnwrap(model.presentedBackupSchedules.first { $0.title == "Weekly maintenance" })
        XCTAssertFalse(weekly.cadence.contains("Nightly skipped"), weekly.cadence)
        XCTAssertEqual(model.groupStatus(.maintenance).health, .healthy)
    }

    func testFailedWeeklyDoesNotCoverNightlyAndStaysAttention() throws {
        let observations = BrainBarSupervisorStatusTests.supervised(history: try history(Self.failedNight)).sampleActivity()
        XCTAssertNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
        let maintenance = status(.maintenance, observations)
        XCTAssertEqual(maintenance.health, .unhealthy)
        XCTAssertEqual(maintenance.noteTone, .warning)
        XCTAssertFalse(maintenance.lastRunText.contains("Covered by weekly"))
    }

    func testNewerSuccessfulWeeklyCoversAnOlderFailedNightly() throws {
        let oldNightly = """
        {"event":"finished","step":"maintenance-nightly","ts":\(Self.base - 1000),"status":"failed","returncode":1}

        """
        let observations = BrainBarSupervisorStatusTests.supervised(
            history: try history(oldNightly + Self.healthyWeeklyNight)).sampleActivity()
        XCTAssertNotNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
        XCTAssertEqual(status(.maintenance, observations).health, .healthy)
        XCTAssertEqual(status(.maintenance, observations).noteTone, .info)
    }

    func testNightlyThatRanAfterTheWeeklyIsItsOwnResult() throws {
        let nightly = """
        {"event":"spawn","step":"maintenance-nightly","ts":\(Self.base - 100),"pid":4250}
        {"event":"finished","step":"maintenance-nightly","ts":\(Self.base - 50),"status":"failed","returncode":1}

        """
        let observations = BrainBarSupervisorStatusTests.supervised(
            history: try history(Self.healthyWeeklyNight + nightly)).sampleActivity()
        XCTAssertNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
        let maintenance = status(.maintenance, observations)
        XCTAssertEqual(maintenance.health, .unhealthy)
        XCTAssertEqual(maintenance.noteTone, .warning)
    }

    @MainActor
    func testReviewerAClockRollbackKeepsLaterFailedAndStoppedNightlyVisible() throws {
        for (result, code) in [("failed", 1), ("stopped", -15)] {
            try assertNightlyWarning(history("""
            {"event":"runner","ts":1790769600.0,"scheduled_date":"2026-10-21"}
            {"event":"spawn","step":"maintenance-weekly","ts":1790772600.0,"pid":4242}
            {"event":"finished","step":"maintenance-weekly","ts":1790773200.0,"status":"ok","returncode":0,"replaces":"maintenance-nightly"}
            {"event":"runner","ts":1790769800.0,"scheduled_date":"2026-10-21"}
            {"event":"spawn","step":"maintenance-nightly","ts":1790769900.0,"pid":4250}
            {"event":"finished","step":"maintenance-nightly","ts":1790770000.0,"status":"\(result)","returncode":\(code)}

            """), result: result, code: Int32(code))
        }
    }

    @MainActor
    func testReviewerBClockRollbackKeepsLaterFailedAndStoppedNightlyVisible() throws {
        for (result, code) in [("failed", 1), ("stopped", -15)] {
            try assertNightlyWarning(history("""
            {"event":"runner","ts":1792548000,"scheduled_date":"2026-10-21"}
            {"event":"spawn","step":"maintenance-weekly","ts":1792548010,"pid":4242}
            {"event":"finished","step":"maintenance-weekly","ts":1792548100,"status":"ok","returncode":0,"replaces":"maintenance-nightly"}
            {"event":"runner","ts":1792548050,"scheduled_date":"2026-10-21"}
            {"event":"spawn","step":"maintenance-nightly","ts":1792548060,"pid":4250}
            {"event":"finished","step":"maintenance-nightly","ts":1792548070,"status":"\(result)","returncode":\(code)}

            """), result: result, code: Int32(code))
        }
    }

    @MainActor
    func testMissingJournalOrderCannotCoverAFailedNightly() throws {
        let history = BrainBarCalendarHistory(runs: [
            "maintenance-weekly": .init(at: Date(timeIntervalSince1970: Self.base), code: 0, status: "ok",
                startedAt: Date(timeIntervalSince1970: Self.base - 100), replaces: "maintenance-nightly",
                scheduledDate: "2026-10-21"),
            "maintenance-nightly": .init(at: Date(timeIntervalSince1970: Self.base - 50), code: 1, status: "failed"),
        ], latestScheduledDate: "2026-10-21")
        try assertNightlyWarning(history, result: "failed", code: 1)
    }

    @MainActor
    private func assertNightlyWarning(_ history: BrainBarCalendarHistory, result: String, code: Int32,
                                     file: StaticString = #filePath, line: UInt = #line) throws {
        let observations = BrainBarSupervisorStatusTests.supervised(history: history).sampleActivity()
        let expected = "Maintenance Nightly: calendar result \(result), exit \(code)"
        let nightly = try XCTUnwrap(observations[.maintenanceNightly])
        XCTAssertNil(nightly.calendarCoveredByWeeklyAt, file: file, line: line)
        XCTAssertEqual(nightly.loadState, .failed(expected), file: file, line: line)
        XCTAssertEqual(nightly.lastExitCode, code, file: file, line: line)
        XCTAssertEqual(nightly.ownerNoteTone, .warning, file: file, line: line)
        let maintenance = status(.maintenance, observations)
        XCTAssertEqual(maintenance.health, .unhealthy, file: file, line: line)
        XCTAssertEqual(maintenance.noteTone, .warning, file: file, line: line)
        XCTAssertEqual(maintenance.attentionReason, expected, file: file, line: line)
        // S5: the raw group still names the failure; the shared snapshot shows a single failed
        // nightly as neutral text that retries, never hidden and never "covered".
        XCTAssertFalse(maintenance.lastRunText.contains("Covered by weekly"), file: file, line: line)
        let model = try settingsModel(observations)
        let schedule = try XCTUnwrap(model.presentedBackupSchedules.first { $0.title == "Nightly maintenance" })
        XCTAssertFalse(schedule.lastRun.contains("Covered by weekly"), file: file, line: line)
        XCTAssertEqual(model.groupStatus(.maintenance).health, .notice, file: file, line: line)
        XCTAssertTrue(model.groupStatus(.maintenance).attentionReason?.hasPrefix(expected) == true, file: file, line: line)
        XCTAssertTrue(model.publishedStatus.verdict.notices.contains { $0.hasPrefix(expected) }, file: file, line: line)
    }

    func testWeeklyCoverageExpiresAfterALaterScheduledNightWithoutNightly() throws {
        let laterNight = """
        {"event":"runner","ts":\(Self.base + 86400),"scheduled_date":"2026-10-22"}
        {"event":"finished","step":"maintenance-nightly","ts":\(Self.base + 86500),"status":"skipped","returncode":null}

        """
        for oldNightly in ["", "{\"event\":\"finished\",\"step\":\"maintenance-nightly\",\"ts\":\(Self.base - 1000),\"status\":\"failed\",\"returncode\":1}\n"] {
            let observations = BrainBarSupervisorStatusTests.supervised(
                history: try history(oldNightly + Self.healthyWeeklyNight + laterNight)).sampleActivity()
            XCTAssertNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
            let maintenance = status(.maintenance, observations)
            XCTAssertFalse(maintenance.lastRunText.contains("Covered by weekly"))
            XCTAssertEqual(maintenance.health, oldNightly.isEmpty ? .awaitingRun : .unhealthy)
        }
    }

    func testRetryOfSameScheduledNightKeepsWeeklyCoverage() throws {
        let retry = "{\"event\":\"runner\",\"ts\":\(Self.base + 100),\"scheduled_date\":\"2026-10-21\"}\n"
        let observations = BrainBarSupervisorStatusTests.supervised(
            history: try history(Self.healthyWeeklyNight + retry)).sampleActivity()
        XCTAssertNotNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
        XCTAssertEqual(status(.maintenance, observations).health, .healthy)
    }

    func testWeeklyWithoutScheduledDateCannotClaimCoverage() throws {
        let unknownNight = Self.healthyWeeklyNight.replacingOccurrences(of: ",\"scheduled_date\":\"2026-10-21\"", with: "")
        let observations = BrainBarSupervisorStatusTests.supervised(history: try history(unknownNight)).sampleActivity()
        XCTAssertNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
        XCTAssertEqual(status(.maintenance, observations).health, .awaitingRun)
    }

    func testNeverRunNightlyIsNeutralScheduledInfo() {
        let observations = BrainBarSupervisorStatusTests.supervised().sampleActivity()
        XCTAssertNil(observations[.maintenanceNightly]?.calendarCoveredByWeeklyAt)
        XCTAssertTrue(observations[.maintenanceNightly]?.ownerNote?.contains("Scheduled, not yet run under the supervisor") == true)
        let maintenance = status(.maintenance, observations)
        XCTAssertEqual(maintenance.health, .awaitingRun)
        XCTAssertEqual(maintenance.noteTone, .info)
    }

    // MARK: 3. Dashboard reasons re-derive from the newest sample on every tick.

    @MainActor
    func testDashboardJobReasonsRederiveFromTheNewestSampleOnEveryTick() async throws {
        let samples = SampleSequence([try history(Self.failedNight), try history(Self.failedNight + Self.healthyWeeklyNight)])
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { command in
            if command.first == "/bin/launchctl" { return .init(terminationStatus: 0, output: "runs = 0") }
            return .init(terminationStatus: 0, output: """
            {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":4242,
            "starts":3,"failures":0,"last_exit_code":0,"last_run_at":1784466000}}
            """)
        }, now: { Date(timeIntervalSince1970: Self.base) }, ownership: { .supervisor },
        calendarHistory: { samples.next() }, calendarSchedule: { DateComponents(hour: 3, minute: 0) },
        calendarIsDisabled: { _ in false })
        let model = try settingsModel([:], provider: provider)
        let seen = ReasonLog()
        await model.monitorLaunchdStatus(waitForNextTick: {
            let reasons = await MainActor.run { model.dashboardJobAttentionReasons }
            if seen.append(reasons) >= 2 { throw CancellationError() }
        })
        let ticks = seen.values
        XCTAssertEqual(ticks.count, 2)
        XCTAssertTrue(ticks[0].contains("Backup Daily: calendar result failed, exit 1"), "\(ticks[0])")
        XCTAssertTrue(ticks[0].contains { $0.contains("exit 76") }, "\(ticks[0])")
        XCTAssertEqual(ticks[1], [], "A newer ok sample must clear the reasons on the next tick")
    }

    func testDashboardRefreshCadenceIsNotRebuiltInTheViewBody() throws {
        // `Timer.publish(...).autoconnect()` inside `body` is a new timer on every re-render; the
        // collector re-renders the Dashboard far more often than every 60 s, so it never fired.
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/BrainBarWindowRootView.swift"), encoding: .utf8)
        let start = try XCTUnwrap(source.range(of: "private struct BrainBarJobsDashboardContent"))
        let end = try XCTUnwrap(source.range(of: "\n}\n", range: start.upperBound..<source.endIndex))
        let body = source[start.lowerBound..<end.upperBound]
        XCTAssertFalse(body.contains("Timer.publish"), String(body))
        // S5: the one cadence is owned for the app's lifetime, not by a mounted view.
        XCTAssertFalse(body.contains("monitorLaunchdStatus"), String(body))
        let app = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/BrainBarApp.swift"), encoding: .utf8)
        XCTAssertTrue(app.contains("statusSamplingTask = Task { await jobsModel.monitorStatus() }"))
        XCTAssertTrue(app.contains("statusSamplingTask?.cancel()"))
    }

    // MARK: Helpers

    @MainActor
    private func settingsModel(_ observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation],
                               provider: (any BrainLayerLaunchdStatusSampling)? = nil) throws -> BrainBarSettingsViewModel {
        let root = try canonicalTemporaryDirectory("zero-warnings-model", self)
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        return BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: provider ?? FixedActivity(observations: observations),
            initialLaunchdObservations: observations, refreshStatusOnLoad: false,
            initialWatcherHealth: .readable(.init(updatedAt: Date(), pollCount: 50)))
    }
}

private final class SampleSequence: @unchecked Sendable {
    private let lock = NSLock()
    private var remaining: [BrainBarCalendarHistory]
    init(_ samples: [BrainBarCalendarHistory]) { remaining = samples }
    func next() -> BrainBarCalendarHistory {
        lock.lock(); defer { lock.unlock() }
        return remaining.count > 1 ? remaining.removeFirst() : remaining[0]
    }
}

private final class ReasonLog: @unchecked Sendable {
    private let lock = NSLock()
    private var log: [[String]] = []
    var values: [[String]] { lock.lock(); defer { lock.unlock() }; return log }
    func append(_ reasons: [String]) -> Int {
        lock.lock(); defer { lock.unlock() }
        log.append(reasons)
        return log.count
    }
}

/// #1420: the DB copy's age reaches the Dashboard dot and the Database copy check, in human words.
@MainActor
final class BrainBarDBCopyAgeTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_790_769_600)

    /// The production shape from #1420: no `local_snapshot`, no `transcript_attempt`.
    private func document(offsiteAge: Double, freshness: String, error: String?) throws -> ObservabilityDocument {
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else {
            throw NSError(domain: "fixture", code: 1)
        }
        var root = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
        root["generatedAt"] = now.timeIntervalSinceReferenceDate
        root["schemaVersion"] = 2
        var backups = try XCTUnwrap(root["backups"] as? [String: Any])
        backups["freshness"] = freshness
        backups["retentionInvariant"] = "PASS"
        backups["errorType"] = error as Any? ?? NSNull()
        backups["thresholdHours"] = 36
        backups["survivingArchives30D"] = 28
        backups["lastVerifiedUpload"] = ["at": now.addingTimeInterval(-3600).timeIntervalSinceReferenceDate,
                                         "ageHours": 1, "archiveId": "archive", "verified": true]
        backups["launchd"] = ["label": "com.brainlayer.jsonl-backup", "bootstrapped": true, "disabledDirPresent": false]
        backups["dbSnapshot"] = ["lastAt": now.addingTimeInterval(-offsiteAge * 3600).timeIntervalSinceReferenceDate,
                                 "destination": "2026-10-17.db.zst", "verified": true]
        backups.removeValue(forKey: "localSnapshot")
        backups.removeValue(forKey: "transcriptAttempt")
        root["backups"] = backups
        return try JSONDecoder().decode(ObservabilityDocument.self, from: JSONSerialization.data(withJSONObject: root))
    }

    private func dashboard(_ doc: ObservabilityDocument) -> BrainBarOnePagePresentation {
        let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: now)
        let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now)
        let hero = BrainBarHeroPresentation.derive(flow: flow, stats: collector.stats,
            backupTruth: BrainBarHeroBackupTruth.derive(from: .readable(doc), now: now, cadence: .known(300)))
        return BrainBarOnePagePresentation.derive(snapshotFreshness: collector.snapshotFreshnessState,
            hero: hero, observability: .readable(doc), stats: collector.stats, agentActivity: collector.agentActivity,
            now: now, observabilityCadence: .known(300))
    }

    func testStaleDBCopyTurnsTheDashboardDotAndDatabaseCopyCheckRed() throws {
        let doc = try document(offsiteAge: 87, freshness: "stale",
            error: "db_snapshot_stale: newest verified DB copy 2026-10-17.db.zst is 87.0 h old (threshold 36 h)")
        let status = ObservabilityPresentation.backupStatus(for: doc.backups, now: now)
        XCTAssertEqual(status.snapshot.tone, .red)
        XCTAssertEqual(dashboard(doc).backupLines.first?.tone, .red, "\(dashboard(doc).backupLines)")
        let checks = BrainBarBackupChecks.derive(doc.backups, now: now)
        let copy = try XCTUnwrap(checks.checks.first { $0.id == "snapshot" })
        XCTAssertEqual(copy.tone, .red)
        XCTAssertTrue(copy.value.contains("3 days old"), copy.value)
    }

    func testFreshDBCopyStaysGreen() throws {
        let doc = try document(offsiteAge: 2, freshness: "fresh", error: nil)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: doc.backups, now: now).snapshot.tone, .green)
        XCTAssertEqual(dashboard(doc).backupLines.first?.tone, .green)
        XCTAssertEqual(BrainBarBackupChecks.derive(doc.backups, now: now).checks.first { $0.id == "snapshot" }?.tone, .green)
    }

    func testDBSnapshotErrorTypesReadAsHumanSentences() throws {
        let cases: [(String, String)] = [
            ("db_snapshot_stale: newest verified DB copy x.db.zst is 87.0 h old (threshold 36 h)",
             "The database copy in Google Drive is older than 36 hours"),
            ("db_snapshot_missing: no verified offsite DB copy on record (threshold 36 h)",
             "No verified database copy in Google Drive"),
            ("db_snapshot_unverified: newest offsite DB copy x.db.zst failed verification",
             "The newest database copy in Google Drive failed verification"),
            ("db_snapshot_clock_skew: newest verified DB copy x.db.zst is dated after generated_at",
             "The newest database copy is dated in the future; check this Mac's clock"),
        ]
        for (code, sentence) in cases {
            let doc = try document(offsiteAge: 87, freshness: "stale", error: code)
            let status = ObservabilityPresentation.backupStatus(for: doc.backups, now: now)
            XCTAssertEqual(status.error?.text, sentence, code)
            XCTAssertEqual(status.error?.tone, .red, code)
            let alert = try XCTUnwrap(BrainBarBackupChecks.derive(doc.backups, now: now).alert)
            XCTAssertEqual(alert, sentence, code)
            XCTAssertFalse(alert.contains("reported an error"), alert)
            XCTAssertFalse(alert.contains("Db Snapshot"), alert)
        }
    }
}

/// Before/after evidence: the Jobs and Backups pages rendered from the MBP's 10-21 calendar run.
@MainActor
final class BrainBarZeroWarningsRenderTests: XCTestCase {
    func testHealthyJobsAndBackupsRenderWithoutWarningText() throws {
        let root = try canonicalTemporaryDirectory("zero-warnings-render", self)
        let base: TimeInterval = 1_784_466_000
        try Data("""
        {"event":"runner","ts":\(base - 900),"scheduled_date":"2026-10-21"}
        {"event":"spawn","step":"index","ts":\(base - 899),"pid":4241}
        {"event":"finished","step":"index","ts":\(base - 800),"status":"ok","returncode":0}
        {"event":"spawn","step":"maintenance-weekly","ts":\(base - 700),"pid":4242}
        {"event":"finished","step":"maintenance-weekly","ts":\(base - 600),"status":"ok","returncode":0,"replaces":"maintenance-nightly"}
        {"event":"spawn","step":"jsonl-backup","ts":\(base - 500),"pid":4243}
        {"event":"finished","step":"jsonl-backup","ts":\(base - 400),"status":"ok","returncode":0}
        {"event":"spawn","step":"backup-daily","ts":\(base - 300),"pid":4244}
        {"event":"finished","step":"backup-daily","ts":\(base - 200),"status":"ok","returncode":0}

        """.utf8).write(to: root.appendingPathComponent("runs.jsonl"))
        let observations = BrainBarSupervisorStatusTests.supervised(
            history: BrainBarCalendarHistory.read(directory: root)).sampleActivity()
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: FixedActivity(observations: observations),
            initialLaunchdObservations: observations, refreshStatusOnLoad: false,
            initialObservabilityResult: BrainBarDashboardFixture.healthyObservabilityResult,
            initialWatcherHealth: .readable(.init(updatedAt: Date(), pollCount: 50)))
        let output = ProcessInfo.processInfo.environment["BRAINBAR_ZERO_WARNINGS_RENDER_DIR"].map { URL(fileURLWithPath: $0) } ?? root
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        for section in [BrainBarSettingsSection.jobs, .backups] {
            let text = try BrainBarRenderHarness.renderSettingsText(model, section: section, directory: output)
            try text.write(to: output.appendingPathComponent("supervised-\(section.rawValue).txt"), atomically: true, encoding: .utf8)
            XCTAssertFalse(text.contains("Nightly skipped"), text)
            XCTAssertFalse(text.contains("No calendar run recorded"), text)
            XCTAssertFalse(text.contains("Scheduled, not yet run"), text)
            XCTAssertFalse(text.contains("Awaiting next run"), text)
            XCTAssertTrue(text.localizedCaseInsensitiveContains("covered by weekly"), text)
        }
    }
}
