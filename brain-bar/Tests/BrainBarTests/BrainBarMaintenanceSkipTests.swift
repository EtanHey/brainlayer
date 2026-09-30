import XCTest
@testable import BrainBar

/// Jobs → Maintenance: a deliberate gate deferral (exit 75) renders neutral, not "Needs attention";
/// exit 76 (weekly VACUUM skipped because the fresh backup failed) stays attention with human text;
/// and skips never hide a weekly pass that has stopped completing.
final class BrainBarMaintenanceSkipTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
    private let lastRun = ISO8601DateFormatter().date(from: "2026-09-29T21:07:23Z")!
    private let show: (Date) -> String = { ISO8601DateFormatter().string(from: $0) }
    private let quietWindow = "outside quiet window: now=2026-09-30T00:07:23.566976+03:00 start_hour=4 duration_minutes=120"

    private func daysAgo(_ days: Double) -> Date { now.addingTimeInterval(-days * 86_400) }

    private func observation(exit: Int32?, runs: Int? = 3) -> BrainLayerLaunchdJobObservation {
        .init(loadState: .loaded, runs: runs, lastExitCode: exit, lastRunAt: lastRun, nextRunAt: nil, isContinuous: false)
    }

    private func maintenance(
        nightly: Int32? = 0,
        weekly: Int32? = 0,
        completion: BrainLayerMaintenanceEvidence.WeeklyCompletion,
        reasons: [BrainLayerLaunchdJob: String] = [:]
    ) -> BrainLayerLaunchdGroupStatus {
        BrainLayerLaunchdJobGroup.maintenance.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: [.maintenanceNightly: observation(exit: nightly), .maintenanceWeekly: observation(exit: weekly)],
            formatDate: show,
            maintenance: .init(weeklyCompletion: completion, abortReasons: reasons),
            now: now
        )
    }

    // MARK: exit 75 is a skip, not a failure

    func testWeeklyGateDeferralIsSkippedWithItsOwnReason() {
        let status = maintenance(weekly: 75, completion: .completed(daysAgo(2)), reasons: [.maintenanceWeekly: quietWindow])
        XCTAssertEqual(status.health, .skipped)
        XCTAssertEqual(status.health.title, "Skipped")
        XCTAssertEqual(status.attentionReason, "Weekly skipped at \(show(lastRun)): outside the 04:00–06:00 quiet window.")
    }

    func testOtherGateDeferralsKeepTheirReasonVerbatim() {
        let status = maintenance(
            nightly: 75, completion: .completed(daysAgo(2)),
            reasons: [.maintenanceNightly: "recent queue write activity: 5 file(s) modified recently"]
        )
        XCTAssertEqual(status.health, .skipped)
        XCTAssertEqual(status.attentionReason,
                       "Nightly skipped at \(show(lastRun)): recent queue write activity: 5 file(s) modified recently.")
    }

    func testUnreadableSkipReasonNeverClaimsAGateItCannotSee() {
        let status = maintenance(weekly: 75, completion: .completed(daysAgo(2)))
        XCTAssertEqual(status.health, .skipped)
        XCTAssertEqual(status.attentionReason, "Weekly skipped at \(show(lastRun)): a maintenance safety gate deferred it.")
    }

    /// MaintenanceAbort defaults to 75 for real failures too: 21 nightly aborts in the live
    /// maintenance-nightly.out.log were `failed to resume N launchd services`.
    func testAnExit75ThatIsARealFailureStaysAttention() {
        for reason in [
            "failed to resume 1 launchd service: watch: Command '[launchctl]' returned non-zero exit status 5.",
            "outside quiet window: now=x start_hour=4 duration_minutes=120; failed to resume 2 launchd services: watch: boom",
            "burn drain failed; queue files preserved",
            "post-maintenance search latency too high: 912.0ms > 500.0ms",
        ] {
            let status = maintenance(nightly: 75, completion: .completed(daysAgo(2)), reasons: [.maintenanceNightly: reason])
            XCTAssertEqual(status.health, .unhealthy, reason)
            XCTAssertEqual(status.attentionReason, "Nightly last run exited 75 at \(show(lastRun)): \(reason)", reason)
        }
    }

    // MARK: exit 76 stays attention, in words

    func testWeeklyExit76IsAttentionWithHumanText() {
        let status = maintenance(
            weekly: 76, completion: .completed(daysAgo(2)),
            reasons: [.maintenanceWeekly: "fresh weekly backup timed out; VACUUM skipped"]
        )
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(status.attentionReason,
                       "Weekly VACUUM skipped at \(show(lastRun)): the fresh backup failed (fresh weekly backup timed out).")
        XCTAssertFalse(status.attentionReason?.contains("exited 76") == true)

        let bare = maintenance(weekly: 76, completion: .completed(daysAgo(2)))
        XCTAssertEqual(bare.health, .unhealthy)
        XCTAssertEqual(bare.attentionReason, "Weekly VACUUM skipped at \(show(lastRun)): the fresh backup failed.")
    }

    // MARK: skips cannot hide a weekly pass that never completes

    func testSkippedWeeklyWithStaleCompletionIsAttention() {
        let completed = daysAgo(8.5)
        let status = maintenance(weekly: 75, completion: .completed(completed), reasons: [.maintenanceWeekly: quietWindow])
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(
            status.attentionReason,
            "Weekly maintenance hasn't completed since \(show(completed)); last run skipped: outside the 04:00–06:00 quiet window."
        )
        XCTAssertEqual(maintenance(weekly: 75, completion: .completed(daysAgo(7.9))).health, .skipped)
    }

    func testSkippedWeeklyWithNoCompletedPassOnRecordIsAttention() {
        let status = maintenance(weekly: 75, completion: .noneRecorded, reasons: [.maintenanceWeekly: quietWindow])
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(
            status.attentionReason,
            "Weekly maintenance has no completed pass on record; last run skipped: outside the 04:00–06:00 quiet window."
        )
    }

    func testSkippedWeeklyWithUnreadableCompletionIsUnknownNotHealthy() {
        let status = maintenance(weekly: 75, completion: .unread, reasons: [.maintenanceWeekly: quietWindow])
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(
            status.attentionReason,
            "Weekly skipped at \(show(lastRun)): outside the 04:00–06:00 quiet window; its last completed pass could not be read."
        )
    }

    // MARK: everything else is unchanged

    func testHealthyMaintenanceIsHealthy() {
        let status = maintenance(nightly: 0, weekly: 0, completion: .completed(daysAgo(20)))
        XCTAssertEqual(status.health, .healthy)
        XCTAssertNil(status.attentionReason)
    }

    func testOtherMaintenanceExitCodesStayAttention() {
        let status = maintenance(weekly: 77, completion: .completed(daysAgo(2)))
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(status.attentionReason, "Weekly last run exited 77 at \(show(lastRun)).")
    }

    func testANonMaintenanceJobExiting75IsStillAttention() {
        let status = BrainLayerLaunchdJobGroup.ingest.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: [
                .watch: .init(loadState: .running, runs: 1, lastExitCode: 0, lastRunAt: lastRun, nextRunAt: nil, isContinuous: true),
                .index: observation(exit: 75),
            ],
            formatDate: show,
            maintenance: .init(weeklyCompletion: .completed(daysAgo(1)), abortReasons: [.index: quietWindow]),
            now: now
        )
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(status.attentionReason, "Index last run exited 75 at \(show(lastRun)).")
    }

    func testAnAttentionJobOutranksASkipInTheSameGroup() {
        let status = maintenance(nightly: 1, weekly: 75, completion: .completed(daysAgo(2)))
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(status.attentionReason, "Nightly last run exited 1 at \(show(lastRun)).")
    }

    // MARK: the evidence, from the jobs' own files

    private func plist(_ body: String) -> Data {
        Data("""
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0"><dict><key>Label</key><string>x</string>\(body)</dict></plist>
        """.utf8)
    }

    private func sources(_ files: [String: Data]) -> BrainBarBackupSources {
        BrainBarBackupSources(
            paths: .init(
                launchAgents: URL(fileURLWithPath: "/LA"),
                databaseLog: URL(fileURLWithPath: "/d/backup-daily.log"),
                archiveLog: URL(fileURLWithPath: "/d/jsonl-backup.log"),
                maintenanceLog: URL(fileURLWithPath: "/d/maintenance.log"),
                snapshotDirectory: URL(fileURLWithPath: "/d/backups"),
                archiveDirectory: URL(fileURLWithPath: "/d/jsonl-backups")
            ),
            readFile: { files[$0.path] },
            listDirectory: { _ in [] },
            isRegularFile: { _ in true }
        )
    }

    func testEvidenceReadsTheCompletedPassAndEachJobsLastAbortFromItsStdout() {
        let files: [String: Data] = [
            "/LA/com.brainlayer.maintenance-weekly.plist": plist("<key>StandardOutPath</key><string>/logs/weekly.out.log</string>"),
            "/LA/com.brainlayer.maintenance-nightly.plist": plist("<key>StandardOutPath</key><string>/logs/nightly.out.log</string>"),
            "/d/maintenance.log": Data("""
            {"ts": "2026-08-30T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 15926001664}
            {"ts": "2026-09-27T01:40:00+00:00", "mode": "full", "dry_run": false, "backup_status": "unavailable", "vacuum_after_bytes": null}
            """.utf8),
            "/logs/weekly.out.log": Data("""
            {"reason": "recent queue write activity: 5 file(s) modified recently", "status": "aborted"}
            not json
            {"reason": "\(quietWindow)", "status": "aborted"}

            """.utf8),
            // The nightly's last run succeeded: an older abort is not its reason.
            "/logs/nightly.out.log": Data("""
            {"reason": "burn drain failed; queue files preserved", "status": "aborted"}
            {"mode": "light", "status": "ok"}
            """.utf8),
        ]
        let evidence = sources(files).maintenanceEvidence()
        XCTAssertEqual(evidence.weeklyCompletion, .completed(ISO8601DateFormatter().date(from: "2026-08-30T02:23:03Z")!))
        XCTAssertEqual(evidence.abortReasons, [.maintenanceWeekly: quietWindow])
    }

    func testEvidenceDistinguishesNoCompletedPassFromAnUnreadableLog() {
        let neverCompleted = sources([
            "/d/maintenance.log": Data(#"{"ts": "2026-09-27T01:40:00+00:00", "mode": "light", "dry_run": false}"#.utf8),
        ]).maintenanceEvidence()
        XCTAssertEqual(neverCompleted.weeklyCompletion, .noneRecorded)
        XCTAssertEqual(neverCompleted.abortReasons, [:])
        XCTAssertEqual(sources([:]).maintenanceEvidence(), .unread)
    }

    // MARK: the Settings view model publishes the evidence beside the schedules

    @MainActor
    func testRefreshPublishesMaintenanceEvidenceIntoTheMaintenanceCard() async throws {
        let files: [String: Data] = [
            "/LA/com.brainlayer.maintenance-weekly.plist": plist("<key>StandardOutPath</key><string>/logs/weekly.out.log</string>"),
            "/d/maintenance.log": Data(#"{"ts": "2026-09-27T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 1}"#.utf8),
            "/logs/weekly.out.log": Data(#"{"reason": "\#(quietWindow)", "status": "aborted"}"#.utf8),
        ]
        let viewModel = BrainBarSettingsViewModel(
            store: BrainLayerConfigStore(configURL: FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).env")),
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdObservations: [.maintenanceNightly: observation(exit: 0), .maintenanceWeekly: observation(exit: 75)],
            refreshStatusOnLoad: false,
            now: { [now] in now },
            backupSources: sources(files)
        )
        XCTAssertEqual(viewModel.groupStatus(.maintenance).health, .unknown, "evidence not read yet")
        viewModel.refreshBackupSchedules()
        for _ in 0..<200 where viewModel.maintenanceEvidence == .unread { try await Task.sleep(nanoseconds: 10_000_000) }
        let status = viewModel.groupStatus(.maintenance)
        XCTAssertEqual(status.health, .skipped)
        XCTAssertEqual(status.attentionReason?.hasSuffix(": outside the 04:00–06:00 quiet window."), true)
    }
}
