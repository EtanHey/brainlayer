import XCTest
@testable import BrainBar

/// Jobs → Maintenance: only a POSITIVELY VERIFIED deliberate gate deferral (exit 75) renders neutral
/// "Skipped". Missing, unreadable, incomplete, uncorrelated or unrecognised evidence is Status
/// unknown or attention, never neutral. Exit 76 stays attention with human text, and skips never
/// hide a weekly pass that has stopped completing.
final class BrainBarMaintenanceSkipTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
    private let lastRun = ISO8601DateFormatter().date(from: "2026-09-29T21:07:23Z")!
    private let show: (Date) -> String = { ISO8601DateFormatter().string(from: $0) }
    private let quietWindow = "outside quiet window: now=2026-09-30T00:07:23.566976+03:00 start_hour=4 duration_minutes=120"

    private func daysAgo(_ days: Double) -> Date { now.addingTimeInterval(-days * 86_400) }

    private func observation(exit: Int32?, runs: Int? = 3) -> BrainLayerLaunchdJobObservation {
        .init(loadState: .loaded, runs: runs, lastExitCode: exit, lastRunAt: lastRun, nextRunAt: nil, isContinuous: false)
    }

    /// The abort record this run wrote, a second after it started.
    private func aborted(_ reason: String) -> BrainLayerMaintenanceEvidence.RunRecord {
        .aborted(reason: reason, writtenAt: lastRun.addingTimeInterval(1))
    }

    private func maintenance(
        nightly: Int32? = 0,
        weekly: Int32? = 0,
        completion: BrainLayerMaintenanceEvidence.WeeklyCompletion,
        records: [BrainLayerLaunchdJob: BrainLayerMaintenanceEvidence.RunRecord] = [:]
    ) -> BrainLayerLaunchdGroupStatus {
        BrainLayerLaunchdJobGroup.maintenance.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: [.maintenanceNightly: observation(exit: nightly), .maintenanceWeekly: observation(exit: weekly)],
            formatDate: show,
            maintenance: .init(weeklyCompletion: completion, runRecords: records),
            now: now
        )
    }

    // MARK: exit 75 is Skipped only for a verified, allowlisted gate deferral

    func testWeeklyQuietWindowDeferralIsSkippedWithItsOwnReason() {
        let status = maintenance(weekly: 75, completion: .completed(daysAgo(2)), records: [.maintenanceWeekly: aborted(quietWindow)])
        XCTAssertEqual(status.health, .skipped)
        XCTAssertEqual(status.health.title, "Skipped")
        XCTAssertEqual(status.attentionReason, "Weekly skipped at \(show(lastRun)): outside the 04:00–06:00 quiet window.")
    }

    /// One case per deliberate deferral `maintenance.py` raises with exit 75 before any service is
    /// quiesced (`_check_quiet_window`, `_check_idle` ×2, `_check_lsof_clean`), matched case-insensitively.
    func testEveryAllowlistedGateDeferralIsSkipped() {
        let deferrals = [
            (quietWindow, "outside the 04:00–06:00 quiet window"),
            ("OUTSIDE QUIET WINDOW: now=2026-09-30T00:07:23+03:00 start_hour=4 duration_minutes=120", "outside the 04:00–06:00 quiet window"),
            ("recent queue write activity: 5 file(s) modified recently", "recent queue write activity: 5 file(s) modified recently"),
            ("queue depth growing: before=0 after=1", "queue depth growing: before=0 after=1"),
            ("unexpected writer holds BrainLayer DB: pid=812 command='python3 -m brainlayer.enrich' fd=7u",
             "unexpected writer holds BrainLayer DB: pid=812 command='python3 -m brainlayer.enrich' fd=7u"),
        ]
        for (reason, shown) in deferrals {
            let status = maintenance(nightly: 75, completion: .completed(daysAgo(2)), records: [.maintenanceNightly: aborted(reason)])
            XCTAssertEqual(status.health, .skipped, reason)
            XCTAssertEqual(status.attentionReason, "Nightly skipped at \(show(lastRun)): \(shown).", reason)
        }
    }

    /// `MaintenanceAbort` defaults to 75 for real failures too (21 live nightly aborts were
    /// `failed to resume N launchd services`). Anything not on the allowlist is attention.
    func testAnyExit75ReasonOffTheAllowlistIsAttention() {
        for reason in [
            "failed to resume 1 launchd service: watch: Command '[launchctl]' returned non-zero exit status 5.",
            "FAILED TO RESUME 2 launchd services: watch: boom",
            "failed to quiesce launchd service com.brainlayer.watch; it remains loaded",
            "Burn drain failed; queue files preserved",
            "Post-maintenance search latency too high: 912.0ms > 500.0ms",
            "lsof failed: permission denied",
            "cannot determine whether launchd service com.brainlayer.watch is loaded",
            "maintenance aborted: refusing to run STALE code. working tree HEAD=a but merged origin/main=b.",
            "outside quiet window: now=x start_hour=4 duration_minutes=120; failed to resume 1 launchd service: watch: boom",
            "outside quiet window",
            "a reason no one has written yet",
        ] {
            let status = maintenance(nightly: 75, completion: .completed(daysAgo(2)), records: [.maintenanceNightly: aborted(reason)])
            XCTAssertEqual(status.health, .unhealthy, reason)
            XCTAssertEqual(status.attentionReason, "Nightly last run exited 75 at \(show(lastRun)): \(reason)", reason)
        }
    }

    // MARK: B1/B2: no verified record for this run → Status unknown, never Skipped

    func testExit75WithNoRecordIsUnknownNotSkipped() {
        let status = maintenance(weekly: 75, completion: .completed(daysAgo(2)))
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(status.attentionReason,
                       "Weekly exited 75 at \(show(lastRun)); reason unavailable (its stdout log was not read).")
    }

    func testExit75WithAnUnreadableRecordIsUnknown() {
        let status = maintenance(
            nightly: 75, completion: .completed(daysAgo(2)),
            records: [.maintenanceNightly: .unavailable("the newest record in nightly.out.log is incomplete")]
        )
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(status.attentionReason,
                       "Nightly exited 75 at \(show(lastRun)); reason unavailable (the newest record in nightly.out.log is incomplete).")
    }

    func testExit75WhoseNewestRecordIsNotAnAbortIsUnknown() {
        let status = maintenance(
            nightly: 75, completion: .completed(daysAgo(2)),
            records: [.maintenanceNightly: .notAnAbort(writtenAt: lastRun.addingTimeInterval(60))]
        )
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(status.attentionReason,
                       "Nightly exited 75 at \(show(lastRun)); reason unavailable (the newest stdout record is not an abort).")
    }

    /// Stale stdout: the newest abort was written before this run started, so it is another run's.
    func testAnAbortRecordThatPredatesThisRunIsUnknown() {
        let status = maintenance(
            weekly: 75, completion: .completed(daysAgo(2)),
            records: [.maintenanceWeekly: .aborted(reason: quietWindow, writtenAt: lastRun.addingTimeInterval(-7 * 86_400))]
        )
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(status.attentionReason,
                       "Weekly exited 75 at \(show(lastRun)); reason unavailable (the newest stdout record is not from this run).")
    }

    /// A record written after the lock wait could have ended belongs to a later run, not this one.
    func testAnAbortRecordFarAfterThisRunStartedIsUnknown() {
        let status = maintenance(
            weekly: 75, completion: .completed(daysAgo(2)),
            records: [.maintenanceWeekly: .aborted(reason: quietWindow, writtenAt: lastRun.addingTimeInterval(5 * 3_600))]
        )
        XCTAssertEqual(status.health, .unknown)
        // Within the 4 h maintenance-lock wait, the record still belongs to this run.
        let waited = maintenance(
            weekly: 75, completion: .completed(daysAgo(2)),
            records: [.maintenanceWeekly: .aborted(reason: quietWindow, writtenAt: lastRun.addingTimeInterval(3 * 3_600))]
        )
        XCTAssertEqual(waited.health, .skipped)
    }

    func testExit75WithNoRunStartOnRecordIsUnknown() {
        let status = BrainLayerLaunchdJobGroup.maintenance.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: [
                .maintenanceNightly: observation(exit: 0),
                .maintenanceWeekly: .init(loadState: .loaded, runs: 3, lastExitCode: 75, lastRunAt: nil, nextRunAt: nil, isContinuous: false),
            ],
            formatDate: show,
            maintenance: .init(weeklyCompletion: .completed(daysAgo(2)), runRecords: [.maintenanceWeekly: aborted(quietWindow)]),
            now: now
        )
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(status.attentionReason, "Weekly exited 75; reason unavailable (no run start is on record).")
    }

    // MARK: exit 76 stays attention, in words

    func testWeeklyExit76IsAttentionWithHumanText() {
        let status = maintenance(
            weekly: 76, completion: .completed(daysAgo(2)),
            records: [.maintenanceWeekly: aborted("fresh weekly backup timed out; VACUUM skipped")]
        )
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(status.attentionReason,
                       "Weekly VACUUM skipped at \(show(lastRun)): the fresh backup failed (fresh weekly backup timed out).")

        let bare = maintenance(weekly: 76, completion: .completed(daysAgo(2)))
        XCTAssertEqual(bare.health, .unhealthy)
        XCTAssertEqual(bare.attentionReason, "Weekly VACUUM skipped at \(show(lastRun)): the fresh backup failed.")

        // Another run's detail is never quoted as this run's.
        let stale = maintenance(
            weekly: 76, completion: .completed(daysAgo(2)),
            records: [.maintenanceWeekly: .aborted(reason: "weekly backup was not verified; VACUUM skipped",
                                                   writtenAt: lastRun.addingTimeInterval(-86_400))]
        )
        XCTAssertEqual(stale.attentionReason, "Weekly VACUUM skipped at \(show(lastRun)): the fresh backup failed.")
    }

    // MARK: skips cannot hide a weekly pass that never completes

    func testSkippedWeeklyWithStaleCompletionIsAttention() {
        let completed = daysAgo(8.5)
        let status = maintenance(weekly: 75, completion: .completed(completed), records: [.maintenanceWeekly: aborted(quietWindow)])
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(
            status.attentionReason,
            "Weekly maintenance hasn't completed since \(show(completed)); last run skipped: outside the 04:00–06:00 quiet window."
        )
        XCTAssertEqual(
            maintenance(weekly: 75, completion: .completed(daysAgo(7.9)), records: [.maintenanceWeekly: aborted(quietWindow)]).health,
            .skipped
        )
    }

    /// Stale history outranks a missing reason: the pass that stopped completing is the finding.
    func testUnexplainedWeekly75WithStaleCompletionIsAttention() {
        let completed = daysAgo(31)
        let status = maintenance(weekly: 75, completion: .completed(completed))
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(
            status.attentionReason,
            "Weekly maintenance hasn't completed since \(show(completed)); last run exited 75, reason unavailable (its stdout log was not read)."
        )
    }

    func testSkippedWeeklyWithNoCompletedPassOnRecordIsAttention() {
        let status = maintenance(weekly: 75, completion: .noneRecorded, records: [.maintenanceWeekly: aborted(quietWindow)])
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(
            status.attentionReason,
            "Weekly maintenance has no completed pass on record; last run skipped: outside the 04:00–06:00 quiet window."
        )
    }

    func testSkippedWeeklyWithUnreadableCompletionIsUnknownNotHealthy() {
        let status = maintenance(weekly: 75, completion: .unread, records: [.maintenanceWeekly: aborted(quietWindow)])
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
            maintenance: .init(weeklyCompletion: .completed(daysAgo(1)), runRecords: [.index: aborted(quietWindow)]),
            now: now
        )
        XCTAssertEqual(status.health, .unhealthy)
        XCTAssertEqual(status.attentionReason, "Index last run exited 75 at \(show(lastRun)).")
    }

    func testAnAttentionJobOutranksASkipInTheSameGroup() {
        let status = maintenance(nightly: 1, weekly: 75, completion: .completed(daysAgo(2)), records: [.maintenanceWeekly: aborted(quietWindow)])
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

    private let weeklyPlist = "<key>StandardOutPath</key><string>/logs/weekly.out.log</string>"
    private let nightlyPlist = "<key>StandardOutPath</key><string>/logs/nightly.out.log</string>"

    private func sources(
        _ files: [String: Data],
        modified: [String: Date] = [:],
        changing: Set<String> = []
    ) -> BrainBarBackupSources {
        let stamp = lastRun
        return BrainBarBackupSources(
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
            isRegularFile: { _ in true },
            readStdoutTail: { url in
                if changing.contains(url.path) { return .changedWhileReading }
                guard let data = files[url.path] else { return .unreadable }
                return .read(data, modifiedAt: modified[url.path] ?? stamp)
            }
        )
    }

    func testEvidenceReadsTheCompletedPassAndOnlyTheNewestStdoutRecord() {
        let written = lastRun.addingTimeInterval(1)
        let files: [String: Data] = [
            "/LA/com.brainlayer.maintenance-weekly.plist": plist(weeklyPlist),
            "/LA/com.brainlayer.maintenance-nightly.plist": plist(nightlyPlist),
            "/d/maintenance.log": Data("""
            {"ts": "2026-08-30T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 15926001664}
            {"ts": "2026-09-27T01:40:00+00:00", "mode": "full", "dry_run": false, "backup_status": "unavailable", "vacuum_after_bytes": null}
            """.utf8),
            "/logs/weekly.out.log": Data("""
            {"reason": "recent queue write activity: 5 file(s) modified recently", "status": "aborted"}
            {"reason": "\(quietWindow)", "status": "aborted"}

            """.utf8),
            "/logs/nightly.out.log": Data("""
            {"reason": "burn drain failed; queue files preserved", "status": "aborted"}
            {"mode": "light", "status": "ok"}
            """.utf8),
        ]
        let evidence = sources(files, modified: ["/logs/weekly.out.log": written, "/logs/nightly.out.log": written]).maintenanceEvidence()
        XCTAssertEqual(evidence.weeklyCompletion, .completed(ISO8601DateFormatter().date(from: "2026-08-30T02:23:03Z")!))
        XCTAssertEqual(evidence.runRecords[.maintenanceWeekly], .aborted(reason: quietWindow, writtenAt: written))
        XCTAssertEqual(evidence.runRecords[.maintenanceNightly], .notAnAbort(writtenAt: written))
    }

    /// B2: an older benign abort is never reused when the newest record is partial or malformed.
    func testAPartialNewestRecordNeverFallsBackToAnOlderGateAbort() {
        let files: [String: Data] = [
            "/LA/com.brainlayer.maintenance-weekly.plist": plist(weeklyPlist),
            "/d/maintenance.log": Data(#"{"ts": "2026-09-27T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 1}"#.utf8),
            "/logs/weekly.out.log": Data("""
            {"reason": "\(quietWindow)", "status": "aborted"}
            {"reason": "failed to resume 1 launchd service: watch: bo
            """.utf8),
        ]
        let evidence = sources(files, modified: ["/logs/weekly.out.log": lastRun.addingTimeInterval(1)]).maintenanceEvidence()
        XCTAssertEqual(evidence.runRecords[.maintenanceWeekly], .unavailable("the newest record in weekly.out.log is incomplete"))
        let status = BrainLayerLaunchdJobGroup.maintenance.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: [.maintenanceNightly: observation(exit: 0), .maintenanceWeekly: observation(exit: 75)],
            formatDate: show, maintenance: evidence, now: now
        )
        XCTAssertEqual(status.health, .unknown)
    }

    func testAnAbortWithoutAReasonIsIncomplete() {
        let files: [String: Data] = [
            "/LA/com.brainlayer.maintenance-weekly.plist": plist(weeklyPlist),
            "/logs/weekly.out.log": Data(#"{"status": "aborted"}"#.utf8),
        ]
        let evidence = sources(files, modified: ["/logs/weekly.out.log": lastRun]).maintenanceEvidence()
        XCTAssertEqual(evidence.runRecords[.maintenanceWeekly], .unavailable("the newest record in weekly.out.log is incomplete"))
    }

    func testMissingStdoutEvidenceIsNamed() {
        let noStdoutPath = sources(["/LA/com.brainlayer.maintenance-weekly.plist": plist("")]).maintenanceEvidence()
        XCTAssertEqual(noStdoutPath.runRecords[.maintenanceWeekly], .unavailable("its LaunchAgent names no StandardOutPath"))
        XCTAssertEqual(noStdoutPath.runRecords[.maintenanceNightly], .unavailable("its LaunchAgent could not be read"))

        let unreadable = sources(["/LA/com.brainlayer.maintenance-weekly.plist": plist(weeklyPlist)]).maintenanceEvidence()
        XCTAssertEqual(unreadable.runRecords[.maintenanceWeekly], .unavailable("weekly.out.log could not be read"))

        let changed = sources([
            "/LA/com.brainlayer.maintenance-weekly.plist": plist(weeklyPlist),
            "/logs/weekly.out.log": Data(#"{"reason": "\#(quietWindow)", "status": "aborted"}"#.utf8),
        ], changing: ["/logs/weekly.out.log"]).maintenanceEvidence()
        XCTAssertEqual(changed.runRecords[.maintenanceWeekly], .unavailable("weekly.out.log changed while reading"))
    }

    // MARK: #1039 R2 B1: the bytes and their mtime come from one stable snapshot

    private func tempLog(_ text: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("maint-race-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("weekly.out.log")
        try Data(text.utf8).write(to: url)
        return url
    }

    private static func append(_ text: String, to url: URL) {
        let handle = try! FileHandle(forWritingTo: url)
        handle.seekToEndOfFile()
        handle.write(Data(text.utf8))
        try! handle.close()
    }

    private func mtime(_ url: URL) -> timespec {
        var info = stat()
        XCTAssertEqual(stat(url.path, &info), 0)
        return info.st_mtimespec
    }

    func testAStableLogReadsItsBytesWithThePreReadMtime() throws {
        let text = #"{"reason": "\#(quietWindow)", "status": "aborted"}"# + "\n"
        let url = try tempLog(text)
        let stamp = mtime(url)
        let expected = Date(timeIntervalSince1970: TimeInterval(stamp.tv_sec) + TimeInterval(stamp.tv_nsec) / 1_000_000_000)
        XCTAssertEqual(BrainBarStableFileReader.readTail(url, maxBytes: 64 * 1024), .read(Data(text.utf8), modifiedAt: expected))
        // Only the tail is read, but the snapshot still carries the whole file's mtime.
        XCTAssertEqual(BrainBarStableFileReader.readTail(url, maxBytes: 9), .read(Data(text.utf8.suffix(9)), modifiedAt: expected))
        XCTAssertEqual(
            BrainBarStableFileReader.readTail(url.deletingLastPathComponent().appendingPathComponent("missing.log"), maxBytes: 64),
            .unreadable
        )
    }

    /// The reviewer's interleaving: yesterday's benign abort is read, then this run appends its
    /// failed-quiesce abort before the mtime is sampled. That snapshot is unstable, never Skipped.
    func testAnAppendBetweenTheReadAndTheSecondStatIsUnknownNeverSkipped() throws {
        let url = try tempLog(#"{"reason": "\#(quietWindow)", "status": "aborted"}"# + "\n")
        let failure = #"{"reason": "failed to quiesce launchd service com.brainlayer.watch; it remains loaded", "status": "aborted"}"# + "\n"
        let read = BrainBarStableFileReader.readTail(url, maxBytes: 64 * 1024, afterRead: { Self.append(failure, to: url) })
        XCTAssertEqual(read, .changedWhileReading)

        let raced = try tempLog(#"{"reason": "\#(quietWindow)", "status": "aborted"}"# + "\n")
        let files = ["/LA/com.brainlayer.maintenance-weekly.plist": plist("<key>StandardOutPath</key><string>\(raced.path)</string>")]
        var evidenceSources = sources(files)
        evidenceSources.readStdoutTail = { url in
            BrainBarStableFileReader.readTail(url, maxBytes: 64 * 1024, afterRead: { Self.append(failure, to: url) })
        }
        let evidence = evidenceSources.maintenanceEvidence()
        XCTAssertEqual(evidence.runRecords[.maintenanceWeekly], .unavailable("weekly.out.log changed while reading"))
        let status = BrainLayerLaunchdJobGroup.maintenance.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: [.maintenanceNightly: observation(exit: 0), .maintenanceWeekly: observation(exit: 75)],
            formatDate: show,
            maintenance: .init(weeklyCompletion: .completed(daysAgo(2)), runRecords: evidence.runRecords),
            now: now
        )
        XCTAssertEqual(status.health, .unknown)
        XCTAssertEqual(status.attentionReason,
                       "Weekly exited 75 at \(show(lastRun)); reason unavailable (weekly.out.log changed while reading).")
    }

    /// A size change with the mtime restored to its pre-read value is still an unstable snapshot.
    func testASizeChangeWithTheSameMtimeIsUnstable() throws {
        let url = try tempLog(#"{"reason": "\#(quietWindow)", "status": "aborted"}"# + "\n")
        let before = mtime(url)
        let read = BrainBarStableFileReader.readTail(url, maxBytes: 64 * 1024, afterRead: {
            Self.append("{\"status\": \"aborted\", \"reason\": \"x\"}\n", to: url)
            var times = [before, before]
            XCTAssertEqual(utimensat(AT_FDCWD, url.path, &times, 0), 0)
        })
        XCTAssertEqual(mtime(url).tv_sec, before.tv_sec)
        XCTAssertEqual(mtime(url).tv_nsec, before.tv_nsec)
        XCTAssertEqual(read, .changedWhileReading)
    }

    /// N1: an unreadable or wholly malformed history is not "no completed pass".
    func testEvidenceKeepsUnreadableHistoryApartFromHistoryWithNoCompletion() {
        let neverCompleted = sources([
            "/d/maintenance.log": Data(#"{"ts": "2026-09-27T01:40:00+00:00", "mode": "light", "dry_run": false}"#.utf8),
        ]).maintenanceEvidence()
        XCTAssertEqual(neverCompleted.weeklyCompletion, .noneRecorded)
        XCTAssertEqual(sources([:]).maintenanceEvidence().weeklyCompletion, .unread)
        XCTAssertEqual(sources(["/d/maintenance.log": Data([0x7B, 0xFF, 0xFE, 0x7D])]).maintenanceEvidence().weeklyCompletion, .unread)
        XCTAssertEqual(sources(["/d/maintenance.log": Data("not json\n{broken\n".utf8)]).maintenanceEvidence().weeklyCompletion, .unread)
        XCTAssertEqual(sources(["/d/maintenance.log": Data()]).maintenanceEvidence().weeklyCompletion, .unread)
    }

    // MARK: the Settings view model publishes the evidence beside the schedules

    @MainActor
    func testRefreshPublishesMaintenanceEvidenceIntoTheMaintenanceCard() async throws {
        let files: [String: Data] = [
            "/LA/com.brainlayer.maintenance-weekly.plist": plist(weeklyPlist),
            "/d/maintenance.log": Data(#"{"ts": "2026-09-27T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 1}"#.utf8),
            "/logs/weekly.out.log": Data(#"{"reason": "\#(quietWindow)", "status": "aborted"}"#.utf8),
        ]
        let viewModel = BrainBarSettingsViewModel(
            store: BrainLayerConfigStore(configURL: FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).env")),
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdObservations: [.maintenanceNightly: observation(exit: 0), .maintenanceWeekly: observation(exit: 75)],
            refreshStatusOnLoad: false,
            now: { [now] in now },
            backupSources: sources(files, modified: ["/logs/weekly.out.log": lastRun.addingTimeInterval(1)])
        )
        XCTAssertEqual(viewModel.groupStatus(.maintenance).health, .unknown, "evidence not read yet")
        viewModel.refreshBackupSchedules()
        for _ in 0..<200 where viewModel.maintenanceEvidence == .unread { try await Task.sleep(nanoseconds: 10_000_000) }
        let status = viewModel.groupStatus(.maintenance)
        XCTAssertEqual(status.health, .skipped)
        XCTAssertEqual(status.attentionReason?.hasSuffix(": outside the 04:00–06:00 quiet window."), true)
    }
}
