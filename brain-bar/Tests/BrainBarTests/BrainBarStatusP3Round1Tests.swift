import Darwin
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusP3Round1Tests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests

    private func history(_ rows: [[String: Any]]) throws -> BrainBarCalendarHistory {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var data = Data()
        for row in rows { data.append(try JSONSerialization.data(withJSONObject: row)); data.append(10) }
        try data.write(to: directory.appendingPathComponent("runs.jsonl"))
        let value = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(value.error)
        return value
    }

    private func summary(_ date: String, failed: Int, kind: String? = "nightly") -> [String: Any] {
        var row: [String: Any] = ["event": "summary", "scheduled_date": date,
                                 "failed": failed, "ts": T.hoursAgo(12).timeIntervalSince1970]
        if let kind { row["kind"] = kind }
        return row
    }

    private func abandoned(_ date: String) -> [String: Any] {
        ["event": "abandoned", "kind": "nightly", "scheduled_date": date, "step": "index",
         "incomplete_steps": ["maintenance-nightly"], "reason": "runner-killed",
         "ts": T.hoursAgo(12).timeIntervalSince1970]
    }

    private func finished(_ step: String, date: String) -> [String: Any] {
        ["event": "finished", "step": step, "scheduled_date": date, "status": "failed", "returncode": 76,
         "ts": T.hoursAgo(12).timeIntervalSince1970]
    }

    private func inputs(_ history: BrainBarCalendarHistory) -> BrainBarStatusInputs {
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { _ in .init(terminationStatus: 0, output: "runs = 0") },
            now: { T.now }, ownership: { .supervisor }, calendarHistory: { history },
            calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
        var inputs = T.healthy()
        let sampled = provider.sampleActivity()
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] { inputs.observations?[job] = sampled[job] }
        return inputs
    }

    func testAbandonedSuccessAbandonedIsOnlyOneFailedNight() throws {
        let value = try history([abandoned("2026-10-20"), summary("2026-10-21", failed: 0), abandoned("2026-10-22")])
        XCTAssertNil(value.runs["maintenance-weekly"], "incomplete_steps must not invent a weekly run")
        T.assertAgree(inputs(value), dot: false, "abandoned/ok/abandoned")
    }

    func testNightlyThenWeeklyFailuresCountAcrossPlansAndRecoveryCannotReset() throws {
        let rows = [finished("maintenance-nightly", date: "2026-10-24"), summary("2026-10-24", failed: 1),
                    finished("maintenance-weekly", date: "2026-10-25"), summary("2026-10-25", failed: 1)]
        T.assertAgree(inputs(try history(rows)), dot: true, "nightly then weekly")
        T.assertAgree(inputs(try history(rows + [summary("2026-10-25", failed: 0, kind: "recovery")])),
                      dot: true, "hourly recovery cannot reset nightly failures")
        T.assertAgree(inputs(try history(rows + [summary("2026-10-26", failed: 0)])), dot: false, "successful night resets")
    }

    func testSameDateCountsOnceAndLegacyRecordsCannotEarnDot() throws {
        T.assertAgree(inputs(try history([finished("maintenance-nightly", date: "2026-10-25"),
            summary("2026-10-25", failed: 1), abandoned("2026-10-25"), abandoned("2026-10-25")])),
            dot: false, "one failed night counted once")
        var legacy = abandoned("2026-10-24"); legacy.removeValue(forKey: "kind")
        T.assertAgree(inputs(try history([legacy, summary("2026-10-24", failed: 1, kind: nil),
            finished("maintenance-weekly", date: "2026-10-25"), finished("maintenance-weekly", date: "2026-10-26")])),
            dot: false, "legacy history is not a nightly outcome")
    }

    func testAbandonmentAndItsSummaryAreOneNightWithSharedRunnerIdentity() throws {
        var event = abandoned("2026-10-25")
        event["runner_started_at"] = 1792880000.0
        var terminal = summary("2026-10-25", failed: 1)
        terminal["runner_started_at"] = 1792880000.0
        terminal["incomplete_steps"] = ["maintenance-nightly"]
        let oneNight = try history([event, terminal])
        XCTAssertEqual(oneNight.maintenanceNight?.consecutiveFailures, 1)
        T.assertAgree(inputs(oneNight), dot: false, "abandonment and caused summary share one night")
        let secondNight = try history([event, terminal, summary("2026-10-26", failed: 1)])
        XCTAssertEqual(secondNight.maintenanceNight?.consecutiveFailures, 2)
        T.assertAgree(inputs(secondNight), dot: true, "a distinct failed scheduled night earns the dot")
    }

    func testLegacyOwnershipSkipsMissingSupervisorDaemon() {
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { args in
            if args.last?.hasSuffix("/com.brainlayer.daemon") == true {
                XCTFail("Legacy ownership must not probe the supervisor daemon")
                return .init(terminationStatus: 113, output: "Could not find service")
            }
            return .init(terminationStatus: 0, output: "pid = 42\nlast exit code = 0")
        }, now: { T.now }, ownership: { .legacy })
        var inputs = T.healthy()
        let watcher = provider.sampleActivity()[.watch]
        XCTAssertNil(watcher?.killCheckUnavailable)
        inputs.observations?[.watch] = watcher
        let (snapshot, surfaces) = T.assertAgree(inputs, dot: false, "legacy without supervisor daemon")
        XCTAssertTrue(snapshot.notices.isEmpty)
        XCTAssertEqual(surfaces.dashboard.headline, "All good")
    }

    func testExpiringDrivePreservesUnavailableAndFirstReadBackupStates() {
        var inputs = T.healthy()
        inputs.drive = .init(line: "Drive access expiring", detail: nil, tone: .expiring,
                            showsReconnect: true, reconnectEnabled: true, needsAttention: true)
        inputs.backups = .unavailable("Backup evidence unavailable")
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        let unavailable = snapshot.project(.init(badge: .attention, reason: "Backup evidence unavailable"))
        XCTAssertEqual(unavailable.badge, .unknown)
        XCTAssertEqual(unavailable.reason, "Backup evidence unavailable")
        inputs.backups = .notRead
        let firstRead = BrainBarStatusSnapshot.derive(inputs).project(.init(badge: .checking, reason: nil))
        XCTAssertEqual(firstRead.badge, .checking)
        XCTAssertNil(firstRead.reason)
    }

    func testKnownBackupErrorsTakePrecedenceOverUnknownStatus() {
        for error in ["upload_failed", "job_alert:Fixture failure"] {
            var inputs = T.healthy()
            inputs.jobAlerts = nil
            inputs.backups = .measured(T.backups(error: error))
            inputs.backupsNotice = "Receipt envelope freshness unknown"
            let (_, surfaces) = T.assertAgree(inputs, dot: false, error)
            XCTAssertEqual(surfaces.backups.badge, .notice)
            XCTAssertNotEqual(surfaces.backups.reason, inputs.backupsNotice)
        }
    }

    func testInformationalFleetAlertDoesNotChangeBackupsBadge() {
        var inputs = T.healthy()
        inputs.jobAlerts = .init(active: ["fleet-watchdog-hold": "Fleet watchdog held"])
        let (snapshot, surfaces) = T.assertAgree(inputs, dot: false, "fleet notice")
        XCTAssertFalse(snapshot.notices.isEmpty)
        XCTAssertEqual(surfaces.backups.badge, .healthy)
    }

    func testKillCheckFailuresAreVisibleAndNeverHealthy() {
        for result in [BrainLayerLaunchdCommandResult(terminationStatus: 113, output: "service unavailable"),
                       .init(terminationStatus: 0, output: "unparseable fixture")] {
            let provider = BrainLayerLaunchdStatusProvider(commandRunner: { _ in result }, now: { T.now },
                ownership: { .supervisor }, calendarHistory: { .empty },
                calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
            var inputs = T.healthy()
            inputs.observations?[.watch] = provider.sampleActivity()[.watch]
            let (snapshot, surfaces) = T.assertAgree(inputs, dot: false, "kill check unavailable")
            XCTAssertTrue(snapshot.notices.contains { $0.text.contains("Kill check unavailable") })
            XCTAssertNotEqual(surfaces.dashboard.headline, "All good")
        }
    }

    func testDayFormatRecoveryAndMissingDaemonPID() {
        for running in [true, false] {
            let provider = BrainLayerLaunchdStatusProvider(commandRunner: { args in
                if args.first == "/bin/ps" { return .init(terminationStatus: 0, output: "1-00:00:00") }
                return .init(terminationStatus: 0,
                    output: (running ? "pid = 42\n" : "") + "last terminating signal = Killed: 9")
            }, now: { T.now }, ownership: { .supervisor }, calendarHistory: { .empty },
                calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
            var inputs = T.healthy()
            inputs.observations?[.watch] = provider.sampleActivity()[.watch]
            T.assertAgree(inputs, dot: !running, "day-format uptime or absent process")
        }
    }

    func testUnprovableBackupSafetyEscalatesAfterSixtyMinutes() {
        for (seconds, dot) in [(3599.0, false), (3600.0, false), (3601.0, true)] {
            var inputs = T.healthy()
            inputs.backups = .unavailable("unmeasurable backup receipt source")
            inputs.observabilityUnreadableFor = seconds
            let (snapshot, _) = T.assertAgree(inputs, dot: dot, "backup safety unknown for \(seconds)s")
            if dot { XCTAssertTrue(snapshot.attention.contains { $0.text.contains("backup safety can't be proven") }) }
        }
    }

    func testRunningWatcherWithoutCaptureForFifteenMinutesGivesDot() {
        for (seconds, dot) in [(899.0, false), (900.0, false), (901.0, true)] {
            var inputs = T.healthy()
            let heartbeat = T.now.addingTimeInterval(-seconds)
            inputs.watcher = .degraded(issues: [], heartbeatAt: heartbeat)
            inputs.watcherHeartbeatAt = heartbeat
            T.assertAgree(inputs, dot: dot, "running watcher capture age \(seconds)s")
        }
    }
}
