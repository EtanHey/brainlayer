import Darwin
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusP3Tests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests

    func testSingleFailureIsExplicitWithAbsoluteFailureAndRetryTimes() {
        var inputs = T.healthy()
        inputs.observations?[.maintenanceWeekly] = T.failed(.maintenanceWeekly, code: 76)
        let (snapshot, surfaces) = T.assertAgree(inputs, dot: false, "one failure")
        XCTAssertNotEqual(surfaces.dashboard.headline, "No action needed")
        XCTAssertNotEqual(surfaces.jobs[.maintenance]?.health.title, "No action needed")
        let text = snapshot.notices.first { $0.jobs.contains(.maintenanceWeekly) }?.text ?? ""
        XCTAssertTrue(text.contains("failed"), text)
        XCTAssertTrue(text.contains("2026-10-26 04:10"), text)
        XCTAssertTrue(text.contains("2026-10-27 03:15"), text)
        XCTAssertFalse(text.contains("tonight"), text)
    }

    func testOtherNoticesDoNotHideTheSingleMaintenanceFailureOnDashboard() {
        var inputs = T.healthy()
        inputs.watcher = .unknown(reason: "Watcher health unreadable")
        inputs.observations?[.maintenanceWeekly] = T.failed(.maintenanceWeekly, code: 76)
        let (_, surfaces) = T.assertAgree(inputs, dot: false, "multiple notices")
        XCTAssertTrue(surfaces.dashboard.reason?.contains("maintenance failed at") == true,
                      surfaces.dashboard.reason ?? "missing reason")
        inputs.backups = .measured(T.backups(db: T.hoursAgo(37)))
        let (_, withDot) = T.assertAgree(inputs, dot: true, "backup risk and single maintenance failure")
        XCTAssertTrue(withDot.dashboard.reason?.contains("maintenance failed at") == true)
    }

    func testTwoMaintenanceFailuresGiveDotAndOldCompletionAloneDoesNot() {
        var inputs = T.healthy()
        inputs.observations?[.maintenanceWeekly] = T.failed(.maintenanceWeekly, code: 76, consecutive: 2)
        T.assertAgree(inputs, dot: true, "two failures")
        inputs = T.healthy()
        inputs.maintenance = .init(weeklyCompletion: .completed(T.hoursAgo(24 * 15)), runRecords: [:])
        T.assertAgree(inputs, dot: false, "old completion is not a failed run")
    }

    func testOnlyTheThreeAllowedConditionsGiveDots() {
        var inputs = T.healthy()
        inputs.drive = .init(line: "Drive access lost", detail: nil, tone: .attention,
                            showsReconnect: true, reconnectEnabled: true, needsAttention: true)
        inputs.badge = .init(badgeOn: true, reason: "search failed", activeCodes: ["brain_search_canary_failed"])
        inputs.observations?[.index] = T.failed(.index, code: 1, consecutive: 2)
        let (snapshot, _) = T.assertAgree(inputs, dot: false, "other faults remain visible")
        XCTAssertFalse(snapshot.notices.isEmpty)
        inputs.backups = .measured(T.backups(db: T.hoursAgo(37), threshold: 48))
        T.assertAgree(inputs, dot: true, "36 h rule overrides old producer threshold")
    }

    func testAbandonedRunsCountOncePerScheduledRunAndSuccessResets() throws {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("runs.jsonl")
        let abandoned = """
        {"event":"abandoned","kind":"nightly","incomplete_steps":["maintenance-weekly"],"scheduled_date":"2026-10-25","step":"maintenance-weekly","pid":42,"runner_started_at":1792880000,"detected_at":1792880100,"reason":"runner-killed","ts":1792880100}
        {"event":"abandoned","kind":"nightly","incomplete_steps":["maintenance-weekly"],"scheduled_date":"2026-10-26","step":null,"pid":null,"runner_started_at":1792966400,"detected_at":1792966500,"reason":"runner-vanished","ts":1792966500}

        """
        try Data((abandoned + abandoned).utf8).write(to: file)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        _ = try XCTUnwrap(BrainBarJobOwnership.readOwnedFile(file, privateMode: false, limit: 4096))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(history.error)
        XCTAssertEqual(history.runs["maintenance-weekly"]?.consecutiveFailures, 2)
        var inputs = T.healthy()
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { _ in .init(terminationStatus: 0, output: "") },
            ownership: { .supervisor }, calendarHistory: { history },
            calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
        let sampled = provider.sampleActivity()
        inputs.observations?[.maintenanceWeekly] = sampled[.maintenanceWeekly]
        inputs.observations?[.maintenanceNightly] = sampled[.maintenanceNightly]
        let (snapshot, _) = T.assertAgree(inputs, dot: true, "two abandoned runners")
        XCTAssertEqual(snapshot.attention.filter { $0.component == .maintenance }.count, 1)
        XCTAssertTrue(snapshot.attention.contains { $0.text.contains("Weekly maintenance was cut off (step maintenance-weekly)") })
        try Data((abandoned + "{\"event\":\"finished\",\"step\":\"maintenance-weekly\",\"ts\":1793052900,\"status\":\"ok\",\"returncode\":0}\n").utf8).write(to: file)
        XCTAssertEqual(BrainBarCalendarHistory.read(directory: directory).runs["maintenance-weekly"]?.consecutiveFailures, 0)
    }

    func testCompletedWeeklyCoverIsNotAbandonedWhenLaterUploadDies() throws {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("""
        {"event":"runner","scheduled_date":"2026-10-26","ts":1792966400}
        {"event":"spawn","step":"maintenance-weekly","ts":1792966401,"pid":42}
        {"event":"finished","step":"maintenance-weekly","ts":1792966450,"status":"ok","returncode":0,"replaces":"maintenance-nightly"}
        {"event":"abandoned","kind":"nightly","incomplete_steps":["maintenance-weekly"],"scheduled_date":"2026-10-26","step":"backup-daily","runner_started_at":1792966400,"ts":1792966500,"reason":"runner-vanished"}

        """.utf8).write(to: directory.appendingPathComponent("runs.jsonl"))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(history.error)
        XCTAssertEqual(history.runs["maintenance-weekly"]?.consecutiveFailures, 0)
        XCTAssertNil(history.runs["maintenance-nightly"])
    }

    func testLegacyWatcherKillUsesLaunchdEvidence() {
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { args in
            .init(terminationStatus: 0, output: args.last?.hasSuffix("/com.brainlayer.watch") == true
                ? "pid = 42\nlast terminating signal = Killed: 9" : "last exit code = 0")
        }, ownership: { .legacy })
        var inputs = T.healthy()
        inputs.observations?[.watch] = provider.sampleActivity()[.watch]
        T.assertAgree(inputs, dot: true, "legacy watcher SIGKILL")
    }

    func testDaemonKillAndWatcherUncleanExitButNotCleanBrewRestart() {
        for (signal, code, uptime, dot) in [("Killed: 9", 0, 60, true), ("", -9, 60, true),
                                          ("", 1, 60, true), ("Terminated: 15", 0, 60, false), ("", -15, 60, false),
                                          ("Killed: 9", 0, 21_599, true), ("Killed: 9", 0, 21_600, false), ("Killed: 9", -9, 21_600, true), ("", -9, 21_600, false)] {
            let start = T.now.timeIntervalSince1970 - Double(signal == "Killed: 9" && code == -9 ? 60 : uptime)
            let provider = BrainLayerLaunchdStatusProvider(commandRunner: { args in
                if args.contains("component") {
                    return .init(terminationStatus: 0, output: """
                    {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":42,"starts":2,"failures":0,"last_exit_code":\(code),"last_run_at":\(start)}}
                    """)
                }
                if args.first == "/bin/ps" { return .init(terminationStatus: 0, output: String(format: "%02d:%02d:%02d", uptime / 3600, uptime / 60 % 60, uptime % 60)) }
                return .init(terminationStatus: 0, output: "pid = 42\nlast terminating signal = \(signal)\n")
            }, now: { T.now }, ownership: { .supervisor }, calendarHistory: { .empty },
               calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
            var inputs = T.healthy()
            inputs.observations?[.watch] = provider.sampleActivity()[.watch]
            T.assertAgree(inputs, dot: dot, "signal \(signal), exit \(code), healthy \(uptime)s")
        }
    }
}
