import Darwin
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarNightRecoveryTests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests
    private let date = "2026-10-29"

    private func timestamp(_ date: String, _ time: String) -> Double {
        ISO8601DateFormatter().date(from: "\(date)T\(time)Z")!.timeIntervalSince1970
    }

    private func runner(_ date: String, kind: String = "nightly") -> [String: Any] {
        ["event": kind == "recovery" ? "transcript-recovery" : "runner", "kind": kind,
         "scheduled_date": date, "steps": ["maintenance-nightly", "jsonl-backup"],
         "ts": timestamp(date, kind == "recovery" ? "07:40:00" : "01:15:00")]
    }

    // The producer puts the date/kind on runner and summary, not on finished rows.
    private func finished(_ step: String, _ status: String = "failed", _ code: Any = 1) -> [String: Any] {
        ["event": "finished", "step": step, "status": status, "returncode": code, "ts": timestamp(date, "01:20:00")]
    }

    private func summary(_ date: String, failed: Int = 1, kind: String = "nightly") -> [String: Any] {
        ["event": "summary", "kind": kind, "scheduled_date": date, "failed": failed,
         "ts": timestamp(date, kind == "recovery" ? "07:44:00" : "03:00:00")]
    }

    private func abandoned(_ date: String) -> [String: Any] {
        ["event": "abandoned", "kind": "nightly", "scheduled_date": date,
         "incomplete_steps": ["maintenance-nightly"], "reason": "runner-vanished", "ts": timestamp(date, "03:00:00")]
    }

    private func history(_ rows: [[String: Any]], suffix: String = "") throws -> BrainBarCalendarHistory {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var data = Data()
        for row in rows { data.append(try JSONSerialization.data(withJSONObject: row)); data.append(10) }
        data.append(Data(suffix.utf8))
        try data.write(to: directory.appendingPathComponent("runs.jsonl"))
        return BrainBarCalendarHistory.read(directory: directory)
    }

    @discardableResult
    private func snapshot(_ history: BrainBarCalendarHistory, dot: Bool) -> BrainBarStatusSnapshot {
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { _ in .init(terminationStatus: 0, output: "runs = 0") },
            now: { T.now }, ownership: { .supervisor }, calendarHistory: { history },
            calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
        var inputs = T.healthy()
        let sampled = provider.sampleActivity()
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] { inputs.observations?[job] = sampled[job] }
        return T.assertAgree(inputs, dot: dot, "night recovery").0
    }

    func testOctober272829ThenTranscriptRecoveryResetsCurrentStreak() throws {
        let rows = [runner("2026-10-27"), abandoned("2026-10-27"), summary("2026-10-27"),
                    runner("2026-10-28"), abandoned("2026-10-28"), summary("2026-10-28"),
                    runner(date), finished("maintenance-nightly", "ok", 0), finished("jsonl-backup"), summary(date),
                    runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
                    summary(date, failed: 0, kind: "recovery")]
        let value = try history(rows)
        XCTAssertNil(value.error)
        XCTAssertEqual(value.maintenanceNight?.failed, false)
        XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 0)
        XCTAssertFalse(snapshot(value, dot: false).issues.contains { $0.id == "maintenance.night" })
    }

    func testPartialRecoveryKeepsFailureAndNamesOnlyUnresolvedStep() throws {
        let value = try history([runner(date), finished("jsonl-backup"), finished("maintenance-nightly"), summary(date, failed: 2),
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0), summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, true)
        XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 1)
        let text = try XCTUnwrap(snapshot(value, dot: false).issues.first { $0.id == "maintenance.night" }?.text)
        XCTAssertTrue(text.contains("Nightly maintenance failed"), text)
        XCTAssertTrue(text.contains("step maintenance-nightly"), text)
        XCTAssertFalse(text.contains("jsonl-backup"), text)
    }

    func testAbandonedNightCompletedByRecoveryClears() throws {
        let value = try history([runner(date), abandoned(date), summary(date),
            runner(date, kind: "recovery"), finished("maintenance-nightly", "ok", 0), summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, false)
        XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 0)
        snapshot(value, dot: false)
    }

    func testAbandonmentRequiresBothFailedAndIncompleteSteps() throws {
        let rows = [runner(date), finished("jsonl-backup"), abandoned(date), summary(date),
            runner(date, kind: "recovery"), finished("maintenance-nightly", "ok", 0), summary(date, failed: 0, kind: "recovery")]
        XCTAssertEqual(try history(rows).maintenanceNight?.failed, true)
        let value = try history(rows + [runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
            summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, false)
        XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 0)
    }

    func testOlderRecoveredNightBreaksStreakWithoutHidingLatestFailure() throws {
        let value = try history([runner("2026-10-28"), finished("jsonl-backup"), summary("2026-10-28"),
            runner(date), finished("jsonl-backup"), summary(date),
            runner("2026-10-28", kind: "recovery"), finished("jsonl-backup", "ok", 0),
            summary("2026-10-28", failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.scheduledDate, date)
        XCTAssertEqual(value.maintenanceNight?.failed, true)
        XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 1)
        snapshot(value, dot: false)
    }

    func testUnknownFailedStepCannotBeRecoveredByUnrelatedSuccess() throws {
        let value = try history([runner(date), summary(date),
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0), summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, true)
    }

    func testDeferredSkippedAndNonzeroOKCannotRecover() throws {
        for (status, code): (String, Any) in [("deferred", 75), ("skipped", NSNull()), ("ok", 75), ("ok", 1)] {
            let value = try history([runner(date), finished("jsonl-backup"), summary(date),
                runner(date, kind: "recovery"), finished("jsonl-backup", status, code), summary(date, failed: 0, kind: "recovery")])
            XCTAssertNil(value.error)
            XCTAssertEqual(value.maintenanceNight?.failed, true, status)
            XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 1, status)
        }
    }

    func testRecoveryNeedsSameDateAndLaterCompletedRun() throws {
        let failure = [runner(date), finished("jsonl-backup"), summary(date)]
        for rows in [
            failure + [runner("2026-10-30", kind: "recovery"), finished("jsonl-backup", "ok", 0), summary("2026-10-30", failed: 0, kind: "recovery")],
            failure + [runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0)],
            [runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0), summary(date, failed: 0, kind: "recovery")] + failure,
            failure + [summary(date, failed: 0, kind: "recovery")],
        ] {
            XCTAssertEqual(try history(rows).maintenanceNight?.failed, true)
        }
    }

    func testTranscriptFailureNamesBackupAndPreservesStreakDot() throws {
        let value = try history([runner("2026-10-28"), finished("jsonl-backup"), summary("2026-10-28"),
            runner(date), finished("jsonl-backup"), summary(date)])
        let text = try XCTUnwrap(snapshot(value, dot: true).attention.first { $0.id == "maintenance.night" }?.text)
        XCTAssertTrue(text.contains("Transcript backup failed"), text)
        XCTAssertTrue(text.contains("step jsonl-backup"), text)
        XCTAssertTrue(text.contains("nightly runs have had problems 2 nights in a row"), text)
    }

    func testAbandonedMessageNamesIncompleteStepAndKeepsNightCount() throws {
        let value = try history([runner("2026-10-28"), abandoned("2026-10-28"), summary("2026-10-28"),
            runner(date), abandoned(date), summary(date)])
        let text = try XCTUnwrap(snapshot(value, dot: true).attention.first { $0.id == "maintenance.night" }?.text)
        XCTAssertTrue(text.contains("Nightly maintenance was cut off (step maintenance-nightly)"), text)
        XCTAssertTrue(text.contains("nightly runs have had problems 2 nights in a row"), text)
    }

    private func issueText(_ rows: [[String: Any]], dot: Bool = false) throws -> String {
        try XCTUnwrap(snapshot(history(rows), dot: dot).issues.first { $0.id == "maintenance.night" }?.text)
    }

    func testAbandonedDatabaseBackupNamesOnlyUnfinishedBackup() throws {
        var interruption = abandoned(date)
        interruption["incomplete_steps"] = ["backup-daily"]
        interruption["reason"] = "step-orphaned"
        let text = try issueText([runner(date), finished("maintenance-weekly", "ok", 0), interruption, summary(date)])
        XCTAssertTrue(text.contains("Database backup was cut off (step backup-daily) at 2026-10-29 05:00 GMT+2"), text)
        XCTAssertFalse(text.contains("maintenance"), text)
        XCTAssertFalse(text.contains("runner stopped"), text)
    }

    func testMixedAbandonmentShowsEarlierFailureAndBothEventTimes() throws {
        var interruption = abandoned(date)
        interruption["incomplete_steps"] = ["backup-daily"]
        var terminal = summary(date)
        terminal["ts"] = timestamp(date, "04:00:00")
        let text = try issueText([runner(date), finished("jsonl-backup"), interruption, terminal])
        XCTAssertTrue(text.contains("Database backup was cut off (step backup-daily) at 2026-10-29 05:00 GMT+2"), text)
        XCTAssertTrue(text.contains("Transcript backup failed (step jsonl-backup) at 2026-10-29 03:20 GMT+2"), text)
        XCTAssertFalse(text.contains("06:00 GMT+2"), text)
    }

    func testFailureShowsStepFinishTimeRatherThanSummaryTime() throws {
        let text = try issueText([runner(date), finished("jsonl-backup"), summary(date)])
        XCTAssertTrue(text.contains("Transcript backup failed (step jsonl-backup) at 2026-10-29 03:20 GMT+2"), text)
        XCTAssertFalse(text.contains("05:00 GMT+2"), text)
    }

    func testMultipleFailuresKeepTheirOwnFinishTimes() throws {
        var later = finished("p0-counter")
        later["ts"] = timestamp(date, "02:10:00")
        let text = try issueText([runner(date), finished("jsonl-backup"), later, summary(date, failed: 2)])
        XCTAssertTrue(text.contains("Transcript backup (step jsonl-backup) at 2026-10-29 03:20 GMT+2"), text)
        XCTAssertTrue(text.contains("Memory counts (step p0-counter) at 2026-10-29 04:10 GMT+2"), text)
    }

    func testOtherDateSuccessDoesNotReplaceFailedStepTime() throws {
        let text = try issueText([runner(date), finished("jsonl-backup"), summary(date),
            runner("2026-10-28", kind: "recovery"), finished("jsonl-backup", "ok", 0),
            summary("2026-10-28", failed: 0, kind: "recovery")])
        XCTAssertTrue(text.contains("Transcript backup failed (step jsonl-backup) at 2026-10-29 03:20 GMT+2"), text)
    }

    func testDeferredOrSkippedRecoveryKeepsOriginalFailureTimeAndWording() throws {
        for (status, code): (String, Any) in [("deferred", 75), ("skipped", NSNull())] {
            var retry = finished("jsonl-backup", status, code)
            retry["ts"] = timestamp(date, "07:42:00")
            let text = try issueText([runner(date), finished("jsonl-backup"), summary(date),
                runner(date, kind: "recovery"), retry, summary(date, failed: 0, kind: "recovery")])
            XCTAssertTrue(text.contains("Transcript backup failed (step jsonl-backup) at 2026-10-29 03:20 GMT+2"), text)
        }
    }

    func testAbandonmentWithoutStepInformationKeepsGenericFallback() throws {
        var interruption = abandoned(date)
        interruption.removeValue(forKey: "incomplete_steps")
        let text = try issueText([runner(date), interruption, summary(date)])
        XCTAssertTrue(text.contains("Nightly maintenance was abandoned at 2026-10-29 05:00 GMT+2"), text)
    }

    func testHeaderCannotForgetFailureBeforeAbandonmentWhenOnlyIncompleteStepsRecover() throws {
        // Review R1's exact sequence: unterminated nightly failure, transcript recovery,
        // abandonment, then a manual run completing only the two unfinished steps.
        var interruption = abandoned(date)
        interruption["incomplete_steps"] = ["jsonl-backup", "backup-daily"]
        let rows = [runner(date), finished("p0-counter"),
            runner(date, kind: "recovery"), finished("jsonl-backup"), summary(date, kind: "recovery"),
            interruption, summary(date), runner(date, kind: "manual"),
            finished("jsonl-backup", "ok", 0), finished("backup-daily", "ok", 0),
            summary(date, failed: 0, kind: "manual")]
        let value = try history(rows)
        XCTAssertEqual(value.maintenanceNight?.failed, true)
        XCTAssertEqual(value.maintenanceNight?.failedSteps, ["p0-counter"])
        XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 1)
        let text = try XCTUnwrap(snapshot(value, dot: false).issues.first { $0.id == "maintenance.night" }?.text)
        XCTAssertTrue(text.contains("Memory counts failed (step p0-counter) at 2026-10-29 03:20 GMT+2"), text)
        XCTAssertFalse(text.contains("was cut off"), text)
        XCTAssertFalse(text.contains("jsonl-backup"), text)
        XCTAssertFalse(text.contains("backup-daily"), text)
    }

    func testMalformedRowsRemainUnreadable() throws {
        let rows = [runner(date), finished("jsonl-backup"), summary(date)]
        for suffix in ["{broken", #"{"event":"finished","step":"jsonl-backup","status":"ok","ts":1003}"#] {
            let value = try history(rows, suffix: suffix)
            XCTAssertEqual(value.error, "Maintenance calendar history is unreadable")
            XCTAssertNil(value.maintenanceNight)
        }
        var malformedRecovery = summary(date, failed: 0, kind: "recovery")
        malformedRecovery.removeValue(forKey: "failed")
        XCTAssertEqual(try history(rows + [malformedRecovery]).error, "Maintenance calendar history is unreadable")
    }
}
