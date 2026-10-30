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
        XCTAssertTrue(text.contains("Nightly maintenance (step maintenance-nightly) found cut off"), text)
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
        XCTAssertTrue(text.contains("Database backup (step backup-daily) found cut off at 2026-10-29 05:00 GMT+2"), text)
        XCTAssertFalse(text.contains("maintenance"), text)
        XCTAssertFalse(text.contains("runner stopped"), text)
    }

    func testMixedAbandonmentShowsEarlierFailureAndBothEventTimes() throws {
        var interruption = abandoned(date)
        interruption["incomplete_steps"] = ["backup-daily"]
        var terminal = summary(date)
        terminal["ts"] = timestamp(date, "04:00:00")
        let text = try issueText([runner(date), finished("jsonl-backup"), interruption, terminal])
        XCTAssertTrue(text.contains("Database backup (step backup-daily) found cut off at 2026-10-29 05:00 GMT+2"), text)
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
        // One cause per terminal: the abandonment already explains the night (Bugbot r4411982204).
        XCTAssertFalse(text.contains("unknown failure"), text)
    }

    func testLaterIndependentUnknownFailureIsNotBlamedOnAnEarlierAbandonment() throws {
        func at(_ row: [String: Any], _ time: String) -> [String: Any] {
            row.merging(["ts": timestamp(date, time)]) { _, new in new }
        }
        let synthetic = at(summary(date), "03:00:00").merging(["abandoned": true]) { _, new in new }
        let retry = [at(runner(date), "03:30:00"), finished("maintenance-nightly", "ok", 0),
                     at(summary(date), "04:00:00")]
        // Review sequence: the abandonment named its step, which a later attempt recovered;
        // that attempt then reported an unexplained failure of its own.
        let named = try issueText([runner(date), abandoned(date), synthetic] + retry)
        XCTAssertTrue(named.contains("Scheduled maintenance reported an unknown failure at 2026-10-29 06:00 GMT+2"), named)
        XCTAssertFalse(named.contains("abandoned"), named)
        // An abandonment that named nothing and a later independent failure are two causes,
        // each shown once at its own time.
        var interruption = abandoned(date)
        interruption.removeValue(forKey: "incomplete_steps")
        let both = try issueText([runner(date), interruption, synthetic] + retry)
        XCTAssertTrue(both.contains("Nightly maintenance was abandoned at 2026-10-29 05:00 GMT+2"), both)
        XCTAssertTrue(both.contains("Scheduled maintenance reported an unknown failure at 2026-10-29 06:00 GMT+2"), both)
        XCTAssertEqual(both.components(separatedBy: "abandoned").count, 2, both)
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

    func testCoincidingCountsCannotHideRunnerFailureBehindDeferredOrInheritedSteps() throws {
        let attempts = [
            [runner(date), finished("jsonl-backup"), finished("maintenance-nightly", "deferred", 75), summary(date, failed: 2)],
            [runner(date), finished("jsonl-backup"), finished("maintenance-nightly"), summary(date, failed: 2),
             runner(date), finished("jsonl-backup"), summary(date, failed: 2)],
        ]
        for attempt in attempts {
            let value = try history(attempt + [runner(date, kind: "recovery"),
                finished("jsonl-backup", "ok", 0), finished("maintenance-nightly", "ok", 0),
                summary(date, failed: 0, kind: "recovery")])
            XCTAssertNil(value.error)
            XCTAssertEqual(value.maintenanceNight?.failed, true)
            XCTAssertEqual(value.maintenanceNight?.failedSteps, [])
            let text = snapshot(value, dot: false).issues.first { $0.id == "maintenance.night" }?.text ?? ""
            XCTAssertTrue(text.contains("unknown failure"), text)
        }
    }

    func testExplicitRunnerErrorSurvivesFullStepRecoveryEvenWithoutExcessCount() throws {
        // Producer rows have no date/kind; active-step and cleanup errors both inherit the header.
        for details: [String: Any] in [["step": "jsonl-backup"], ["phase": "cleanup"], ["step": NSNull()]] {
            let error = details.merging(["event": "runner-error", "error": "RuntimeError",
                "ts": timestamp(date, "02:00:00")]) { _, new in new }
            let value = try history([runner(date), finished("jsonl-backup"), error, summary(date),
                runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
                finished("maintenance-nightly", "ok", 0), summary(date, failed: 0, kind: "recovery")])
            XCTAssertEqual(value.maintenanceNight?.failed, true)
            XCTAssertEqual(value.maintenanceNight?.failedSteps, [])
            let text = snapshot(value, dot: false).issues.first { $0.id == "maintenance.night" }?.text ?? ""
            XCTAssertTrue(text.contains("Maintenance runner failed at 2026-10-29 04:00 GMT+2"), text)
        }
    }

    func testPreLaunchRunnerErrorBelongsToItsOwnSummaryNotThePreviousNight() throws {
        let previous = "2026-10-28"
        let undated: [String: Any] = ["event": "runner-error", "error": "OSError", "ts": timestamp(date, "01:15:30")]
        // First row ever: the history stays readable and the failed night is reported.
        let first = try history([undated, summary(date)])
        XCTAssertNil(first.error)
        XCTAssertEqual(first.maintenanceNight?.scheduledDate, date)
        XCTAssertEqual(first.maintenanceNight?.failed, true)
        let text = snapshot(first, dot: false).issues.first { $0.id == "maintenance.night" }?.text ?? ""
        XCTAssertTrue(text.contains("Maintenance runner failed at 2026-10-29 03:15 GMT+2"), text)
        // After a clean previous night, the error is not blamed on that night.
        let clean = [runner(previous), finished("jsonl-backup", "ok", 0), finished("maintenance-nightly", "ok", 0),
                     summary(previous, failed: 0)]
        for error in [undated, undated.merging(["kind": "nightly", "scheduled_date": date]) { _, new in new }] {
            let value = try history(clean + [error, summary(date)])
            XCTAssertNil(value.error)
            XCTAssertEqual(value.maintenanceNight?.scheduledDate, date)
            XCTAssertEqual(value.maintenanceNight?.failed, true)
            let previousOnly = try history(clean + [error])
            XCTAssertEqual(previousOnly.maintenanceNight?.scheduledDate, previous)
            XCTAssertEqual(previousOnly.maintenanceNight?.failed, false)
        }
    }

    func testRunnerErrorInUnterminatedAttemptCannotBeClearedByRecovery() throws {
        let value = try history([runner(date), finished("jsonl-backup"), summary(date),
            runner(date),
            ["event": "runner-error", "phase": "cleanup", "error": "RuntimeError", "ts": timestamp(date, "02:00:00")],
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
            summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, true)
        XCTAssertEqual(value.maintenanceNight?.failedSteps, [])
    }

    func testSignalStoppedNightIsExplainedByCutOffStepsAndClearsAfterRecovery() throws {
        // A graceful stop writes failed=max(1, 0) with no failed rows; the stopped/skipped steps explain it.
        for cutOff in [finished("maintenance-nightly", "stopped", -15), finished("maintenance-nightly", "skipped", NSNull())] {
            let night = [runner(date), finished("jsonl-backup", "ok", 0), cutOff, summary(date, failed: 1)]
            let stopped = try history(night)
            XCTAssertEqual(stopped.maintenanceNight?.failed, true)
            XCTAssertEqual(stopped.maintenanceNight?.failedSteps, ["maintenance-nightly"])
            let text = snapshot(stopped, dot: false).issues.first { $0.id == "maintenance.night" }?.text ?? ""
            XCTAssertFalse(text.contains("unknown failure"), text)
            let recovered = try history(night + [runner(date, kind: "recovery"),
                finished("maintenance-nightly", "ok", 0), summary(date, failed: 0, kind: "recovery")])
            XCTAssertEqual(recovered.maintenanceNight?.failed, false)
        }
        // A runner failure on top of the stop still exceeds the floor and cannot be recovered away.
        let value = try history([runner(date), finished("jsonl-backup", "ok", 0),
            finished("maintenance-nightly", "stopped", -15), summary(date, failed: 2),
            runner(date, kind: "recovery"), finished("maintenance-nightly", "ok", 0),
            summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, true)
    }

    func testDeferredStepWithoutExcessSummaryFailureCanRecover() throws {
        let value = try history([runner(date), finished("jsonl-backup"),
            finished("maintenance-nightly", "deferred", 75), summary(date),
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
            finished("maintenance-nightly", "ok", 0), summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, false)
    }

    func testSharedCutOffDetectionTimeAppearsOnceForAllNineSteps() throws {
        let names = ["index", "backup-snapshot", "maintenance-weekly", "jsonl-backup", "p0-counter",
                     "wal-checkpoint", "decay", "repair-fts", "backup-daily"]
        var interruption = abandoned(date)
        interruption["incomplete_steps"] = names
        let text = try issueText([runner(date), interruption, summary(date)])
        XCTAssertTrue(text.contains("found cut off at 2026-10-29 05:00 GMT+2"), text)
        XCTAssertEqual(text.components(separatedBy: "2026-10-29 05:00 GMT+2").count - 1, 1, text)
        for name in names { XCTAssertTrue(text.contains("step \(name)"), text) }
    }

    func testFailedRecoveryOrManualRunCannotRefailCompleteOrRecoveredNight() throws {
        for kind in ["recovery", "manual"] {
            let complete = [runner(date), finished("jsonl-backup", "ok", 0), summary(date, failed: 0)]
            let recovered = [runner(date), finished("jsonl-backup"), summary(date),
                runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0), summary(date, failed: 0, kind: "recovery")]
            for initial in [complete, recovered] {
                let value = try history(initial + [runner(date, kind: kind), finished("jsonl-backup"),
                    ["event": "runner-error", "phase": "cleanup", "error": "RuntimeError", "ts": timestamp(date, "08:00:00")],
                    summary(date, failed: 2, kind: kind)])
                XCTAssertEqual(value.maintenanceNight?.failed, false, kind)
                XCTAssertEqual(value.maintenanceNight?.failedSteps, [], kind)
                XCTAssertEqual(value.maintenanceNight?.consecutiveFailures, 0, kind)
                XCTAssertFalse(snapshot(value, dot: false).issues.contains { $0.id == "maintenance.night" })
            }
        }
    }

    func testFailedRecoveryCanOnlyResolveExistingProblemsAndCannotAddNewOnes() throws {
        let value = try history([runner(date), finished("jsonl-backup"), finished("maintenance-nightly"), summary(date, failed: 2),
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0), finished("p0-counter"),
            summary(date, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, true)
        XCTAssertEqual(value.maintenanceNight?.failedSteps, ["maintenance-nightly"])
        let cleared = try history([runner(date), finished("jsonl-backup"), summary(date),
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0), finished("p0-counter"),
            summary(date, kind: "recovery")])
        XCTAssertEqual(cleared.maintenanceNight?.failed, false)
    }

    func testUnterminatedFailedRecoveryCannotLeakProblemsThroughNextHeader() throws {
        let value = try history([runner(date), finished("jsonl-backup"), summary(date),
            runner(date, kind: "recovery"), finished("p0-counter"),
            runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
            summary(date, failed: 0, kind: "recovery")])
        XCTAssertEqual(value.maintenanceNight?.failed, false)
        XCTAssertEqual(value.maintenanceNight?.failedSteps, [])
    }

    func testDeferralsAreNeutralAndCannotBlockActualFailureRecovery() throws {
        for (status, code) in [("deferred", 75), ("ok", 75)] {
            let initial = [runner(date), finished("jsonl-backup"), finished("maintenance-nightly", status, code), summary(date)]
            let failed = try history(initial)
            XCTAssertEqual(failed.maintenanceNight?.failedSteps, ["jsonl-backup"])
            XCTAssertNil(failed.maintenanceNight?.stepProblems["maintenance-nightly"])
            let recovered = try history(initial + [runner(date, kind: "recovery"), finished("jsonl-backup", "ok", 0),
                summary(date, failed: 0, kind: "recovery")])
            XCTAssertEqual(recovered.maintenanceNight?.failed, false)
            XCTAssertEqual(recovered.maintenanceNight?.failedSteps, [])
        }
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
