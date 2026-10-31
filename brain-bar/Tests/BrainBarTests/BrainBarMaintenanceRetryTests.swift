import XCTest
@testable import BrainBar

private final class RetryCLI: BrainLayerCLIRunning, @unchecked Sendable {
    let output: String
    var calls: [[String]] = []
    init(_ output: String) { self.output = output }
    func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
        calls.append(arguments)
        return .init(terminationStatus: 0, stdout: output)
    }
}

private final class ScriptedRetryCLI: BrainLayerCLIRunning, @unchecked Sendable {
    var outputs: [String]
    init(_ outputs: [String]) { self.outputs = outputs }
    func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
        .init(terminationStatus: 0, stdout: outputs.count > 1 ? outputs.removeFirst() : outputs[0])
    }
}

private final class BlockingRetryCLI: BrainLayerCLIRunning, @unchecked Sendable {
    let started: XCTestExpectation
    let release = DispatchSemaphore(value: 0)
    init(started: XCTestExpectation) { self.started = started }
    func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
        if arguments.last == "--status" {
            return .init(terminationStatus: 0, stdout: #"{"status":"nothing-to-retry","reason":"Already recovered"}"#)
        }
        started.fulfill()
        guard release.wait(timeout: .now() + 3) == .success else { return nil }
        return .init(terminationStatus: 0, stdout: #"{"status":"accepted","reason":"Retry started under calendar custody"}"#)
    }
}

private final class BlockingStatusRetryCLI: BrainLayerCLIRunning, @unchecked Sendable {
    let statusStarted: XCTestExpectation
    let release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var probes = 0
    private var actions = 0
    var actionCount: Int { lock.withLock { actions } }
    init(_ started: XCTestExpectation) { statusStarted = started }
    func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
        if arguments.last == "--status" {
            let first = lock.withLock { probes += 1; return probes == 1 }
            if first {
                statusStarted.fulfill()
                guard release.wait(timeout: .now() + 3) == .success else { return nil }
            }
            return .init(terminationStatus: 0, stdout: BrainBarMaintenanceRetryTests.ready)
        }
        lock.withLock { actions += 1 }
        return .init(terminationStatus: 0, stdout: #"{"status":"accepted","reason":"Finished","steps":["jsonl-backup"]}"#)
    }
}

@MainActor
final class BrainBarMaintenanceRetryTests: XCTestCase {
    nonisolated static let ready = #"{"status":"ready","reason":"Only daytime-safe failures retry","retry_pause_start":"2026-10-30T01:07:30+00:00","retry_pause_end":"2026-10-30T06:00:00+00:00","scheduled_date":"2026-10-28","steps":["jsonl-backup"],"failures":[{"step":"jsonl-backup","reason":"it stopped with an error"},{"step":"maintenance-weekly","reason":"Window closed"}],"deferred":[{"step":"maintenance-weekly","reason":"It pauses database writes, so it only runs between 04:00 and 06:00","next_run":"will run tonight at 04:00"}]}"#
    private let noon = ISO8601DateFormatter().date(from: "2026-10-29T12:00:00Z")!

    // Fixed local clock: UTC 02:30 is 04:30 in the user's calendar.
    private func presentationModel(status: String = "ready", steps: [String] = [],
                                   deferred: [String] = [], start: String? = "2026-10-30T02:30:00Z",
                                   now: String = "2026-10-29T12:00:00Z", running: Bool = false) throws -> BrainBarMaintenanceRetryModel {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 2 * 3600)!
        let clock = ISO8601DateFormatter().date(from: now)!
        var json: [String: Any] = ["status": status, "reason": "The maintenance supervisor is unavailable.",
            "steps": steps, "deferred": deferred.map { ["step": $0, "reason": "Internal writer-stop restriction"] },
            "retry_pause_end": "2026-10-30T04:30:00Z"]
        json["retry_pause_start"] = start
        let model = BrainBarMaintenanceRetryModel(runner: nil, now: { clock }, calendar: calendar)
        model.setForPreview(json: String(decoding: try JSONSerialization.data(withJSONObject: json), as: UTF8.self),
                            isRunning: running)
        return model
    }

    func testEveryReceiptStatusHasAnActionOnlyWhenEligible() throws {
        for (status, visible, sentence) in [
            ("ready", true, "Retry will run: Transcript backup. Nothing else pauses."),
            ("accepted", false, "The retry finished."),
            ("refused", false, "The maintenance supervisor is unavailable."),
            ("nothing-to-retry", false, "Everything from last night has finished. Nothing to retry."),
            ("failed", false, "The maintenance supervisor is unavailable.")
        ] {
            let model = try presentationModel(status: status, steps: ["jsonl-backup"])
            XCTAssertEqual(model.showsRetryButton, visible, status)
            XCTAssertEqual(model.explanation, sentence, status)
            XCTAssertEqual(model.details, "", status)
        }
        let empty = try presentationModel()
        XCTAssertFalse(empty.showsRetryButton)
        XCTAssertEqual(empty.explanation, "Nothing to retry now.")
        XCTAssertEqual(empty.details, "")
        let checking = BrainBarMaintenanceRetryModel(runner: nil)
        XCTAssertFalse(checking.showsRetryButton)
        XCTAssertEqual(checking.explanation, "Checking retry availability…")
    }

    func testDeferredOnlyUsesLabelsAndLocalScheduledTimeForEveryStatus() throws {
        for status in ["ready", "accepted", "refused", "nothing-to-retry", "failed"] {
            let model = try presentationModel(status: status, deferred: ["backup-daily"])
            XCTAssertFalse(model.showsRetryButton, status)
            XCTAssertEqual(model.details, "Database backup didn’t finish last night. It runs again automatically tonight during scheduled maintenance, after 04:30. Nothing to do now.", status)
            XCTAssertEqual(model.explanation, "", status)
        }
        let multiple = try presentationModel(deferred: ["backup-snapshot", "maintenance-weekly"])
        XCTAssertFalse(multiple.showsRetryButton)
        XCTAssertEqual(multiple.details, "Database snapshot didn’t finish last night. It runs again automatically tonight during scheduled maintenance, after 04:30. Nothing to do now.\nWeekly maintenance didn’t finish last night. It runs again automatically tonight during scheduled maintenance, after 04:30. Nothing to do now.")
    }

    func testDeferredOnlyWithoutAUsableStartOmitsTheTime() throws {
        for start in [nil, "not-a-date"] as [String?] {
            let model = try presentationModel(deferred: ["backup-daily"], start: start)
            XCTAssertFalse(model.showsRetryButton)
            XCTAssertEqual(model.details, "Database backup didn’t finish last night. It runs again automatically tonight during scheduled maintenance. Nothing to do now.")
            XCTAssertEqual(model.explanation, "")
        }
    }

    func testStaleDeferredTimeDoesNotPromiseAPastNextRun() throws {
        let model = try presentationModel(deferred: ["backup-daily"], now: "2026-10-30T05:00:00Z")
        XCTAssertFalse(model.showsRetryButton)
        XCTAssertEqual(model.details, "Database backup didn’t finish last night. It runs again automatically tonight during scheduled maintenance. Nothing to do now.")
    }

    func testMixedWorkKeepsActionAndOneTimedDeferredLine() throws {
        let model = try presentationModel(steps: ["jsonl-backup"], deferred: ["backup-daily"])
        XCTAssertTrue(model.showsRetryButton)
        XCTAssertEqual(model.explanation, "Retry will run: Transcript backup. Nothing else pauses.")
        XCTAssertEqual(model.details, "Database backup didn’t finish last night. It runs again automatically tonight during scheduled maintenance, after 04:30.")
    }

    func testPauseWindowHidesActionAndNamesLocalEndTime() throws {
        for status in ["ready", "accepted", "refused", "nothing-to-retry", "failed"] {
            let model = try presentationModel(status: status, steps: ["jsonl-backup"], deferred: ["backup-daily"],
                                              now: "2026-10-30T03:00:00Z")
            XCTAssertFalse(model.showsRetryButton, status)
            XCTAssertEqual(model.explanation, "Retries are paused for scheduled maintenance (until 06:30).", status)
            XCTAssertEqual(model.details, "Database backup didn’t finish last night.", status)
        }
        let stale = try presentationModel(steps: ["jsonl-backup"], now: "2026-10-30T05:00:00Z")
        XCTAssertFalse(stale.showsRetryButton)
        XCTAssertEqual(stale.explanation, "Checking retry availability…")
    }

    func testRunningRetryShowsProgressWithoutADisabledButton() throws {
        let model = try presentationModel(steps: ["jsonl-backup"], running: true)
        // A disabled button during the run is a dead control; the spinner and text say what happens.
        XCTAssertFalse(model.showsRetryButton)
        XCTAssertTrue(model.showsProgress)
        XCTAssertFalse(model.canRetry)
        XCTAssertEqual(model.explanation, "Retrying failed steps… This retry does not pause services.")
    }

    func testReadyRowListsFailuresAndWindowOnlyReason() async {
        let cli = RetryCLI(Self.ready)
        var calendar = Calendar(identifier: .iso8601)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { self.noon }, calendar: calendar)
        await model.refresh()
        XCTAssertTrue(model.canRetry)
        XCTAssertTrue(model.details.contains("Transcript backup didn’t finish last night. You can retry it now."))
        XCTAssertFalse(model.details.contains("jsonl-backup"))
        XCTAssertFalse(model.details.contains("exit 1"))
        XCTAssertTrue(model.explanation.contains("Retry will run: Transcript backup"))
        XCTAssertTrue(model.details.contains("Weekly maintenance didn’t finish last night. It runs again automatically tonight during scheduled maintenance, after 01:07."), model.details)
        XCTAssertEqual(model.details.components(separatedBy: "Weekly maintenance").count, 2, "one line per deferred step")
        XCTAssertEqual(cli.calls, [["maintenance-calendar", "--recover-failed", "--status"]])
    }

    func testDisabledStatesAndNoRunnerNeverLaunch() async {
        for status in ["refused", "nothing-to-retry"] {
            let cli = RetryCLI("{\"status\":\"\(status)\",\"reason\":\"fixture explanation\",\"steps\":[]}")
            let model = BrainBarMaintenanceRetryModel(runner: cli)
            await model.refresh()
            XCTAssertFalse(model.canRetry)
            XCTAssertEqual(model.explanation, status == "nothing-to-retry" ? "Everything from last night has finished. Nothing to retry." : "fixture explanation")
            await model.retry()
            XCTAssertEqual(cli.calls.count, 1)
        }
        let model = BrainBarMaintenanceRetryModel(runner: nil)
        await model.refresh()
        XCTAssertFalse(model.canRetry)
    }

    func testQuietWindowDisablesEvenPreviouslyReadyState() async {
        var calendar = Calendar(identifier: .iso8601)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let four = ISO8601DateFormatter().date(from: "2026-10-29T04:30:00Z")!
        let cli = RetryCLI(Self.ready)
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { four }, calendar: calendar)
        await model.refresh()
        model.setForPreview(json: Self.ready.replacingOccurrences(of: "2026-10-30", with: "2026-10-29"))
        XCTAssertFalse(model.canRetry)
        XCTAssertEqual(model.explanation, "Retries are paused for scheduled maintenance (until 06:00).")
        await model.retry()
        XCTAssertEqual(cli.calls.count, 1)
    }

    func testConfirmedRetryThatLostEligibilityExplainsWhy() async {
        // Bugbot r4420969886: the quiet window started while the confirmation was open.
        var clock = noon
        let cli = RetryCLI(Self.ready)
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { clock })
        model.setForPreview(json: Self.ready)
        XCTAssertTrue(model.canRetry)
        clock = ISO8601DateFormatter().date(from: "2026-10-30T04:30:00Z")!
        await model.retry()
        XCTAssertTrue(model.outcome?.hasPrefix("Refused: ") == true, model.outcome ?? "silent")
        XCTAssertTrue(cli.calls.isEmpty, "nothing may launch after eligibility is lost")
    }

    func testOutcomeClearsOnceALaterNightIsShown() async {
        // Bugbot r4412111109: a "Done" from one night must not sit next to the next night's failure.
        let cli = ScriptedRetryCLI([
            #"{"status":"accepted","reason":"ok","scheduled_date":"2026-10-28"}"#,
            #"{"status":"nothing-to-retry","reason":"Everything that can run during the day has already finished.","scheduled_date":"2026-10-28"}"#,
            Self.ready.replacingOccurrences(of: "2026-10-28", with: "2026-10-29"),
        ])
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { self.noon })
        model.setForPreview(json: Self.ready)
        await model.retry()  // also refreshes: still the same night
        XCTAssertEqual(model.outcome, "Done: Transcript backup retried successfully.")
        await model.refresh()  // the next night failed
        XCTAssertNil(model.outcome)
        XCTAssertTrue(model.explanation.contains("Transcript backup"), model.explanation)
    }

    func testActionOutcomesUseSupportedCLIAndNeverClaimMalformedOutputStarted() async {
        for (json, expected) in [
            (#"{"status":"accepted","reason":"Retry started under calendar custody"}"#, "Done: Transcript backup retried successfully."),
            (#"{"status":"refused","reason":"Calendar is running"}"#, "Refused: Calendar is running"),
            (#"{"status":"nothing-to-retry","reason":"Already recovered"}"#, "Nothing to retry: Already recovered"),
            (#"{"status":"failed","reason":"Backup failed"}"#, "Failed: Backup failed"),
            ("broken", "Failed")
        ] {
            let cli = RetryCLI(json)
            let model = BrainBarMaintenanceRetryModel(runner: cli, now: { self.noon })
            model.setForPreview(json: Self.ready)
            await model.retry()
            XCTAssertTrue(model.outcome?.contains(expected) == true, model.outcome ?? "missing outcome")
            XCTAssertEqual(cli.calls.first, ["maintenance-calendar", "--recover-failed"])
            XCTAssertFalse(model.isRunning)
        }
    }

    func testInFlightRetryDisablesButtonAndRejectsDuplicateClick() async {
        let started = expectation(description: "fake CLI started")
        let cli = BlockingRetryCLI(started: started)
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { self.noon })
        model.setForPreview(json: Self.ready)
        let action = Task { await model.retry() }
        await fulfillment(of: [started], timeout: 1)
        XCTAssertTrue(model.isRunning)
        XCTAssertFalse(model.canRetry)
        XCTAssertTrue(model.explanation.contains("does not pause services"))
        await model.retry()
        cli.release.signal()
        await action.value
        XCTAssertFalse(model.isRunning)
        XCTAssertFalse(model.canRetry)
        XCTAssertTrue(model.outcome?.contains("Done: Transcript backup retried successfully.") == true)
    }

    func testConfiguredLeadTimeDisablesStaleReadyReceipt() async {
        var calendar = Calendar(identifier: .iso8601)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let late = ISO8601DateFormatter().date(from: "2026-10-29T03:59:00Z")!
        let model = BrainBarMaintenanceRetryModel(runner: RetryCLI(Self.ready), now: { late }, calendar: calendar)
        model.setForPreview(json: Self.ready.replacingOccurrences(of: "2026-10-30", with: "2026-10-29"))
        XCTAssertFalse(model.canRetry)
        XCTAssertFalse(model.showsRetryButton)
        XCTAssertEqual(model.explanation, "Retries are paused for scheduled maintenance (until 06:00).")
    }

    func testClickQueuesBehindInFlightStatusProbe() async {
        let started = expectation(description: "status probe started")
        let cli = BlockingStatusRetryCLI(started)
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { self.noon })
        model.setForPreview(json: Self.ready)
        let refresh = Task { await model.refresh() }
        await fulfillment(of: [started], timeout: 1)
        let action = Task { await model.retry() }
        for _ in 0..<50 { await Task.yield() }
        XCTAssertEqual(cli.actionCount, 0, "retry must queue until the probe releases custody")
        cli.release.signal()
        await refresh.value
        await action.value
        XCTAssertEqual(cli.actionCount, 1)
        XCTAssertEqual(model.outcome, "Done: Transcript backup retried successfully.")
    }

}
