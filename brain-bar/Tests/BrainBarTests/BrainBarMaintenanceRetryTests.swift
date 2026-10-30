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

    func testReadyRowListsFailuresAndWindowOnlyReason() async {
        let cli = RetryCLI(Self.ready)
        let model = BrainBarMaintenanceRetryModel(runner: cli, now: { self.noon }, calendar: Calendar(identifier: .iso8601))
        await model.refresh()
        XCTAssertTrue(model.canRetry)
        XCTAssertTrue(model.details.contains("Transcript backup didn’t finish last night (it stopped with an error). You can retry it now."))
        XCTAssertFalse(model.details.contains("jsonl-backup"))
        XCTAssertFalse(model.details.contains("exit 1"))
        XCTAssertTrue(model.explanation.contains("Retry will run: Transcript backup"))
        // The step's own reason carries its timing; no blanket 04:00 promise (CodeRabbit r4420436682).
        XCTAssertTrue(model.details.contains("Weekly maintenance — it pauses database writes, so it only runs between 04:00 and 06:00."),
                      model.details)
        XCTAssertFalse(model.details.contains("runs tonight at 04:00"), model.details)
        XCTAssertEqual(cli.calls, [["maintenance-calendar", "--recover-failed", "--status"]])
    }

    func testDisabledStatesAndNoRunnerNeverLaunch() async {
        for status in ["refused", "nothing-to-retry"] {
            let cli = RetryCLI("{\"status\":\"\(status)\",\"reason\":\"fixture explanation\",\"steps\":[]}")
            let model = BrainBarMaintenanceRetryModel(runner: cli)
            await model.refresh()
            XCTAssertFalse(model.canRetry)
            XCTAssertEqual(model.explanation, "fixture explanation")
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
        XCTAssertTrue(model.explanation.contains("nightly maintenance"))
        await model.retry()
        XCTAssertEqual(cli.calls.count, 1)
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
        XCTAssertTrue(model.explanation.contains("01:07"))
        XCTAssertTrue(model.explanation.contains("06:00"))
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
