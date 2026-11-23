import XCTest
import AppKit
import BrainBarLifecycle
@testable import BrainBar

@MainActor
final class BrainBarWakeRefreshTests: XCTestCase {
    final class Clock: @unchecked Sendable {
        var date = BrainBarStatusSnapshotTests.now
        var uptime = 1000.0
    }

    final class Source: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        var reads: Int { lock.withLock { count } }
        func sample(_ url: URL) -> WatcherHealthFileRead {
            lock.withLock { count += 1 }
            return .unreadable(path: url.path, reason: "Fixture has no heartbeat.")
        }
    }

    actor Gate {
        let initial: XCTestExpectation, held: XCTestExpectation
        var reads = 0
        var continuation: CheckedContinuation<ObservabilityReadResult, Never>?
        init(initial: XCTestExpectation, held: XCTestExpectation) {
            self.initial = initial; self.held = held
        }
        func sample() async -> ObservabilityReadResult {
            reads += 1
            if reads == 1 { initial.fulfill() }
            else if reads == 2 {
                return await withCheckedContinuation {
                    continuation = $0
                    held.fulfill()
                }
            }
            return .unreadable("Invented backup status unavailable.")
        }
        func release() {
            continuation?.resume(returning: .unreadable("Invented older read."))
            continuation = nil
        }
    }

    func model(_ clock: Clock, source: Source? = nil, gate: Gate? = nil, log: BrainBarLogFile? = nil,
               scheduler: @escaping @MainActor (TimeInterval, @escaping @MainActor @Sendable () -> Void) -> Void = { _, _ in }) -> BrainBarSettingsViewModel {
        BrainBarSettingsViewModel(
            workspace: BrainBarRecordingWorkspace(),
            store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            refreshStatusOnLoad: false, now: { clock.date },
            observabilityURL: gate == nil ? nil : URL(fileURLWithPath: "/fixture/observability"),
            observabilityCadence: .known(300),
            observabilityRead: { _ in await gate!.sample() },
            watcherHealthURL: source == nil ? nil : URL(fileURLWithPath: "/fixture/health"),
            watcherHealthRead: { source!.sample($0) },
            awakeClock: { clock.uptime }, sleepHistory: BrainBarSleepHistory(uptime: { clock.uptime }),
            statusDeadlineScheduler: scheduler, lifecycleLog: log, observeSystemLifecycle: false)
    }

    func testWakeImmediatelySamplesOnceForRepeatedNotices() async {
        let source = Source()
        let clock = Clock()
        let model = model(clock, source: source)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        model.systemDidWake()
        model.systemDidWake()
        XCTAssertNotNil(model.wakeRefreshTask)
        await model.wakeRefreshTask?.value
        XCTAssertEqual(source.reads, 1)
    }

    func testWakeDoesNotWaitForAnOlderSuspendedTimerRead() async {
        let initial = expectation(description: "initial fixture read")
        let held = expectation(description: "old timer read suspended")
        let gate = Gate(initial: initial, held: held)
        let clock = Clock()
        let model = model(clock, gate: gate)
        await fulfillment(of: [initial], timeout: 1)
        let tick = Task { await model.sampleStatusTick(includeLaunchd: true) }
        await fulfillment(of: [held], timeout: 1)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        XCTAssertNotNil(model.wakeRefreshTask)
        await model.wakeRefreshTask?.value
        let reads = await gate.reads
        XCTAssertEqual(reads, 3, "Wake completes its fresh read while the older read is still held")
        await gate.release()
        await tick.value
    }

    func testWakeWithoutVerdictShowsSleepSentenceInBothDesigns() {
        let clock = Clock()
        let model = model(clock)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        let capture = model.menuCapture(stats: nil)
        XCTAssertEqual(model.publishedStatus.dashboardStatus.headline, "Checking after sleep…")
        XCTAssertEqual(BrainBarStatusPopoverController.statusLineTitle(for: model.publishedStatus.verdict), "Checking after sleep…")
        XCTAssertEqual(BrainBarMenuPresenter.present(capture).headline, "Checking after sleep…")
        XCTAssertEqual(NDModel.make(NDInputs(capture: capture)).shownPresentation.headline, "Checking after sleep…")
    }

    func testActivationAfterSleepRecoversAMissedWakeNotice() async {
        let source = Source()
        let clock = Clock()
        let model = model(clock, source: source)
        model.systemWillSleep()
        XCTAssertEqual(model.sleepStartedAt, clock.date)
        clock.date.addTimeInterval(600)
        model.applicationDidBecomeActive()
        model.applicationDidBecomeActive()
        XCTAssertNotNil(model.wakeRefreshTask)
        await model.wakeRefreshTask?.value
        XCTAssertNil(model.sleepStartedAt)
        XCTAssertEqual(source.reads, 1)
    }

    func testPreWakeGreenCannotBecomeCurrentAfterDeadline() {
        let clock = Clock()
        var deadlines: [@MainActor @Sendable () -> Void] = []
        let model = model(clock, scheduler: { _, action in deadlines.append(action) })
        var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        badge.healthVerdict = .init(dot: .green, checks: [], notes: [], evaluatedAt: clock.date)
        model.applyHealthBadge(badge, alerts: nil)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        clock.date.addTimeInterval(500)
        clock.uptime += 500
        deadlines.last?()
        XCTAssertNil(model.publishedStatus.healthVerdict)
        XCTAssertNotEqual(model.publishedStatus.verdict.dot, .green)
        XCTAssertTrue(model.publishedStatus.dashboardStatus.reason?.contains("Health check stale") == true)
    }

    func testFreshPostWakeVerdictEndsCheckingImmediately() {
        let clock = Clock()
        let model = model(clock)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        clock.date.addTimeInterval(2)
        clock.uptime += 2
        var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        badge.healthVerdict = .init(dot: .green, checks: [], notes: [], evaluatedAt: clock.date)
        model.applyHealthBadge(badge, alerts: nil)
        XCTAssertEqual(model.publishedStatus.healthVerdict, badge.healthVerdict)
        XCTAssertFalse(model.menuCapture(stats: nil).wakeCatchUpPending)
    }

    private func logURL() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(".build.noindex/wake-refresh-log-fixtures")
            .appendingPathComponent(UUID().uuidString + ".log")
    }

    func testWakeTimeoutWithoutVerdictExplainsMissingFreshReading() {
        for unreadable in [false, true] {
            let clock = Clock()
            var deadlines: [@MainActor @Sendable () -> Void] = []
            let model = model(clock, scheduler: { _, action in deadlines.append(action) })
            var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
            badge.healthVerdictUnreadable = unreadable
            model.applyHealthBadge(badge, alerts: nil)
            let wasChecking = model.publishedStatus.isChecking
            model.systemWillSleep()
            clock.date.addTimeInterval(600)
            model.systemDidWake()
            clock.date.addTimeInterval(361); clock.uptime += 361
            deadlines.last?()
            XCTAssertTrue(model.publishedStatus.dashboardStatus.reason?.contains("No health reading from after wake.") == true)
            XCTAssertNotEqual(model.publishedStatus.dashboardStatus.tone, .green)
            XCTAssertEqual(model.publishedStatus.verdict.dot, .amber)
            XCTAssertEqual(model.publishedStatus.isChecking, wasChecking)
        }
    }

    func testAlreadyStaleVerdictKeepsAmberAfterWakeDeadlineAndLogsItsReason() throws {
        for state in [BrainBarHealthVerdict.State.green, .amber] {
            let clock = Clock()
            let url = logURL()
            defer { try? FileManager.default.removeItem(at: url) }
            var deadlines: [@MainActor @Sendable () -> Void] = []
            let model = model(clock, log: BrainBarLog.lifecycleLogFile(path: url.path),
                              scheduler: { _, action in deadlines.append(action) })
            var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
            badge.healthVerdict = .init(dot: state, checks: [], notes: [],
                                       evaluatedAt: clock.date.addingTimeInterval(-901))
            model.applyHealthBadge(badge, alerts: nil)
            XCTAssertEqual(model.publishedStatus.verdict.dot, .amber)
            model.systemWillSleep()
            clock.date.addTimeInterval(600)
            model.systemDidWake()
            clock.date.addTimeInterval(361); clock.uptime += 361
            deadlines.last?()
            XCTAssertEqual(model.publishedStatus.verdict.dot, .amber)
            XCTAssertEqual(model.publishedStatus.staleHealthVerdict, badge.healthVerdict)
            let reasons = model.publishedStatus.issues.filter { $0.id == "verdict.stale" }
            XCTAssertEqual(reasons.count, 1)
            let reason = try XCTUnwrap(reasons.first?.text)
            XCTAssertTrue(reason.contains("Health check stale"))
            let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
            XCTAssertTrue(try XCTUnwrap(lines.last).contains("status still not confirmed 361 s after wake: " + reason))
        }
    }

    func testLifecycleLogRecordsSleepWakeAndFirstConfirmation() throws {
        let url = logURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let clock = Clock()
        let model = model(clock, log: BrainBarLog.lifecycleLogFile(path: url.path))
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        model.systemDidWake()
        clock.date.addTimeInterval(2); clock.uptime += 2
        var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        badge.healthVerdict = .init(dot: .green, checks: [], notes: [], evaluatedAt: clock.date)
        model.applyHealthBadge(badge, alerts: nil)
        model.republishStatus()
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 3)
        XCTAssertTrue(lines[0].contains("sleep at 2026-10-26T14:10:00Z"))
        XCTAssertTrue(lines[1].contains("woke at 2026-10-26T14:20:00Z, slept 600 s, status refresh asked"))
        XCTAssertTrue(lines[2].contains("status confirmed 2 s after wake: "))
        XCTAssertTrue(lines[2].hasSuffix(model.publishedStatus.dashboardStatus.headline))
        XCTAssertFalse(lines[2].contains("Checking after sleep"))
    }

    func testLifecycleTimeoutIsLoggedOnceWithReason() throws {
        let url = logURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let clock = Clock()
        var deadlines: [@MainActor @Sendable () -> Void] = []
        let model = model(clock, log: BrainBarLog.lifecycleLogFile(path: url.path),
                          scheduler: { _, action in deadlines.append(action) })
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        clock.date.addTimeInterval(361); clock.uptime += 361
        deadlines.last?()
        deadlines.last?()
        model.republishStatus()
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 3)
        XCTAssertTrue(lines[2].contains("status still not confirmed 361 s after wake: No health reading from after wake."))
        XCTAssertFalse(model.menuCapture(stats: nil).wakeCatchUpPending)
    }
}
