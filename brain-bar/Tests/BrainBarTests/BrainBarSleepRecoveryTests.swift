import XCTest
@testable import BrainBar

@MainActor
final class BrainBarSleepRecoveryTests: XCTestCase {
    private final class Clock: @unchecked Sendable {
        var date = BrainBarStatusSnapshotTests.now
        var uptime = 1000.0
    }

    func testRealPublicationExpiresWithoutAnotherRefreshIncludingMissedWake() {
        for deliveredWake in [true, false] {
            let clock = Clock()
            let base = clock.date
            let history = BrainBarSleepHistory(uptime: { clock.uptime })
            var deadlines: [@MainActor @Sendable () -> Void] = []
            let model = BrainBarSettingsViewModel(
                workspace: BrainBarRecordingWorkspace(),
                store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                    loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
                launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                initialLaunchdStates: [.watch: .running], refreshStatusOnLoad: false, now: { clock.date },
                initialWatcherHealth: .readable(.init(updatedAt: base, pollCount: 1)),
                awakeClock: { clock.uptime }, sleepHistory: history,
                statusDeadlineScheduler: { _, action in deadlines.append(action) })
            clock.date = base.addingTimeInterval(12); clock.uptime = 1012
            model.systemWillSleep()
            clock.date = base.addingTimeInterval(832)
            if deliveredWake { model.systemDidWake() }
            clock.date = base.addingTimeInterval(855); clock.uptime = 1035
            model.republishStatus()
            XCTAssertFalse(model.publishedStatus.needsAttention)
            XCTAssertEqual(model.publishedStatus.dashboardStatus.headline, "Checking…")
            if deliveredWake {
                var fault = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
                fault.healthVerdict = .init(dot: .red, checks: [
                    .init(name: "integrity", state: .red, note: "confirmed corruption")
                ], notes: [], evaluatedAt: clock.date)
                model.applyHealthBadge(fault, alerts: nil)
                XCTAssertTrue(model.publishedStatus.attention.contains { $0.id == "verdict.integrity" })
                model.applyHealthBadge(.init(badgeOn: false, reason: "", activeCodes: []), alerts: nil)
            }
            // Only the scheduled deadline publishes this outage; no new input or manual refresh.
            clock.date = base.addingTimeInterval(2700); clock.uptime = 2880
            deadlines.last?()
            XCTAssertTrue(model.publishedStatus.attention.contains { $0.id == "watcher.capture" })
            XCTAssertEqual(model.publishedStatus.verdict.dot, .red)
            XCTAssertEqual(model.publishedStatus.dashboardStatus.headline,
                           model.publishedStatus.verdict.dashboardStatus.headline)
        }
    }

    func testRealSnapshotFreshnessAndDisplayUseTheSameSleepWindow() {
        let clock = Clock()
        let base = clock.date
        let history = BrainBarSleepHistory(uptime: { clock.uptime })
        clock.date = base.addingTimeInterval(12); clock.uptime = 1012
        history.sleeping(at: clock.date)
        clock.date = base.addingTimeInterval(832)
        history.woke(at: clock.date)
        var inputs = BrainBarStatusSnapshotTests.healthy()
        inputs.sleepHistory = history
        inputs.now = base.addingTimeInterval(855)
        let file = WatcherHealthFileRead.readable(.init(updatedAt: base, pollCount: 1))
        inputs.watcher = .derive(launchd: .running, file: file, now: inputs.now, sleepHistory: history)
        inputs.watcherHeartbeatAt = base
        inputs.badge?.healthVerdict = .init(dot: .green, checks: [], notes: [], evaluatedAt: base.addingTimeInterval(-120))
        let fresh = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertEqual(fresh.verdict.dot, .green)
        XCTAssertNotNil(fresh.healthVerdict)
        XCTAssertFalse(fresh.needsAttention)
        clock.uptime = 1330
        let later = base.addingTimeInterval(1150)
        let stalled = WatcherHealthStatus.derive(launchd: .running, file: file, now: later, sleepHistory: history)
        XCTAssertTrue(stalled.reasonText(now: later, sleepHistory: history)?.contains("since 5m ago") == true)
        let health = inputs.badge!.healthVerdict!
        var utc = Calendar(identifier: .gregorian); utc.timeZone = TimeZone(secondsFromGMT: 0)!
        XCTAssertEqual(health.fallbackNotice(now: later, calendar: utc, sleepHistory: history),
                       "Health check stale since 14:21")
        inputs.now = base.addingTimeInterval(3000)
        inputs.watcher = .derive(launchd: .running, file: file, now: inputs.now, sleepHistory: history)
        let old = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertNil(old.healthVerdict)
        XCTAssertNotNil(old.staleHealthVerdict)
        // Old intervals cannot affect current freshness after the one-day horizon.
        XCTAssertEqual(history.elapsed(since: base, now: base.addingTimeInterval(90000)), 90000)
    }
}
