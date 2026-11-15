import Foundation
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarFaultNotificationTests: XCTestCase {
    private final class Center: BrainBarFaultNotificationCenter {
        var allowed = true
        var requests = 0
        var posts: [(String, String)] = []
        var authorization: CheckedContinuation<Bool, Never>?
        var holdAuthorization = false
        func authorize() async throws -> Bool {
            requests += 1
            if holdAuthorization { return await withCheckedContinuation { authorization = $0 } }
            return allowed
        }
        func post(id: String, body: String) async throws { posts.append((id, body)) }
    }

    private func defaults() -> UserDefaults {
        let name = "BrainBarFaultNotificationTests.\(UUID())"
        let value = UserDefaults(suiteName: name)!
        addTeardownBlock { value.removePersistentDomain(forName: name) }
        return value
    }

    private func capture(_ name: String? = nil, state: BrainBarHealthVerdict.State = .red) -> BrainBarMenuCapture {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        var c = BrainBarMenuCapture(now: now, timeZone: TimeZone(secondsFromGMT: 0)!)
        c.verdict = BrainBarHealthVerdict(dot: name == nil || state != .red ? .green : .red,
            checks: BrainBarMenuPresenter.known.map {
                .init(name: $0, state: $0 == name ? state : .green, note: "Invented check", unknownTicks: 0, blind: false)
            }, notes: [], evaluatedAt: now)
        c.maintenanceNightFailures = 0
        c.backupDailyFailures = 0
        c.jsonlBackupFailures = 0
        return c
    }

    func testNewSameGoneAndReturned() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let c = capture("search")
        await subject.update(c)
        XCTAssertEqual(center.posts.count, BrainBarMenuPresenter.notificationFaults(c).count)
        XCTAssertFalse(center.posts.isEmpty)
        let count = center.posts.count
        for _ in 0..<20 { await subject.update(c) }
        XCTAssertEqual(center.posts.count, count)
        await subject.update(capture())
        await subject.update(c)
        XCTAssertEqual(center.posts.count, count * 2)
        XCTAssertEqual(center.posts.first?.1, BrainBarMenuPresenter.present(c).rows.first { $0.kind == .needsYou }?.strings.first)
    }

    func testRelaunchKeepsSet() async {
        let d = defaults(), center = Center()
        await BrainBarFaultNotifications(center: center, defaults: d).update(capture("disk"))
        let next = Center()
        await BrainBarFaultNotifications(center: next, defaults: d).update(capture("disk"))
        XCTAssertEqual(next.posts.count, 0)
        XCTAssertEqual(next.requests, 0)
    }

    func testCheckingAndCantTellNeverRequestAuthorization() async {
        let center = Center()
        let checking = BrainBarMenuCapture(now: Date(), timeZone: .current)
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults())
        await observed.update(checking)
        await observed.update(capture("search", state: .unknown))
        XCTAssertEqual(center.posts.count, 0)
        XCTAssertEqual(center.requests, 0)
    }

    func testStaleProducerSuppressedButAppFactsNotifyAndRearm() async {
        let center = Center()
        var c = capture("disk")
        c.verdictAwakeAgeS = 901
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults())
        await observed.update(c)
        XCTAssertTrue(center.posts.isEmpty)
        c.maintenanceNightFailures = 3
        await observed.update(c)
        XCTAssertEqual(center.posts.count, 1)
        c.maintenanceNightFailures = 0
        await observed.update(c)
        c.maintenanceNightFailures = 4
        await observed.update(c)
        XCTAssertEqual(center.posts.count, 2)
    }

    func testStaleGapDoesNotRearmKnownProducer() async {
        let center = Center()
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults())
        let c = capture("search")
        await observed.update(c)
        let count = center.posts.count
        var missing = c
        missing.verdict = nil
        await observed.update(missing)
        await observed.update(c)
        XCTAssertEqual(center.posts.count, count)
    }

    func testDeniedLogsOnceAndDoesNotRepeatRequests() async {
        let center = Center(), d = defaults()
        center.allowed = false
        var logs: [String] = []
        let subject = BrainBarFaultNotifications(center: center, defaults: d, log: { logs.append($0) })
        await subject.update(capture("disk"))
        await subject.update(capture())
        await subject.update(capture("search"))
        XCTAssertTrue(center.posts.isEmpty)
        XCTAssertEqual(center.requests, 1)
        XCTAssertEqual(logs.count, 1)
        await BrainBarFaultNotifications(center: center, defaults: d, log: { logs.append($0) }).update(capture("disk"))
        XCTAssertEqual(logs.count, 1)
    }

    func testRecoveryWhileAuthorizationPendingDoesNotPostOldFault() async {
        let center = Center()
        center.holdAuthorization = true
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let update = Task { await subject.update(capture("disk")) }
        while center.authorization == nil { await Task.yield() }
        await subject.update(capture())
        center.authorization?.resume(returning: true)
        await update.value
        XCTAssertTrue(center.posts.isEmpty)
    }

    func testUnknownProducerWhileAuthorizationPendingOnlyPostsAppFault() async {
        let center = Center()
        center.holdAuthorization = true
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let update = Task { await subject.update(capture("disk")) }
        while center.authorization == nil { await Task.yield() }
        var c = capture("disk")
        c.verdictAwakeAgeS = 901
        c.maintenanceNightFailures = 3
        await subject.update(c)
        center.authorization?.resume(returning: true)
        await update.value
        XCTAssertEqual(center.posts.map { $0.0 }, BrainBarMenuPresenter.notificationFaults(c).map(\.id))
    }

    func testDisabledMenuStillNotifiesAndAllNeedsRowsAreCovered() async throws {
        let d = defaults(), center = Center()
        d.set(false, forKey: "BrainBarNewMenuEnabled")
        await BrainBarFaultNotifications(center: center, defaults: d).update(capture("search"))
        XCTAssertFalse(center.posts.isEmpty)
        for fixture in try BrainBarMenuPresentationTests.loadCases() {
            let c = try BrainBarMenuCases.capture(fixture["inputs"] as! [String: Any])
            XCTAssertEqual(BrainBarMenuPresenter.present(c), BrainBarMenuPresentationTests.expected(fixture))
            let rows = BrainBarMenuPresenter.present(c).rows.filter { $0.kind == .needsYou }
            let faults = BrainBarMenuPresenter.notificationFaults(c)
            if c.verdict != nil, (0...900).contains(c.verdictAwakeAgeS) {
                XCTAssertEqual(faults.map(\.title), rows.compactMap { $0.strings.first })
                XCTAssertEqual(Set(faults.map(\.id)).count, faults.count)
            }
        }
    }

    func testIdentityUsesThingAndActionRatherThanWords() {
        var c = capture()
        c.maintenanceNightFailures = 3
        let first = BrainBarMenuPresenter.notificationFaults(c)
        c.maintenanceNightFailures = 9
        let second = BrainBarMenuPresenter.notificationFaults(c)
        XCTAssertEqual(first.map(\.id), second.map(\.id))
        XCTAssertNotEqual(first.map(\.title), second.map(\.title))
    }
}
