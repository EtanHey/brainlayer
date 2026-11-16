import Combine
import Foundation
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarFaultNotificationTests: XCTestCase {
    @MainActor private final class Box {
        var count = 0
        var logs: [String] = []
        var now = Date(timeIntervalSince1970: 1_800_000_000)
    }

    private final class CountingDefaults: UserDefaults {
        var deliveredWrites = 0
        override func set(_ value: Any?, forKey key: String) {
            if key.hasPrefix("BrainBarNotificationDelivered") || key.hasPrefix("BrainBarNotificationRecent") { deliveredWrites += 1 }
            super.set(value, forKey: key)
        }
    }

    private final class Center: BrainBarFaultNotificationCenter {
        enum Failure: Error { case invented }
        var postFailures = 0
        var authorizationFailures = 0
        var holdPost = false
        var posting: CheckedContinuation<Void, Never>?
        var didPost: (@MainActor () -> Void)?
        var allowed = true
        var requests = 0
        var posts: [(String, String)] = []
        var authorization: CheckedContinuation<Bool, Never>?
        var holdAuthorization = false
        func authorize() async throws -> Bool {
            requests += 1
            if authorizationFailures > 0 { authorizationFailures -= 1; throw Failure.invented }
            if holdAuthorization { return await withCheckedContinuation { authorization = $0 } }
            return allowed
        }
        func post(id: String, body: String) async throws {
            if postFailures > 0 { postFailures -= 1; throw Failure.invented }
            if holdPost { await withCheckedContinuation { posting = $0 } }
            posts.append((id, body))
            didPost?()
        }
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

    // Reported 2026-11-16: the producer has noticed lag once, with no attention items.
    private func lagCapture(state: BrainBarHealthVerdict.State = .amber, age: TimeInterval = 300) -> BrainBarMenuCapture {
        var c = capture("ingest", state: state)
        c.verdict = .init(dot: state == .red ? .red : .amber, checks: c.verdict!.checks.map { k in
            k.name == "ingest" ? .init(name: k.name, state: k.state,
                note: "watcher reports offset_lag (1 checks in a row)", unknownTicks: 0, blind: false) : k
        }, notes: [], evaluatedAt: c.now)
        c.watcher = .init(alertReasons: ["offset_lag"])
        c.lagEpisodeStart = c.now.addingTimeInterval(-age)
        return c
    }

    func testReportedAmberLagDoesNotNeedYouOrNotify() async {
        let c = lagCapture()
        let p = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(p.headline, BrainBarMenuPresenter.Say.r02Headline.text)
        XCTAssertEqual(p.form, .calm)
        XCTAssertFalse(p.rows.contains { $0.kind == .needsYou })
        XCTAssertEqual(p.icon, .resting)
        XCTAssertEqual(p.count, 0)
        let report = BrainBarMenuReport.text(presentation: p, verdict: c.verdict, version: "test", now: c.now)
        XCTAssertTrue(report.contains(BrainBarMenuPresenter.Say.r02Headline.text))
        XCTAssertFalse(report.contains(BrainBarMenuPresenter.Say.cantFix.text))
        XCTAssertTrue(BrainBarMenuPresenter.notificationFaults(c).isEmpty)
        let center = Center()
        await BrainBarFaultNotifications(center: center, defaults: defaults()).update(c)
        XCTAssertTrue(center.posts.isEmpty)
        XCTAssertEqual(center.requests, 0)
    }

    func testReportedAmberLagWithIncompleteCensusStaysCalm() async {
        var c = lagCapture()
        c.verdict = .init(dot: .amber, checks: c.verdict!.checks.map { check in
            var check = check
            if check.name == "ingest" { check.contextCodes = ["agent_census_incomplete"] }
            return check
        }, notes: [], evaluatedAt: c.now)
        let p = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(p.headline, BrainBarMenuPresenter.Say.r02Headline.text)
        XCTAssertEqual(p.count, 0)
        XCTAssertEqual(p.icon, .resting)
        XCTAssertFalse(p.rows.contains { $0.kind == .needsYou })
        XCTAssertFalse(p.rows.contains { $0.state == .notice })
        let center = Center()
        await BrainBarFaultNotifications(center: center, defaults: defaults()).update(c)
        XCTAssertTrue(center.posts.isEmpty)
    }

    func testLagBoundaryAndUncertainEvidence() {
        for age in [300.0, 1800.0] {
            XCTAssertEqual(BrainBarMenuPresenter.present(lagCapture(age: age)).form, .calm)
        }
        for state in [BrainBarHealthVerdict.State.green, .amber, .red, .unknown] {
            let c = lagCapture(state: state)
            XCTAssertEqual(BrainBarMenuPresenter.Say.catchingUp(c), state == .green || state == .amber)
        }
        var c = lagCapture()
        c.watcher?.dbProbeFailed = true
        XCTAssertFalse(BrainBarMenuPresenter.Say.catchingUp(c))
        c = lagCapture()
        c.watcher?.fresh = false
        XCTAssertFalse(BrainBarMenuPresenter.Say.catchingUp(c))
    }

    func testClassicAmberLagHasNoAttentionItems() {
        var input = BrainBarStatusSnapshotTests.healthy()
        let c = lagCapture()
        input.now = c.now
        input.badge?.healthVerdict = c.verdict
        let status = BrainBarStatusSnapshot.derive(input)
        XCTAssertEqual(status.verdict.dot, .amber)
        XCTAssertEqual(status.verdict.attentionRowCount, 0)
        XCTAssertEqual(status.dashboardStatus.headline, "Check soon")
    }

    func testLagRedAndThirtyMinuteEscalationStillNeedYou() async {
        for c in [lagCapture(state: .red), lagCapture(age: 1801)] {
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(p.form, .needs)
            XCTAssertEqual(p.icon, .marked)
            XCTAssertEqual(p.count, 1)
            XCTAssertEqual(p.rows.filter { $0.strings.first == BrainBarMenuPresenter.Say.generic.text }.count, 1)
            let center = Center()
            await BrainBarFaultNotifications(center: center, defaults: defaults()).update(c)
            XCTAssertEqual(center.posts.count, 1)
        }
    }

    func testAmberIngestWithoutLagIsR31Notice() {
        var c = lagCapture()
        c.watcher = .init()
        c.verdict = .init(dot: .amber, checks: c.verdict!.checks.map { k in
            k.name == "ingest" ? .init(name: k.name, state: k.state,
                note: "an agent BrainLayer does not index is running", unknownTicks: 0, blind: false) : k
        }, notes: [], evaluatedAt: c.now)
        let p = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(p.form, .calm)
        XCTAssertEqual(p.rows.first { $0.state == .notice }?.strings.first, BrainBarMenuPresenter.Say.generic.text)
    }

    func testNewSameGoneAndReturned() async {
        let clock = Box()
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), clock: { clock.now })
        var c = capture("search")
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
        XCTAssertFalse(center.posts.isEmpty)
        let count = center.posts.count
        for _ in 0..<20 { await subject.update(c) }
        XCTAssertEqual(center.posts.count, count)
        await subject.update(capture())
        c.now = c.now.addingTimeInterval(601)
        clock.now = clock.now.addingTimeInterval(601)
        await subject.update(c)
        XCTAssertEqual(center.posts.count, count * 2)
        XCTAssertEqual(center.posts.first?.1, "Memory search needs attention")
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
        let clock = Box()
        let center = Center()
        var c = capture("disk")
        c.verdictAwakeAgeS = 901
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults(), clock: { clock.now })
        await observed.update(c)
        XCTAssertTrue(center.posts.isEmpty)
        c.maintenanceNightFailures = 3
        await observed.update(c)
        XCTAssertEqual(center.posts.count, 1)
        await observed.update(capture())
        c.maintenanceNightFailures = 4
        c.now = c.now.addingTimeInterval(601)
        clock.now = clock.now.addingTimeInterval(601)
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

    func testGoldenMenuPresentationsRemainUnchanged() throws {
        for fixture in try BrainBarMenuPresentationTests.loadCases() {
            let c = try BrainBarMenuCases.capture(fixture["inputs"] as! [String: Any])
            XCTAssertEqual(BrainBarMenuPresenter.present(c), BrainBarMenuPresentationTests.expected(fixture))
            let rows = BrainBarMenuPresenter.present(c).rows.filter { $0.kind == .needsYou }
            let faults = BrainBarMenuPresenter.notificationFaults(c)
            if c.verdict != nil, (0...900).contains(c.verdictAwakeAgeS) {
                XCTAssertLessThanOrEqual(faults.count, rows.count)
                if !rows.isEmpty { XCTAssertFalse(faults.isEmpty) }
                XCTAssertFalse(faults.contains { $0.title == "Something BrainBar checks needs attention." })
                XCTAssertEqual(Set(faults.map(\.id)).count, faults.count)
            }
        }
    }

    func testIdentityUsesThingRatherThanWords() {
        var c = capture()
        c.maintenanceNightFailures = 3
        let first = BrainBarMenuPresenter.notificationFaults(c)
        c.maintenanceNightFailures = 9
        let second = BrainBarMenuPresenter.notificationFaults(c)
        XCTAssertEqual(first.map(\.id), second.map(\.id))
        XCTAssertNotEqual(first.map(\.title), second.map(\.title))
    }

    func testRound1UnknownProducerIsNotRecovery() async {
        let center = Center()
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults())
        await observed.update(capture("search"))
        await observed.update(capture("search", state: .unknown))
        await observed.update(capture("search"))
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound1PartialRelaunchDoesNotPruneAppFault() async {
        let d = defaults(), center = Center()
        var c = capture()
        c.maintenanceNightFailures = 3
        await BrainBarFaultNotifications(center: center, defaults: d).update(c)
        let next = Center(), relaunched = BrainBarFaultNotifications(center: next, defaults: d)
        await relaunched.update(BrainBarMenuCapture(now: c.now, timeZone: c.timeZone))
        await relaunched.update(c)
        XCTAssertTrue(next.posts.isEmpty)
    }

    func testRound1DriveWordingDoesNotCreateNewThing() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        var c = capture("drive_auth")
        await subject.update(c)
        c.driveAuthModel = .missing
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound1NewRedCauseNotifiesSameThingAgain() async {
        let center = Center()
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults())
        var c = capture("backup_local")
        await observed.update(c)
        c.verdict = BrainBarHealthVerdict(dot: .red, checks: BrainBarMenuPresenter.known.map {
            .init(name: $0, state: ["backup_local", "integrity"].contains($0) ? .red : .green,
                  note: "Invented cause", unknownTicks: 0, blind: false)
        }, notes: [], evaluatedAt: c.now)
        await observed.update(c)
        XCTAssertEqual(center.posts.count, 2)
        XCTAssertNotEqual(center.posts.first?.1, center.posts.last?.1)
        XCTAssertEqual(center.posts.last?.1, BrainBarMenuPresenter.present(c).rows.first { $0.kind == .needsYou }?.strings.first)
    }

    func testRound1FailedAddRetriesNextUpdate() async {
        let center = Center()
        center.postFailures = 1
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), log: { _ in })
        await subject.update(capture("disk"))
        await subject.update(capture("disk"))
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound1PendingAuthorizationIsNotPersistedAsDelivered() async {
        let d = defaults(), center = Center()
        center.holdAuthorization = true
        let subject = BrainBarFaultNotifications(center: center, defaults: d)
        let pending = Task { await subject.update(capture("disk")) }
        while center.authorization == nil { await Task.yield() }
        let next = Center()
        await BrainBarFaultNotifications(center: next, defaults: d).update(capture("disk"))
        center.authorization?.resume(returning: false)
        await pending.value
        XCTAssertEqual(next.posts.count, 1)
    }

    func testRound1StalePendingPostsWhenFreshAgain() async {
        let center = Center()
        center.holdAuthorization = true
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let pending = Task { await subject.update(capture("disk")) }
        while center.authorization == nil { await Task.yield() }
        var c = capture("disk")
        c.verdictAwakeAgeS = 901
        await subject.update(c)
        center.authorization?.resume(returning: true)
        await pending.value
        XCTAssertTrue(center.posts.isEmpty)
        center.holdAuthorization = false
        await subject.update(capture("disk"))
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound1DenialRechecksAndLogsStateChangesOnly() async {
        let center = Center()
        center.allowed = false
        var logs: [String] = []
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), log: { logs.append($0) })
        await subject.update(capture("disk"))
        await subject.update(capture("disk"))
        XCTAssertEqual(logs.count, 1)
        center.allowed = true
        await subject.update(capture("disk"))
        XCTAssertEqual(center.posts.count, 1)
        center.allowed = false
        await subject.update(capture("search"))
        XCTAssertEqual(logs.count, 2)
    }

    func testRound1GenericNotificationNamesThing() async {
        let center = Center()
        await BrainBarFaultNotifications(center: center, defaults: defaults()).update(capture("search"))
        XCTAssertEqual(center.posts.first?.1, "Memory search needs attention")
    }

    func testWatcherNoticeUnknownGapNeverNotifies() async {
        let center = Center()
        let observed = BrainBarFaultNotifications(center: center, defaults: defaults())
        var c = capture()
        c.watcher?.alertReasons = ["file_ingestion_failure"]
        await observed.update(c)
        var missing = c
        missing.watcher = nil
        await observed.update(missing)
        await observed.update(c)
        XCTAssertEqual(center.posts.count, 0)
    }

    func testAppKnownHistoryUnknownGapDoesNotRearm() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        var c = capture()
        c.maintenanceNightFailures = 3
        await subject.update(c)
        var missing = c
        missing.maintenanceNightFailures = nil
        await subject.update(missing)
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
    }

    func testWatcherNoticeDoesNotAddACauseToAnExistingRedFault() async {
        let clock = Box()
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), clock: { clock.now })
        var c = capture("ingest")
        await subject.update(c)
        c.watcher?.alertReasons = ["file_ingestion_failure"]
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
        c.watcher?.alertReasons = []
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
        c.watcher?.alertReasons = ["file_ingestion_failure"]
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
        c.now = c.now.addingTimeInterval(601)
        clock.now = clock.now.addingTimeInterval(601)
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 1)
    }

    func testAuthorizationErrorRemainsPending() async {
        let center = Center()
        center.authorizationFailures = 1
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), log: { _ in })
        await subject.update(capture("disk"))
        await subject.update(capture("disk"))
        XCTAssertEqual(center.posts.count, 1)
    }

    func testJobAlertsOwnMessageUnknownRecoveryAndRelaunch() async {
        let clock = Box()
        let center = Center(), d = defaults()
        let subject = BrainBarFaultNotifications(center: center, defaults: d, clock: { clock.now })
        let alerts: [String: String] = ["mcp-transport": "Invented transport fault"]
        await subject.update(capture(), jobAlerts: alerts)
        await subject.update(capture(), jobAlerts: nil)
        await subject.update(capture(), jobAlerts: alerts)
        XCTAssertEqual(center.posts.count, 1)
        XCTAssertEqual(center.posts.first?.0, "job:mcp-transport")
        XCTAssertEqual(center.posts.first?.1, "Invented transport fault")
        let next = Center()
        let relaunched = BrainBarFaultNotifications(center: next, defaults: d, clock: { clock.now })
        await relaunched.update(BrainBarMenuCapture(now: Date(), timeZone: .current))
        await relaunched.update(capture(), jobAlerts: alerts)
        XCTAssertTrue(next.posts.isEmpty)
        await relaunched.update(capture(), jobAlerts: [:])
        var returned = capture()
        returned.now = returned.now.addingTimeInterval(601)
        clock.now = clock.now.addingTimeInterval(601)
        await relaunched.update(returned, jobAlerts: alerts)
        XCTAssertEqual(next.posts.count, 1)
    }

    func testEveryJobKeyHasAnIndependentEpisode() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let keys: [String] = [
            "brainbar-activation",
            "brainbar-activation-brew",
            "backup-daily",
            "backup-snapshot",
            "backup-upload-abandoned",
            "drive-consent",
            "fleet-watchdog-hold",
            "index",
            "jsonl-backup",
            "jsonl-retention",
            "maintenance-light",
            "maintenance-full",
            "maintenance-burn",
            "mcp-transport",
            "repair-fts",
            "wal-checkpoint",
            "wal-checkpoint-fallback",
        ]
        let alerts: [String: String] = Dictionary(uniqueKeysWithValues: keys.map { ($0, "Invented fault for " + $0) })
        await subject.update(capture(), jobAlerts: alerts)
        XCTAssertEqual(Set(center.posts.map { $0.0 }), Set(keys.filter { $0 != "brainbar-activation" }.map { "job:" + $0 }))
    }

    func testRecoveryAcrossUnknownWhileAddIsInFlightDoesNotSuppressNextEpisode() async {
        let center = Center()
        center.holdPost = true
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let alerts: [String: String] = ["mcp-transport": "Invented transport fault"]
        let pending = Task { await subject.update(capture(), jobAlerts: alerts) }
        while center.posting == nil { await Task.yield() }
        await subject.update(capture(), jobAlerts: nil)
        await subject.update(capture(), jobAlerts: [:])
        await subject.update(capture(), jobAlerts: alerts)
        center.holdPost = false
        center.posting?.resume()
        await pending.value
        XCTAssertEqual(center.posts.count, 2)
    }

    func testNotificationDismissalDoesNotOpenSettings() {
        for selected in [false, true] {
            var settings = 0, popups = 0
            BrainBarNotificationClick.perform(hasButton: true, popup: { popups += 1; return selected }, settings: { settings += 1 })
            XCTAssertEqual(popups, 1)
            XCTAssertEqual(settings, 0)
        }
        var settings = 0
        BrainBarNotificationClick.perform(hasButton: false, popup: { XCTFail("No button"); return false }, settings: { settings += 1 })
        XCTAssertEqual(settings, 1)
    }

    func testMenuOffStillDeliversThroughTheAppLifetimeSubscription() async {
        let d = defaults(), center = Center()
        d.set(false, forKey: "BrainBarNewMenuEnabled")
        let subject = BrainBarFaultNotifications(center: center, defaults: d)
        let changes = PassthroughSubject<Void, Never>()
        let posted = expectation(description: "Menu-off publisher delivers")
        center.didPost = { posted.fulfill() }
        let c = capture("search")
        subject.observe(changes: changes.eraseToAnyPublisher()) { .init(capture: c) }
        changes.send(())
        await fulfillment(of: [posted], timeout: 5)
        subject.stopObserving()
        XCTAssertEqual(center.posts.count, 1)
    }

    func testPublishedBurstIsCoalescedBeforeCapture() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let changes = PassthroughSubject<Void, Never>()
        let first = expectation(description: "First delivery")
        let coalesced = expectation(description: "One latest capture")
        center.didPost = { first.fulfill() }
        let captures = Box()
        let c = capture("search")
        subject.observe(changes: changes.eraseToAnyPublisher()) {
            captures.count += 1
            if captures.count == 2 { coalesced.fulfill() }
            return .init(capture: c)
        }
        await fulfillment(of: [first], timeout: 5)
        for _ in 0..<100 { changes.send(()) }
        await fulfillment(of: [coalesced], timeout: 5)
        subject.stopObserving()
        XCTAssertEqual(captures.count, 2)
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound1WordedToGenericKeepsOneEpisode() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let ages: [Double] = [10, 30]
        for age in ages {
            var c = capture()
            c.verdict = BrainBarHealthVerdict(dot: .amber, checks: BrainBarMenuPresenter.known.map { name -> BrainBarHealthVerdict.Check in
                .init(name: name, state: name == "embedding" ? .amber : .green,
                      ageS: name == "embedding" ? age : nil, thresholdS: name == "embedding" ? 20 : nil,
                      note: "Invented aging check")
            }, notes: [], evaluatedAt: c.now)
            await subject.update(c)
        }
        XCTAssertEqual(center.posts.count, 0)
    }

    func testRound2RecoveredThingRearmsWhileOtherThingRemainsUnknown() async {
        let clock = Box()
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), clock: { clock.now })
        var c = capture("disk")
        c.watcher = nil
        await subject.update(c)
        for _ in 0..<60 {
            c = capture()
            c.watcher = nil
            c.now = c.now.addingTimeInterval(3600)
            clock.now = clock.now.addingTimeInterval(3600)
            await subject.update(c)
        }
        c = capture("disk")
        c.watcher = nil
        c.now = c.now.addingTimeInterval(3601)
        clock.now = clock.now.addingTimeInterval(3601)
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 2)
    }

    func testRound2NewDiskCauseNamesItsOwnRowInsteadOfDrive() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        var c = capture("drive_auth")
        await subject.update(c)
        c.verdict = BrainBarHealthVerdict(dot: .red, checks: BrainBarMenuPresenter.known.map {
            .init(name: $0, state: ["drive_auth", "disk"].contains($0) ? .red : .green,
                  note: "Invented cause")
        }, notes: [], evaluatedAt: c.now)
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 2)
        XCTAssertEqual(center.posts.last?.1, "Backups need attention")
        XCTAssertNotEqual(center.posts.first?.1, center.posts.last?.1)
    }

    func testRound2CalendarIncidentAcrossSixKeysPostsOnce() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let keys = ["index", "backup-daily", "backup-snapshot", "jsonl-backup", "maintenance-light", "maintenance-full"]
        let alerts = Dictionary(uniqueKeysWithValues: keys.map { ($0, "Invented single abandonment") })
        await subject.update(capture(), jobAlerts: alerts)
        await subject.update(capture(), jobAlerts: alerts)
        XCTAssertEqual(center.posts.count, 1)
        // All alias keys must have receipts, not just the first sorted key.
        await subject.update(capture(), jobAlerts: ["maintenance-full": "Invented single abandonment"])
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound2ActivationBelongsOnlyToPythonAndPruneStaysSilent() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        await subject.update(capture(), jobAlerts: ["brainbar-activation": "Invented activation",
            "jsonl-retention": "Drive prune interrupted: invented interruption"])
        XCTAssertTrue(center.posts.isEmpty)
        XCTAssertEqual(center.requests, 0)
        await subject.update(capture(), jobAlerts: ["jsonl-retention": "Invented retention fault"])
        XCTAssertEqual(center.posts.count, 1)
    }

    func testRound2ChangedDriveConsentReasonNotifiesAgain() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        await subject.update(capture(), jobAlerts: ["drive-consent": "Invented consent expiry"])
        await subject.update(capture(), jobAlerts: ["drive-consent": "Invented consent revoked"])
        await subject.update(capture(), jobAlerts: ["drive-consent": "Invented consent revoked"])
        XCTAssertEqual(center.posts.map { $0.1 }, ["Invented consent expiry", "Invented consent revoked"])
    }

    func testRound2ThingFirstSuppressesPairedJob() async {
        for (key, check) in [("drive-consent", "drive_auth"), ("backup-daily", "backup_local"),
                             ("jsonl-backup", "transcripts"), ("maintenance-full", "app:maintenance")] {
            let center = Center()
            let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
            var c = capture(check == "app:maintenance" ? nil : check)
            if check == "app:maintenance" { c.maintenanceNightFailures = 3 }
            await subject.update(c)
            await subject.update(c, jobAlerts: [key: "Invented paired job"])
            // The job remains the same episode even if its menu row clears first.
            await subject.update(capture(), jobAlerts: [key: "Invented paired job"])
            XCTAssertEqual(center.posts.count, 1, key)
        }
    }

    func testRound2JobFirstSuppressesPairedThingButNotNewCause() async {
        for state: BrainBarHealthVerdict.State in [.red, .amber] {
        for (key, check) in [("drive-consent", "drive_auth"), ("backup-daily", "backup_local"),
                             ("jsonl-backup", "transcripts"), ("maintenance-full", "app:maintenance")] {
            let center = Center()
            let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
            let alerts = [key: "Invented paired job"]
            await subject.update(capture(), jobAlerts: alerts)
            var c = capture(check == "app:maintenance" ? nil : check, state: state)
            if check == "app:maintenance" { c.maintenanceNightFailures = 3 }
            await subject.update(c, jobAlerts: alerts)
            await subject.update(c, jobAlerts: [:])
            XCTAssertEqual(center.posts.count, 1, key)
            if check != "app:maintenance" {
                c.verdict = BrainBarHealthVerdict(dot: .red, checks: BrainBarMenuPresenter.known.map {
                    .init(name: $0, state: [$0 == check, $0 == "disk"].contains(true) ? .red : .green, note: "Invented cause")
                }, notes: [], evaluatedAt: c.now)
                await subject.update(c, jobAlerts: [:])
                XCTAssertEqual(center.posts.count, 2, key)
                if state == .red { XCTAssertEqual(center.posts.last?.1, "Backups need attention") }
                else { XCTAssertEqual(center.posts.last?.1, BrainBarMenuPresenter.notificationFaults(c).first?.title) }
            }
        }
        }
    }

    func testRound2CauseFlapUsesInjectedTenMinuteClockAcrossRelaunch() async {
        let clock = Box(), center = Center(), d = defaults()
        let subject = BrainBarFaultNotifications(center: center, defaults: d, clock: { clock.now })
        await subject.update(capture("disk"))
        clock.now = clock.now.addingTimeInterval(10)
        await subject.update(capture("disk", state: .amber))
        await subject.update(capture("disk"))
        XCTAssertEqual(center.posts.count, 1)
        let next = Center()
        let relaunched = BrainBarFaultNotifications(center: next, defaults: d, clock: { clock.now })
        await relaunched.update(capture("disk", state: .amber))
        clock.now = clock.now.addingTimeInterval(589)
        await relaunched.update(capture("disk"))
        XCTAssertTrue(next.posts.isEmpty)
        clock.now = clock.now.addingTimeInterval(1)
        await relaunched.update(capture("disk"))
        XCTAssertEqual(next.posts.count, 1)
    }

    func testRound2UnchangedCapturesDoNotWriteDeliveryState() async {
        let name = "BrainBarFaultNotificationTests.writes.\(UUID())"
        let d = CountingDefaults(suiteName: name)!
        defer { d.removePersistentDomain(forName: name) }
        let subject = BrainBarFaultNotifications(center: Center(), defaults: d)
        await subject.update(capture("disk"))
        let writes = d.deliveredWrites
        XCTAssertGreaterThan(writes, 0)
        for _ in 0..<20 { await subject.update(capture("disk")) }
        XCTAssertEqual(d.deliveredWrites, writes)
    }

    func testRound2DeliveryAndAuthorizationErrorsLogOncePerMinute() async {
        for authorizationFailure in [true, false] {
            let center = Center(), clock = Box()
            if authorizationFailure { center.authorizationFailures = 100 } else { center.postFailures = 100 }
            let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), clock: { clock.now }, log: { clock.logs.append($0) })
            for _ in 0..<10 { await subject.update(capture("disk")) }
            XCTAssertEqual(clock.logs.count, 1)
            clock.now = clock.now.addingTimeInterval(60)
            await subject.update(capture("disk"))
            XCTAssertEqual(clock.logs.count, 2)
            if authorizationFailure { center.authorizationFailures = 0 } else { center.postFailures = 0 }
            await subject.update(capture("disk"))
            XCTAssertEqual(center.posts.count, 1)
        }
    }

    func testRound2InitialBannerUsesTopRowAndOffsetLagDoesNotHoldOtherThings() async {
        let clock = Box()
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults(), clock: { clock.now })
        var c = capture("drive_auth")
        c.verdict = BrainBarHealthVerdict(dot: .red, checks: BrainBarMenuPresenter.known.map {
            .init(name: $0, state: ["drive_auth", "disk"].contains($0) ? .red : .green, note: "Invented cause")
        }, notes: [], evaluatedAt: c.now)
        c.watcher?.alertReasons = ["offset_lag"]
        XCTAssertNil(c.lagEpisodeStart)
        await subject.update(c)
        XCTAssertEqual(center.posts.first?.1, BrainBarMenuPresenter.present(c).rows.first { $0.kind == .needsYou }?.strings.first)
        var recovered = capture()
        recovered.watcher?.alertReasons = ["offset_lag"]
        await subject.update(recovered)
        c.now = c.now.addingTimeInterval(3600)
        clock.now = clock.now.addingTimeInterval(3600)
        await subject.update(c)
        XCTAssertEqual(center.posts.count, 2)
    }

    func testCodeRabbitDiskThenBackupJobKeepsDistinctIncidents() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let c = capture("disk")
        await subject.update(c)
        await subject.update(c, jobAlerts: ["backup-daily": "Invented daily backup failure"])
        XCTAssertEqual(center.posts.count, 2)
        XCTAssertEqual(center.posts.last?.1, "Invented daily backup failure")
    }

    func testCodeRabbitBackupJobThenDiskKeepsDistinctIncidents() async {
        let center = Center()
        let subject = BrainBarFaultNotifications(center: center, defaults: defaults())
        let alerts = ["backup-daily": "Invented daily backup failure"]
        await subject.update(capture(), jobAlerts: alerts)
        await subject.update(capture("disk"), jobAlerts: alerts)
        XCTAssertEqual(center.posts.count, 2)
        XCTAssertEqual(center.posts.last?.1, "Backups need attention")
    }

    func testCodeRabbitCooldownStartsWhenSlowPostSucceedsIncludingAliases() async {
        let center = Center(), clock = Box(), d = defaults()
        center.holdPost = true
        let subject = BrainBarFaultNotifications(center: center, defaults: d, clock: { clock.now })
        let alerts = ["mcp-transport": "Invented shared incident", "wal-checkpoint": "Invented shared incident"]
        let pending = Task { await subject.update(capture(), jobAlerts: alerts) }
        while center.posting == nil { await Task.yield() }
        clock.now = clock.now.addingTimeInterval(601)
        center.holdPost = false
        center.posting?.resume()
        await pending.value
        XCTAssertEqual(center.posts.count, 1)
        let times = d.dictionary(forKey: "BrainBarNotificationRecentCauses.v1") as? [String: [String: Double]]
        for key in alerts.keys { XCTAssertEqual(times?["job:" + key]?["job:" + key], clock.now.timeIntervalSince1970) }
        await subject.update(capture(), jobAlerts: [:])
        clock.now = clock.now.addingTimeInterval(1)
        await subject.update(capture(), jobAlerts: alerts)
        XCTAssertEqual(center.posts.count, 1)
        clock.now = clock.now.addingTimeInterval(599)
        await subject.update(capture(), jobAlerts: alerts)
        XCTAssertEqual(center.posts.count, 2)
    }

}
