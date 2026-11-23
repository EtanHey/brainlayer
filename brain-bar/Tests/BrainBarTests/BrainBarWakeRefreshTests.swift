import XCTest
import AppKit
import SwiftUI
import Vision
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

    func testWakeWithoutVerdictShowsSleepSentenceInBothDesigns() throws {
        let clock = Clock()
        let model = model(clock)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        let capture = model.menuCapture(stats: nil)
        XCTAssertEqual(model.publishedStatus.dashboardStatus.headline, "Checking after sleep…")
        XCTAssertEqual(BrainBarStatusPopoverController.statusLineTitle(for: model.publishedStatus.verdict), "Checking after sleep…")
        XCTAssertEqual(BrainBarMenuPresenter.present(capture).headline, "Checking after sleep…")
        try assertNewDesignDraws(capture, state: "after-wake", containing: "Checking after sleep")
    }

    /// Read the raster produced by the actual Overview headline and menu rows, rather than the shared presenter.
    private func assertNewDesignDraws(_ capture: BrainBarMenuCapture, state: String, containing text: String,
                                     mark: BrainBarHealthVerdict.State? = nil,
                                     file: StaticString = #filePath, line: UInt = #line) throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let store = NDStore(fixture: NDInputs(capture: capture), page: .overview)
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(".build.noindex/brainbar-ui-lead-w226/renders")
            .appendingPathComponent(ProcessInfo.processInfo.environment["BRAINBAR_WAKE_PROOF_ARM"] ?? "test")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (name, appearance) in NDRenderHarness.appearances {
            let overview = try NDRenderHarness.host(
                NDHeadline(store: store, r1: store.r1).padding(16).background(ND.bg),
                size: NSSize(width: NDGrid.windowWidth, height: 180), appearance: appearance)
            try NDRenderHarness.write(overview, name: "\(state)-headline-\(name)", in: directory)
            try NDRenderHarness.window(store, name: "\(state)-overview-\(name)", appearance: appearance, in: directory)
            let menuName = "\(state)-menu-\(name)"
            try NDRenderHarness.menu(store, name: menuName, appearance: appearance, in: directory)
            let menu = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: directory.appendingPathComponent(menuName + ".png"))))
            let window = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: directory.appendingPathComponent("\(state)-overview-\(name).png"))))
            for (surface, bitmap) in [("Overview", mark == .red ? window : overview), ("menu", menu)] {
                let request = VNRecognizeTextRequest()
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = false
                try VNImageRequestHandler(cgImage: XCTUnwrap(bitmap.cgImage)).perform([request])
                let drawn = request.results?.compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ") ?? ""
                XCTAssertTrue(drawn.contains(text), "\(state) \(surface) \(name): \(drawn)", file: file, line: line)
                if state == "after-wake" || state == "wake-timeout" {
                    XCTAssertFalse(drawn.contains("All good"), "\(state) \(surface) \(name): \(drawn)", file: file, line: line)
                } else if state == "fresh-after-wake" {
                    XCTAssertFalse(drawn.contains("Checking after sleep"), file: file, line: line)
                }
                if let mark {
                    var coloredPixels = 0
                    for y in 0..<bitmap.pixelsHigh {
                        for x in 0..<bitmap.pixelsWide {
                            guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                            let red = color.redComponent, green = color.greenComponent, blue = color.blueComponent
                            if mark == .red ? (red > 0.6 && red > green * 1.5 && red > blue * 1.5)
                                : (red > 0.6 && green > 0.25 && green < red * 0.85 && blue < green * 0.8) {
                                coloredPixels += 1
                            }
                        }
                    }
                    XCTAssertGreaterThan(coloredPixels, 8, "\(state) \(surface) \(name) must draw its \(mark) mark", file: file, line: line)
                }
            }
            if let mark {
                let icon = try NDRenderHarness.bitmap(NSSize(width: 22, height: 22))
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: icon)
                NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
                    NDIcon.image(for: store.model, store.r1, appearance: NSAppearance(named: appearance)!)
                        .draw(in: NSRect(x: 0, y: 0, width: 22, height: 22))
                }
                NSGraphicsContext.restoreGraphicsState()
                try XCTUnwrap(icon.representation(using: .png, properties: [:]))
                    .write(to: directory.appendingPathComponent("\(state)-icon-\(name).png"))
                var markedPixels = 0
                for y in 0..<icon.pixelsHigh {
                    for x in 0..<icon.pixelsWide {
                        guard let color = icon.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                        if color.alphaComponent > 0.5, color.redComponent > 0.6,
                           color.redComponent > color.greenComponent * 1.2,
                           color.greenComponent > color.blueComponent * 1.2 {
                            markedPixels += 1
                        }
                    }
                }
                XCTAssertGreaterThan(markedPixels, 8, "\(state) \(name) must draw the non-green menu-bar mark (\(mark))", file: file, line: line)
            }
        }
    }

    func testNewDesignDrawsWakeTimeoutReasonAndAmberMark() throws {
        let clock = Clock()
        var deadlines: [@MainActor @Sendable () -> Void] = []
        let model = model(clock, scheduler: { _, action in deadlines.append(action) })
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        clock.date.addTimeInterval(361); clock.uptime += 361
        deadlines.last?()
        try assertNewDesignDraws(model.menuCapture(stats: nil), state: "wake-timeout",
                                containing: "No health reading from after wake", mark: .amber)
    }

    func testNewDesignDrawsFreshReadingAfterWake() throws {
        let clock = Clock()
        let model = model(clock)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        clock.date.addTimeInterval(2); clock.uptime += 2
        var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        badge.healthVerdict = .init(dot: .green, checks: BrainBarDesignFixture.calmChecks(), notes: [], evaluatedAt: clock.date)
        model.applyHealthBadge(badge, alerts: nil)
        let fresh = model.menuCapture(stats: nil)
        XCTAssertFalse(fresh.wakeCatchUpPending)
        var capture = BrainBarDesignFixture.capture(BrainBarDesignFixture.calmChecks(), age: 0)
        capture.now = fresh.now
        capture.verdict = fresh.verdict
        capture.wakeCatchUpPending = fresh.wakeCatchUpPending
        try assertNewDesignDraws(capture, state: "fresh-after-wake", containing: "All good")
    }

    func testNewDesignKeepsPreSleepFaultRedDuringWakeAndTimeout() throws {
        let clock = Clock()
        var deadlines: [@MainActor @Sendable () -> Void] = []
        let model = model(clock, scheduler: { _, action in deadlines.append(action) })
        var badge = BadgeStatePresentation(badgeOn: true, reason: "Fixture fault", activeCodes: [])
        badge.healthVerdict = .init(dot: .red, checks: BrainBarDesignFixture.inputs(.driveSignedOut).capture.verdict!.checks,
                                    notes: [], evaluatedAt: clock.date)
        model.applyHealthBadge(badge, alerts: nil)
        model.systemWillSleep()
        clock.date.addTimeInterval(600)
        model.systemDidWake()
        try assertNewDesignDraws(model.menuCapture(stats: nil), state: "red-after-wake", containing: "needs you", mark: .red)
        clock.date.addTimeInterval(361); clock.uptime += 361
        deadlines.last?()
        try assertNewDesignDraws(model.menuCapture(stats: nil), state: "red-wake-timeout", containing: "needs you", mark: .red)
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
