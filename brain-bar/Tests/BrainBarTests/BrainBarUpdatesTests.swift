import XCTest
@testable import BrainBar

@MainActor final class BrainBarUpdatesTests: XCTestCase {
    func preferences() -> UserDefaults { UserDefaults(suiteName: "update-test-" + UUID().uuidString)! }

    func testNumericVersions() {
        XCTAssertTrue(BrainBarReleaseFeed.newer("1.10.0", than: "1.9.9"))
        for version in ["1.9.9", "1.0.0", "1.10", "2.0.0-rc1"] {
            XCTAssertFalse(BrainBarReleaseFeed.newer(version, than: "1.9.9"))
        }
    }

    func testAmbiguousTapFailsClosed() throws {
        XCTAssertEqual(try BrainBarReleaseFeed.tapVersion("cask \"brainbar\" do\n version \"2.0.0\"\nend", cask: true), "2.0.0")
        for text in ["404", "cask \"other\" do\n version \"2.0.0\"\nend",
                     "cask \"brainbar\" do\n version \"2.0.0\"\nversion :latest\nend",
                     "cask \"brainbar\" do\n on_arm do\nversion \"2.0.0\"\nend\nend"] {
            XCTAssertThrowsError(try BrainBarReleaseFeed.tapVersion(text, cask: true))
        }
    }

    func testFailedAutomaticCheckUsesSlowTimer() async {
        var calls = 0
        let model = BrainBarUpdates(installed: "1.0.0", defaults: preferences(), feed: { calls += 1; throw URLError(.timedOut) })
        await model.check()
        await model.tick()
        await model.tick()
        XCTAssertEqual(calls, 1)
    }

    func testOfferPersistsAndNeverDowngradesOnFeedFailure() async {
        let defaults = preferences()
        let model = BrainBarUpdates(installed: "1.0.0", defaults: defaults, feed: { "2.0.0" })
        await model.check()
        XCTAssertEqual(model.offer, "2.0.0")
        model.snoozed = true
        let next = BrainBarUpdates(installed: "1.0.0", defaults: defaults, feed: { throw URLError(.timedOut) })
        await next.check(force: true)
        XCTAssertEqual(next.offer, "2.0.0")
        XCTAssertFalse(next.snoozed)
        let installed = BrainBarUpdates(installed: "2.0.0", defaults: defaults, feed: { "2.0.0" })
        XCTAssertNil(installed.offer)
    }

    func testCoalescingAndSuccessfulCache() async {
        var calls = 0
        var clock = Date(timeIntervalSince1970: 1000)
        let model = BrainBarUpdates(installed: "1.0.0", defaults: preferences(), now: { clock }, feed: {
            calls += 1
            await Task.yield()
            return "1.0.0"
        })
        async let a: Void = model.check()
        async let b: Void = model.check()
        _ = await (a, b)
        await model.check()
        XCTAssertEqual(calls, 1)
        clock.addTimeInterval(6 * 3600)
        await model.check()
        XCTAssertEqual(calls, 2)
        XCTAssertNil(model.offer)
    }

    func testBusyWaitAndCancellationDoNotLaunch() async {
        var launches = 0
        let model = BrainBarUpdates(installed: "1.0.0", defaults: preferences(), feed: { "2.0.0" }, launch: { _, _, _ in launches += 1 })
        model.busy = { "backup" }
        await model.check()
        model.install()
        XCTAssertEqual(model.phase, .waiting)
        XCTAssertEqual(launches, 0)
        model.cancel()
        XCTAssertEqual(model.phase, .idle)
    }

    func testConcurrentClicksLaunchOneHelper() async throws {
        var launches = 0
        let state = FileManager.default.currentDirectoryPath + "/.build.noindex/brainbar-ui-lead-w245/clicks-" + UUID().uuidString
        let model = BrainBarUpdates(installed: "1.0.0", stateRoot: state, defaults: preferences(), feed: { "2.0.0" }, launch: { _, _, _ in launches += 1 })
        model.busy = { nil }
        await model.check()
        model.install(); model.install()
        XCTAssertEqual(launches, 1)
        XCTAssertEqual(model.phase, .installing)
    }

    func testRealFeedReadOnlyWhenExplicitlyAllowed() async throws {
        guard ProcessInfo.processInfo.environment["BRAINBAR_UPDATE_NETWORK_PROBE"] == "1" else { throw XCTSkip("read-only network opt-in") }
        let version = try await BrainBarReleaseFeed.latest()
        XCTAssertNotNil(BrainBarReleaseFeed.components(version))
        print("BRAINBAR_UPDATE_FEED latest release matches formula and cask: \(version)")
    }

    func testDrawnUpdateStates() throws {
        let path = ProcessInfo.processInfo.environment["BRAINBAR_UPDATE_PROOF_DIR"]
        let directory = URL(fileURLWithPath: path ?? FileManager.default.currentDirectoryPath + "/.build.noindex/brainbar-ui-lead-w245/proof").appendingPathComponent("update-control")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var pixels = Set<Data>()
        for (name, phase, offer) in [("up-to-date", BrainBarUpdates.Phase.idle, nil), ("checking", .checking, nil),
                                     ("offer", .idle, "2.0.0"), ("installing", .installing, "2.0.0"), ("failed", .failed, "2.0.0")] {
            let model = BrainBarUpdates(installed: "1.0.0", defaults: preferences(), feed: { "2.0.0" })
            model.phase = phase; model.offer = offer
            model.detail = phase == .failed ? "Activation failed — previous app restored" : (phase == .idle && offer == nil ? "Up to date" : "")
            for (suffix, appearance) in NDRenderHarness.appearances {
                let view = BrainBarUpdateRow(updates: model).background(ND.bg).foregroundStyle(ND.t1)
                let rep = try NDRenderHarness.host(view, size: NSSize(width: 480, height: 88), appearance: appearance)
                XCTAssertGreaterThan(rep.pixelsWide, 0)
                if suffix == "dark", let data = rep.representation(using: .png, properties: [:]) { pixels.insert(data) }
                try NDRenderHarness.write(rep, name: "\(name)-\(suffix)", in: directory)
                if name == "offer" {
                    let header = BrainBarUpdateRow(updates: model, header: true).background(ND.bar).foregroundStyle(ND.t1)
                    try NDRenderHarness.write(NDRenderHarness.host(header, size: NSSize(width: 640, height: 64), appearance: appearance), name: "header-\(suffix)", in: directory)
                }
                let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
                store.updates = model
                let settings = NDSettingsView(store: store)
                try NDRenderHarness.write(NDRenderHarness.host(settings, size: NSSize(width: 480, height: 460), appearance: appearance), name: "settings-\(name)-\(suffix)", in: directory)
                if name == "up-to-date" {
                    let before = NDSettingsView(store: store, showsUpdateControl: false)
                    try NDRenderHarness.write(NDRenderHarness.host(before, size: NSSize(width: 480, height: 460), appearance: appearance), name: "settings-before-\(suffix)", in: directory)
                }
            }
        }
        XCTAssertEqual(pixels.count, 5, "Every drawn state must differ")
    }
}
