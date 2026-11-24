import XCTest
import SwiftUI
@testable import BrainBar

@MainActor final class BrainBarUpdatesTests: XCTestCase {
    func preferences() -> UserDefaults { UserDefaults(suiteName: "update-test-" + UUID().uuidString)! }

    func testSuspendedCheckCannotOverwriteInstallOrCancel() async throws {
        for outcome in ["2.1.0", "error", "tap"] {
            var continuation: CheckedContinuation<String, Error>?
            var launches = 0
            let state = FileManager.default.currentDirectoryPath + "/.build.noindex/w257/race-" + UUID().uuidString
            let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", stateRoot: state,
                defaults: preferences(), feed: { try await withCheckedThrowingContinuation { continuation = $0 } },
                launch: { _, _, _, _ in launches += 1 })
            model.offer = "2.0.0"; model.busy = { nil }
            let check = Task { await model.check() }
            while continuation == nil { await Task.yield() }
            model.install(); model.cancel()
            if outcome == "error" { continuation!.resume(throwing: URLError(.timedOut)) }
            else if outcome == "tap" { continuation!.resume(throwing: BrainBarReleaseFeed.FeedError.tapPending) }
            else { continuation!.resume(returning: outcome) }
            await check.value
            XCTAssertEqual(model.phase, .installing)
            XCTAssertEqual(model.detail, "Cancelling…")
            model.install()
            XCTAssertEqual(launches, 1)
        }
    }

    func testSpawnSubmitsIndependentJobAndPinsScript() throws {
        let root = FileManager.default.currentDirectoryPath + "/.build.noindex/w257/spawn-" + UUID().uuidString
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        let source = root + "/source.sh"
        try "#!/bin/bash\nexit 0\n".write(toFile: source, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: source)
        var submitted: [String] = []
        try BrainBarUpdates.spawn("2.0.0", "fixture-id", root + "/jobs.json", source, root: root) { submitted = $0 }
        XCTAssertEqual(submitted, ["bootstrap", "gui/" + String(getuid()), root + "/update-job-fixture-id.plist"])
        let data = try Data(contentsOf: URL(fileURLWithPath: submitted[2]))
        let job = try XCTUnwrap(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
        XCTAssertTrue((job["Label"] as? String)?.hasPrefix("com.brainlayer.brainbar-update.fixture-id.") == true)
        XCTAssertEqual(job["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(job["LaunchOnlyOnce"] as? Bool, true)
        XCTAssertEqual(job["ProgramArguments"] as? [String], ["/usr/bin/env",
            "BRAINBAR_UPDATE_JOBS=" + root + "/jobs.json", "/bin/bash", root + "/update-fixture-id.sh", "2.0.0", "fixture-id"])
        XCTAssertEqual(try String(contentsOfFile: root + "/update-fixture-id.sh", encoding: .utf8),
                       try String(contentsOfFile: source, encoding: .utf8))
        XCTAssertThrowsError(try BrainBarUpdates.spawn("2.0.0", "other-id", root + "/jobs.json", source, root: root) { _ in
            throw URLError(.cannotConnectToHost)
        })
    }

    func testFixtureWithRuntimeHasInertUpdater() async {
        let runtime = BrainBarRuntime()
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), runtime: runtime)
        XCTAssertFalse(store.updates === runtime.updates)
        store.preferences.removeObject(forKey: "update.offer")
        await store.updates.check(force: true)
        XCTAssertNil(store.updates.offer)
        XCTAssertNil(store.preferences.string(forKey: "update.offer"))
        let normal = NDStore(runtime: runtime, workspace: BrainBarRecordingWorkspace())
        XCTAssertTrue(normal.updates === runtime.updates)
    }

    func testOpenSettingsContentReportsRecoveryHeight() async throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let model = store.updates
        let host = NSHostingView(rootView: NDSettingsView(store: store))
        host.frame = NSRect(x: 0, y: 0, width: 480, height: 800)
        func settle() async {
            for _ in 0..<10 { host.layoutSubtreeIfNeeded(); try? await Task.sleep(for: .milliseconds(10)) }
        }
        await settle()
        func capture(_ name: String) throws {
            host.setFrameSize(NSSize(width: 480, height: host.fittingSize.height))
            host.layoutSubtreeIfNeeded()
            let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: rep)
            let directory = FileManager.default.currentDirectoryPath + "/.build.noindex/w257/refit-proof"
            try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            try XCTUnwrap(rep.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: directory + "/" + name + ".png"))
        }
        let idle = NDSettingsWindowController.contentSize(of: host).height
        try capture("settings-before")
        model.phase = .failed
        model.detail = String(repeating: "Recovery command needs attention. ", count: 12)
        await settle()
        XCTAssertGreaterThan(NDSettingsWindowController.contentSize(of: host).height, idle)
        try capture("settings-recovery")
        model.detail = ""
        await settle()
        XCTAssertEqual(NDSettingsWindowController.contentSize(of: host).height, idle, accuracy: 1)
    }

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
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), feed: { calls += 1; throw URLError(.timedOut) })
        await model.check()
        await model.tick()
        await model.tick()
        XCTAssertEqual(calls, 1)
    }

    func testOfferPersistsAndNeverDowngradesOnFeedFailure() async {
        let defaults = preferences()
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: defaults, feed: { "2.0.0" })
        await model.check()
        XCTAssertEqual(model.offer, "2.0.0")
        model.snoozed = true
        let next = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: defaults, feed: { throw URLError(.timedOut) })
        await next.check(force: true)
        XCTAssertEqual(next.offer, "2.0.0")
        XCTAssertFalse(next.snoozed)
        let installed = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "2.0.0", defaults: defaults, feed: { "2.0.0" })
        XCTAssertNil(installed.offer)
    }

    func testCoalescingAndSuccessfulCache() async {
        var calls = 0
        var clock = Date(timeIntervalSince1970: 1000)
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), now: { clock }, feed: {
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
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), feed: { "2.0.0" }, launch: { _, _, _, _ in launches += 1 })
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
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", stateRoot: state, defaults: preferences(), feed: { "2.0.0" }, launch: { _, _, _, _ in launches += 1 })
        model.busy = { nil }
        await model.check()
        model.install(); model.install()
        XCTAssertEqual(launches, 1)
        XCTAssertEqual(model.phase, .installing)
    }

    func testHelperWaitingRetriesOnlyAtNextScheduledCheck() async throws {
        var clock = Date(timeIntervalSince1970: 1000)
        var ids: [String] = []
        let state = FileManager.default.currentDirectoryPath + "/.build.noindex/brainbar-ui-lead-w245/wait-" + UUID().uuidString
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", stateRoot: state,
                                    defaults: preferences(), now: { clock }, feed: { "2.0.0" }, launch: { _, id, _, _ in
            ids.append(id)
            try "\(id)\nwaiting\nWill update when background jobs finish\n".write(toFile: state + "/update-state", atomically: true, encoding: .utf8)
        })
        model.busy = { nil }
        await model.check()
        model.install()
        await model.tick()
        XCTAssertEqual(model.phase, .waiting)
        for _ in 0..<30 { clock.addTimeInterval(2); await model.tick() }
        XCTAssertEqual(ids.count, 1)
        clock.addTimeInterval(6 * 3600)
        await model.tick()
        XCTAssertEqual(ids.count, 2)
        XCTAssertEqual(ids[0], ids[1])
    }

    func testMissingInstallerRefusesUpdate() async {
        var launches = 0
        let model = BrainBarUpdates(installed: "1.0.0", defaults: preferences(), feed: { "2.0.0" }, launch: { _, _, _, _ in launches += 1 })
        await model.check()
        model.install()
        XCTAssertFalse(model.installerAvailable)
        XCTAssertEqual(model.detail, "Update unavailable — reinstall brainlayer")
        XCTAssertEqual(launches, 0)
    }

    func testCalendarParentRunningSurvivesProjection() {
        let provider = BrainLayerLaunchdStatusProvider(
            commandRunner: { args in
                .init(terminationStatus: 0, output: args.last?.hasSuffix("/com.brainlayer.maintenance") == true ? "pid = 123\nruns = 1\n" : "")
            },
            runRecordDate: { _ in nil }, schedule: { _ in nil }, ownership: { .supervisor },
            calendarHistory: { .init() }, calendarSchedule: { nil }, calendarScheduleProblem: { nil },
            calendarLogsDirectory: { URL(fileURLWithPath: "/invented-calendar.noindex") }, calendarIsDisabled: { _ in false }
        )
        let observations = provider.sampleActivity()
        for job in [BrainLayerLaunchdJob.backupDaily, .jsonlBackup, .maintenanceNightly, .maintenanceWeekly] {
            XCTAssertEqual(observations[job]?.calendarParentRunning, true)
        }
    }

    func testNewReleaseSupersedesFailureOnAutomaticCheck() async {
        var clock = Date(timeIntervalSince1970: 1000)
        var version = "2.0.0"
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), now: { clock }, feed: { version })
        await model.check()
        model.phase = .failed; model.detail = "Could not start update"
        clock.addTimeInterval(6 * 3600); version = "2.1.0"
        await model.tick()
        XCTAssertEqual(model.offer, "2.1.0")
        XCTAssertEqual(model.phase, .idle)
    }

    func testTapLagMakesNoNewOffer() async {
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), feed: { throw BrainBarReleaseFeed.FeedError.tapPending })
        await model.check()
        XCTAssertNil(model.offer)
        XCTAssertEqual(model.detail, "Release is reaching Homebrew")
        XCTAssertEqual(model.phase, .idle)
    }

    func testManualResultReturnsToOneLineAtRest() async {
        var clock = Date(timeIntervalSince1970: 1000)
        let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), now: { clock }, feed: { "1.0.0" })
        await model.check(force: true)
        XCTAssertEqual(model.detail, "Up to date")
        clock.addTimeInterval(6)
        await model.tick()
        XCTAssertEqual(model.detail, "")
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
            let model = BrainBarUpdates(installerPath: "/fixture/update.sh", installed: "1.0.0", defaults: preferences(), feed: { "2.0.0" })
            model.phase = phase; model.offer = offer
            model.detail = phase == .failed ? "Step activate failed. Run: /opt/homebrew/bin/brainlayer brainbar activate --verify --timeout 90" : ""
            for (suffix, appearance) in NDRenderHarness.appearances {
                let view = BrainBarUpdateRow(updates: model).background(ND.bg).foregroundStyle(ND.t1)
                let rep = try NDRenderHarness.host(view, size: NSSize(width: 480, height: 140), appearance: appearance)
                XCTAssertGreaterThan(rep.pixelsWide, 0)
                if suffix == "dark", let data = rep.representation(using: .png, properties: [:]) { pixels.insert(data) }
                try NDRenderHarness.write(rep, name: "\(name)-\(suffix)", in: directory)
                if name == "offer" {
                    let header = BrainBarUpdateRow(updates: model, header: true).background(ND.bg).foregroundStyle(ND.t1)
                    try NDRenderHarness.write(NDRenderHarness.host(header, size: NSSize(width: 640, height: 64), appearance: appearance), name: "header-\(suffix)", in: directory)
                    let windowStore = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
                    windowStore.updates = model
                    _ = try NDRenderHarness.window(windowStore, name: "window-offer-\(suffix)", appearance: appearance, in: directory)
                }
                let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
                store.updates = model
                let settings = NDSettingsView(store: store)
                try NDRenderHarness.write(NDRenderHarness.host(settings, size: NSSize(width: 480, height: 540), appearance: appearance), name: "settings-\(name)-\(suffix)", in: directory)
                if name == "up-to-date" {
                    let before = NDSettingsView(store: store, showsUpdateControl: false)
                    try NDRenderHarness.write(NDRenderHarness.host(before, size: NSSize(width: 480, height: 540), appearance: appearance), name: "settings-before-\(suffix)", in: directory)
                }
            }
        }
        XCTAssertEqual(pixels.count, 5, "Every drawn state must differ")
    }
}
