import AppKit
import XCTest
@testable import BrainBar

/// Headless: an in-memory button, never a system status item, menu tracking or window.
@MainActor final class BrainBarBrainIconTests: XCTestCase {
    private final class Button: NSButton {
        var assignedImages: [NSImage] = []
        var didAssignImage: (() -> Void)?
        override var image: NSImage? { didSet { if let image { assignedImages.append(image) }; didAssignImage?() } }
    }
    private final class Item: BrainBarStatusItem {
        let length: CGFloat = NSStatusItem.variableLength
        var menu: NSMenu?
        let view = Button()
        var button: NSButton? { view }
    }
    private final class Host: BrainBarStatusItemHosting {
        let item = Item()
        func statusItem(withLength: CGFloat) -> any BrainBarStatusItem { item }
        func removeStatusItem(_ item: any BrainBarStatusItem) {}
    }
    private func defaults() -> UserDefaults {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let name = "BrainBarBrainIconTests." + UUID().uuidString
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }
    private func stats(active: Bool = false) -> DashboardStats {
        let agent = active ? [0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1] : Array(repeating: 0, count: 12)
        let watcher = active ? [3, 5, 4, 7, 6, 5, 8, 6, 5, 7, 6, 9] : Array(repeating: 0, count: 12)
        return DashboardStats(chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 0,
            enrichmentPercent: 50, enrichmentRatePerMinute: 1, databaseSizeBytes: 0,
            recentActivityBuckets: agent, recentAgentWriteBuckets: agent, recentWatcherWriteBuckets: watcher, recentEnrichmentBuckets: Array(repeating: 0, count: 12))
    }
    func testAbsentPreferenceUsesNewMenuAndBrainWithoutWritingAValue() throws {
        let d: UserDefaults = defaults()
        let host: Host = Host()
        let runtime: BrainBarRuntime = BrainBarRuntime(launchMode: .menuItemDaemon)
        XCTAssertNil(d.object(forKey: BrainBarNewMenuPreference.key))
        let controller: BrainBarStatusPopoverController = BrainBarStatusPopoverController(
            workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        XCTAssertEqual(try XCTUnwrap(host.item.view.assignedImages.first).size, NSSize(width: 22, height: 22))
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertTrue(BrainBarNewMenuPreference.isEnabled(in: d))
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 22, height: 22))
        XCTAssertFalse(controller.contextMenuForTesting.items.contains { $0.title == "Open Dashboard" })
        XCTAssertTrue(controller.contextMenuForTesting.items.contains { ($0.representedObject as? BrainBarMenuAction) == .settings })
        XCTAssertNil(d.object(forKey: BrainBarNewMenuPreference.key), "Reading the default must not migrate stored preferences")
    }

    func testExplicitFalseUsesPreviousMenuAndExactCharts() throws {
        try assertStoredPreferenceSelectsMenuAndIcon(false)
    }

    func testExplicitTrueUsesNewMenuAndBrain() throws {
        try assertStoredPreferenceSelectsMenuAndIcon(true)
    }

    private func assertStoredPreferenceSelectsMenuAndIcon(_ enabled: Bool) throws {
        let d: UserDefaults = defaults()
        d.set(enabled, forKey: BrainBarNewMenuPreference.key)
        let host: Host = Host()
        let runtime: BrainBarRuntime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let controller: BrainBarStatusPopoverController = BrainBarStatusPopoverController(
            workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
            alertsURL: URL(fileURLWithPath: "/tmp/w89-absent-" + UUID().uuidString))
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        if enabled {
            XCTAssertEqual(host.item.view.image?.size, NSSize(width: 22, height: 22))
            XCTAssertTrue(controller.contextMenuForTesting.items.contains { ($0.representedObject as? BrainBarMenuAction) == .settings })
            XCTAssertFalse(controller.contextMenuForTesting.items.contains { $0.title == "Open Dashboard" })
        } else {
            XCTAssertEqual(controller.contextMenuForTesting.items.map(\.title),
                ["Nothing needs attention", "Show log", "", "", "Open Dashboard", "Settings…", "", "Restart BrainBar", "", "Quit BrainBar"])
            let expected: NSImage = BrainBarStatusPopoverController.statusIconImage(
                stats: stats(), badgeOn: controller.statusDotForTesting, dot: controller.statusVerdictForTesting.dot)
            XCTAssertEqual(try bitmap(try XCTUnwrap(host.item.view.image)).representation(using: .png, properties: [:]),
                           try bitmap(expected).representation(using: .png, properties: [:]))
        }
        XCTAssertEqual(d.object(forKey: BrainBarNewMenuPreference.key) as? Bool, enabled)
    }

    func testSwitchChangesRealControllerImage() throws {
        let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 26, height: 14))
        XCTAssertEqual(host.item.view.toolTip, controller.statusVerdictForTesting.tooltip)
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 22, height: 22))
        let checking = BrainBarMenuLiveCapture.capture(status: .checking, verdictAwakeAgeS: nil,
            wakeCatchUpPending: false, driveAuth: nil, backups: .notRead, backupsNotice: nil,
            watcher: nil, watcherAwakeAgeS: nil, observations: nil, now: Date(), timeZone: .current)
        XCTAssertEqual(host.item.view.accessibilityLabel(), "BrainBar — " + BrainBarMenuPresenter.present(checking).headline)
        XCTAssertEqual(host.item.view.toolTip, BrainBarMenuPresenter.present(checking).headline)
        controller.setMenuTrackingForTesting(true)
        let frozen = host.item.view.image
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(active: true), state: .indexing)
        XCTAssertTrue(host.item.view.image === frozen)
        controller.setMenuTrackingForTesting(false)
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 26, height: 14))
        XCTAssertNil(host.item.view.accessibilityLabel())
        XCTAssertEqual(host.item.view.toolTip, controller.statusVerdictForTesting.tooltip)
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 22, height: 22))
        XCTAssertTrue(controller.contextMenuForTesting.items.contains { ($0.representedObject as? BrainBarMenuAction) == .settings })
        XCTAssertFalse(controller.contextMenuForTesting.items.contains { $0.title == "Open Dashboard" })
    }

    func testOffPathMatchesChartsForAllThreeProofStates() throws {
        let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        for enabled in [false, true, false] {
            d.set(enabled, forKey: BrainBarNewMenuPreference.key)
            for name in BrainBarMenuIconProof.names {
                let badged = name == "status-icon-badged"
                let value = stats(active: name == "status-icon-active")
                controller.setBadgeForTesting(.init(badgeOn: badged, reason: badged ? "Attention" : "", activeCodes: []),
                    alertsURL: URL(fileURLWithPath: "/tmp/w85-absent-" + UUID().uuidString))
                controller.renderStatusIconForTesting(stats: value, state: .idle)
                if !enabled {
                    let expected = BrainBarStatusPopoverController.statusIconImage(stats: value, badgeOn: badged,
                        dot: controller.statusVerdictForTesting.dot)
                    XCTAssertEqual(try bitmap(host.item.view.image!).representation(using: .png, properties: [:]),
                                   try bitmap(expected).representation(using: .png, properties: [:]), name)
                }
            }
        }
    }

    func testFixtureIconsAndHeadlessRenders() throws {
        _ = defaults()
        let output = ProcessInfo.processInfo.environment["BRAINBAR_ICON_RENDER_DIR"].map { URL(fileURLWithPath: $0) }
        if let output { try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
        let fixtures = try BrainBarMenuPresentationTests.loadCases().filter { ["S1", "S3", "S8"].contains($0["id"] as? String ?? "") }
        XCTAssertEqual(fixtures.count, 3)
        for fixture in fixtures {
            let id = fixture["id"] as! String
            let capture = try BrainBarMenuCases.capture(fixture["inputs"] as! [String: Any])
            let p = BrainBarMenuPresenter.present(capture)
            XCTAssertEqual(p.icon, BrainBarMenuPresentationTests.expected(fixture).icon)
            let image = BrainBarBrainIcon.image(for: p)
            XCTAssertEqual(image.size, NSSize(width: 22, height: 22))
            XCTAssertEqual(image.isTemplate, p.icon == .resting)
            XCTAssertEqual(image.accessibilityDescription, "BrainBar — " + p.headline)
            for dark in [false, true] {
                let pixels = try bitmap(image, tint: image.isTemplate ? (dark ? .white : .black) : nil)
                var red = 0, white = 0, ink = 0
                var firstInkRow = pixels.pixelsHigh, lastInkRow = -1
                for y in 0..<pixels.pixelsHigh {
                    for x in 0..<pixels.pixelsWide {
                        let c = try XCTUnwrap(pixels.colorAt(x: x, y: y)?.usingColorSpace(.sRGB))
                        if c.alphaComponent > 0 {
                            firstInkRow = min(firstInkRow, y); lastInkRow = max(lastInkRow, y)
                        }
                        if c.alphaComponent > 0.99 {
                            ink += 1
                            if isAlertRed(c) { red += 1 }
                            if c.redComponent > 0.99 && c.greenComponent > 0.99 && c.blueComponent > 0.99 { white += 1 }
                        }
                    }
                }
                // Centred 20-point glyph: ink rows 3...40 within the 44-pixel slot.
                XCTAssertEqual(firstInkRow, 3); XCTAssertEqual(lastInkRow, 40)
                XCTAssertGreaterThan(ink, 50)
                if p.icon == .marked { XCTAssertGreaterThan(red, 100); XCTAssertGreaterThan(white, 5) }
                else { XCTAssertEqual(red, 0); if dark { XCTAssertEqual(white, ink) } }
                if let output {
                    try XCTUnwrap(pixels.representation(using: .png, properties: [:]))
                        .write(to: output.appendingPathComponent(id + (dark ? "-icon-dark.png" : "-icon-light.png")))
                }
            }
        }
    }

    func testLaunchFirstImageUsesTheSelectedIcon() throws {
        for enabled in [false, true] {
            let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
            d.set(enabled, forKey: BrainBarNewMenuPreference.key)
            let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
                dashboardPanelController: nil, statusBar: host, menuDefaults: d)
            defer { controller.stop() }
            let first: NSImage = try XCTUnwrap(host.item.view.assignedImages.first)
            if enabled {
                XCTAssertEqual(first.size, NSSize(width: 22, height: 22))
                XCTAssertTrue(first.isTemplate)
                XCTAssertTrue(first.accessibilityDescription?.hasPrefix("BrainBar — ") == true)
            } else {
                let legacy: NSImage = try XCTUnwrap(NSImage(systemSymbolName: "brain", accessibilityDescription: "BrainBar"))
                XCTAssertEqual(first.size, legacy.size)
                XCTAssertEqual(try bitmap(first).representation(using: .png, properties: [:]),
                               try bitmap(legacy).representation(using: .png, properties: [:]))
                XCTAssertEqual(host.item.view.toolTip, "BrainBar")
            }
        }
    }

    func testRealModelRedrawFollowsMarkedRestingMarked() throws {
        let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let now: Date = BrainBarStatusSnapshotTests.now
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
            store: .init(loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
            refreshStatusOnLoad: false, now: { now },
            initialObservabilityResult: BrainBarDashboardFixture.healthyObservabilityResult,
            awakeClock: { 100.0 }, sleepHistory: BrainBarSleepHistory(uptime: { 100.0 }),
            statusDeadlineScheduler: { _, _ in })
        runtime.jobsModel = model
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        let alerts = URL(fileURLWithPath: "/tmp/w85b-absent-" + UUID().uuidString)
        for marked in [true, false, true] {
            let state: BrainBarHealthVerdict.State = marked ? .red : .green
            let checks: [BrainBarHealthVerdict.Check] = BrainBarMenuPresenter.known.map {
                .init(name: $0, state: $0 == "stores" ? state : .green, note: "fixture")
            }
            let verdict = BrainBarHealthVerdict(dot: state, checks: checks, notes: [], evaluatedAt: now)
            controller.setBadgeForTesting(.init(badgeOn: marked, reason: marked ? "fixture failure" : "",
                activeCodes: [], healthVerdict: verdict), alertsURL: alerts)
            let p: BrainBarMenuPresentation = BrainBarMenuPresenter.present(model.menuCapture(stats: stats()))
            XCTAssertEqual(p.icon, marked ? .marked : .resting, "The real model must drive both states")
            controller.renderStatusIconForTesting(stats: stats(), state: .idle)
            let image: NSImage = try XCTUnwrap(host.item.view.image)
            XCTAssertEqual(image.isTemplate, !marked)
            XCTAssertEqual(host.item.view.accessibilityLabel(), "BrainBar — " + p.headline)
            XCTAssertEqual(host.item.view.toolTip, p.headline)
            let rep: NSBitmapImageRep = try bitmap(image)
            var red: Int = 0
            for y in 0..<rep.pixelsHigh {
                for x in 0..<rep.pixelsWide {
                    if let color = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB), isAlertRed(color), color.alphaComponent > 0.99 { red += 1 }
                }
            }
            if marked { XCTAssertGreaterThan(red, 100) } else { XCTAssertEqual(red, 0) }
        }
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.toolTip, controller.statusVerdictForTesting.tooltip)
    }

    func testStatusPublishMarksIconBeforeAnyCollectorSample() async throws {
        let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let now: Date = BrainBarStatusSnapshotTests.now
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
            store: .init(loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
            refreshStatusOnLoad: false, now: { now },
            initialObservabilityResult: BrainBarDashboardFixture.healthyObservabilityResult,
            awakeClock: { 100.0 }, sleepHistory: BrainBarSleepHistory(uptime: { 100.0 }),
            statusDeadlineScheduler: { _, _ in })
        runtime.jobsModel = model
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { host.item.view.didAssignImage = nil; controller.stop() }
        XCTAssertNil(runtime.collector)
        XCTAssertTrue(try XCTUnwrap(host.item.view.image).isTemplate)
        let changed: XCTestExpectation = expectation(description: "Published red status marks the icon without collector stats")
        host.item.view.didAssignImage = { [weak button = host.item.view] in
            if button?.image?.isTemplate == false { changed.fulfill() }
        }
        let verdict = BrainBarHealthVerdict(dot: .red,
            checks: [.init(name: "stores", state: .red, note: "fixture failure")], notes: [], evaluatedAt: now)
        model.applyHealthBadge(.init(badgeOn: true, reason: "fixture failure", activeCodes: [], healthVerdict: verdict), alerts: nil)
        let presentation: BrainBarMenuPresentation = BrainBarMenuPresenter.present(model.menuCapture(stats: nil))
        XCTAssertEqual(presentation.icon, .marked)
        await fulfillment(of: [changed], timeout: 3)
        XCTAssertNil(runtime.collector)
        let image: NSImage = try XCTUnwrap(host.item.view.image)
        XCTAssertFalse(image.isTemplate)
        XCTAssertEqual(host.item.view.accessibilityLabel(), "BrainBar — " + presentation.headline)
        XCTAssertEqual(host.item.view.toolTip, presentation.headline)
        let red: NSColor = try XCTUnwrap(try bitmap(image).colorAt(x: 8, y: 22)?.usingColorSpace(.sRGB))
        XCTAssertTrue(isAlertRed(red))
    }

    func testRedToleranceRejectsDifferentRed() {
        XCTAssertTrue(isAlertRed(NSColor(srgbRed: 1, green: 59.0 / 255.0, blue: 48.0 / 255.0, alpha: 1)))
        XCTAssertFalse(isAlertRed(NSColor(srgbRed: 1, green: 0, blue: 0, alpha: 1)))
    }

    private func isAlertRed(_ color: NSColor) -> Bool {
        let tolerance: CGFloat = 6.0 / 255.0
        return abs(color.redComponent - 1) <= tolerance
            && abs(color.greenComponent - 59.0 / 255.0) <= tolerance
            && abs(color.blueComponent - 48.0 / 255.0) <= tolerance
    }

    private func bitmap(_ image: NSImage, tint: NSColor? = nil) throws -> NSBitmapImageRep {
        let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(image.size.width * 2), pixelsHigh: Int(image.size.height * 2),
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0))
        rep.size = image.size
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = try XCTUnwrap(NSGraphicsContext(bitmapImageRep: rep))
        let rect = NSRect(origin: .zero, size: image.size)
        image.draw(in: rect)
        if let tint { tint.setFill(); rect.fill(using: .sourceIn) }
        NSGraphicsContext.current?.flushGraphics()
        return rep
    }
}
