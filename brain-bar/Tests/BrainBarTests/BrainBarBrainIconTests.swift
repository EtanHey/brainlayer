import AppKit
import XCTest
@testable import BrainBar

/// Headless: an in-memory button, never a system status item, menu tracking or window.
@MainActor final class BrainBarBrainIconTests: XCTestCase {
    private final class Item: BrainBarStatusItem {
        let length: CGFloat = NSStatusItem.variableLength
        var menu: NSMenu?
        let view = NSButton()
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
    func testSwitchChangesRealControllerImage() throws {
        let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 26, height: 14))
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 22, height: 22))
        let checking = BrainBarMenuLiveCapture.capture(status: .checking, verdictAwakeAgeS: nil,
            wakeCatchUpPending: false, driveAuth: nil, backups: .notRead, backupsNotice: nil,
            watcher: nil, watcherAwakeAgeS: nil, observations: nil, now: Date(), timeZone: .current)
        XCTAssertEqual(host.item.view.accessibilityLabel(), "BrainBar — " + BrainBarMenuPresenter.present(checking).headline)
        controller.setMenuTrackingForTesting(true)
        let frozen = host.item.view.image
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(active: true), state: .indexing)
        XCTAssertTrue(host.item.view.image === frozen)
        controller.setMenuTrackingForTesting(false)
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 26, height: 14))
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertEqual(host.item.view.image?.size, NSSize(width: 22, height: 22))

    }

    func testOffPathMatchesChartsForAllThreeProofStates() throws {
        let d = defaults(), host = Host(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        for enabled in [false, true, false] {
            if enabled { d.set(true, forKey: BrainBarNewMenuPreference.key) }
            else { d.removeObject(forKey: BrainBarNewMenuPreference.key) }
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
                            if c.redComponent > 0.99 && abs(c.greenComponent - 59.0 / 255) < 1.01 / 255 && abs(c.blueComponent - 48.0 / 255) < 1.01 / 255 { red += 1 }
                            if c.redComponent > 0.99 && c.greenComponent > 0.99 && c.blueComponent > 0.99 { white += 1 }
                        }
                    }
                }
                // A1's 2x reference has ink rows 1...38 in its 44-pixel slot.
                XCTAssertEqual(firstInkRow, 1); XCTAssertEqual(lastInkRow, 38)
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
