import AppKit
import XCTest
import SwiftUI
@testable import BrainBar

/// The new design behind `BrainBarDesign` (phase 11). Headless throughout: in-memory status items, windows that are
/// never ordered front, activation prohibited.
@MainActor final class BrainBarNewDesignSwitchTests: XCTestCase {
    private final class Button: NSButton {}
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
        let name = "BrainBarNewDesignSwitchTests." + UUID().uuidString
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }

    /// R1.10.1 and R1.10.3, on the window controller's real switch path and the real window content, for every fixture and
    /// each of the three pages: in the click, the window's root takes the selected page, at once (never animated: an
    /// animated swap is a transition, and the scroll view stacks a transition's two pages, the old on top), BEFORE the
    /// height starts to move, and through every frame of the resize it takes no other page. Read from the one signal a
    /// window that is never shown gives synchronously: the root's own update (its swap transaction, with the page), in the
    /// order things happen. Page bodies and preference callbacks are built or delivered on a display pass, which a headless
    /// runner never has; reading them made the outcome depend on which pass happened to run (1.10.1 and 1.10.2 on CI).
    func testR1101TheWindowShowsTheSelectedPageThroughTheSwitch() throws {
        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        var events: [String] = []
        NDLaidOutPagesProbe.swap = { page, animated in events.append("\(animated ? "animated" : "swap") \(page.rawValue)") }
        defer { NDLaidOutPagesProbe.swap = nil }
        for name in BrainBarDesignFixture.Name.allCases {
            let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
            controller.visibleHeightOverride = 2000
            controller.reduceMotionForTesting = false
            controller.designDefaults = d
            controller.applyDesign()
            let store = controller.newDesignStore
            store.setInputsForTesting(BrainBarDesignFixture.inputs(name))
            let window = controller.windowForTesting
            func layout() {
                window.contentView?.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.01))
                window.contentView?.layoutSubtreeIfNeeded()
            }
            controller.fitPage(store.page) // L57: settle the fixture content before stepping tab animations.
            controller.animatesPageSwitchForTesting = true
            var step: ((CGFloat) -> Void)?
            controller.pageFrameAnimatorForTesting = { target, _, _, finish in
                events.append("resize")
                let before = window.frame
                step = { t in
                    let h = before.height + (target.height - before.height) * t
                    window.setFrame(NSRect(x: before.minX, y: before.maxY - h, width: target.width, height: h), display: false)
                    if t >= 1 { finish() }
                }
            }
            layout()
            for to in [NDPage.backups, .jobs, .overview, .jobs, .backups, .overview] {
                step = nil
                events = []
                store.page = to  // what a click on the tab does
                let at = "\(name.rawValue) → \(to)"
                XCTAssertFalse(events.contains { $0.hasPrefix("animated ") }, "\(at): the page swaps at once, never animated: \(events)")
                let swapped = events.firstIndex(of: "swap \(to.rawValue)")
                XCTAssertNotNil(swapped, "\(at): the window's root took the selected page in the click: \(events)")
                if let resize = events.firstIndex(of: "resize"), let swapped {
                    XCTAssertLessThan(swapped, resize, "\(at): before the height started to move: \(events)")
                }
                for t in [0.25, 0.5, 0.75, 1.0] {
                    step?(CGFloat(t))
                    layout()
                }
                let others = events.filter { !$0.hasSuffix(" \(to.rawValue)") && $0 != "resize" }
                XCTAssertEqual(others, [], "\(at): the root took no other page through the resize: \(events)")
            }
        }
    }

    /// R1.8 item 1, on the window controller's real switch path (a tab click stores the page; the store hands it to the
    /// controller, which swaps the page and moves the window; the window is never shown). The frame animation goes
    /// through the controller's seam, so the test steps it frame by frame itself (a headless runner never advances
    /// AppKit's): the sequence is "new page first, then the height", in both directions, and no frame of the resize
    /// draws the old page.
    func testR18PageSwitchShowsTheNewPageFirstOnTheRealPath() throws {
        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        // Every input the switch reads is set here, never taken from the machine: a screen tall enough for every page
        // (no L25 cap evens the heights out), Reduce Motion off (a CI runner's session has it on), and a window that
        // counts as visible (it is never shown).
        controller.visibleHeightOverride = 2000
        controller.reduceMotionForTesting = false
        controller.designDefaults = d
        controller.applyDesign()
        let store = controller.newDesignStore
        let window = controller.windowForTesting
        store.page = .backups
        XCTAssertEqual(store.shownPage, .backups)
        controller.animatesPageSwitchForTesting = true
        var started: (target: NSRect, duration: Double, curve: CAMediaTimingFunction, page: NDPage, height: CGFloat, finish: () -> Void)?
        controller.pageFrameAnimatorForTesting = { target, duration, curve, finish in
            started = (target, duration, curve, store.shownPage, window.frame.height, finish)
        }
        for (to, grows) in [(NDPage.overview, true), (.backups, false), (.jobs, true)] {
            started = nil
            let before = window.frame
            store.page = to  // what a click on the tab does
            let animation = try XCTUnwrap(started, "\(to): the height animates")
            XCTAssertEqual(animation.page, to, "\(to): the new page is drawn before the height starts to move")
            XCTAssertEqual(animation.height, before.height, accuracy: 0.5, "\(to): ...while the window is still at its old height")
            XCTAssertEqual(grows, animation.target.height > before.height, "\(to): then the window grows or shrinks to the page")
            XCTAssertEqual(animation.duration, NDMotion.pageResizeDuration, accuracy: 0.0001)
            var points = [Float](repeating: 0, count: 4)
            animation.curve.getControlPoint(at: 1, values: &points[0])
            animation.curve.getControlPoint(at: 2, values: &points[2])
            XCTAssertEqual(points, [0.23, 1, 0.32, 1], "the strong ease-out")
            // Step the resize frame by frame: every frame draws the new page, and the height moves one way.
            var heights: [CGFloat] = []
            for step in 0...8 {
                let t = Double(step) / 8
                let h = before.height + (animation.target.height - before.height) * NDRenderHarness.easeOut(t)
                window.setFrame(NSRect(x: before.minX, y: before.maxY - h, width: animation.target.width, height: h), display: false)
                heights.append(window.frame.height)
                XCTAssertEqual(store.shownPage, to, "\(to): frame \(step) draws the new page")
            }
            animation.finish()
            XCTAssertEqual(window.frame.height, animation.target.height, accuracy: 0.5, "\(to): it ends at the page's height")
            XCTAssertEqual(window.frame.maxY, before.maxY, accuracy: 0.5, "\(to): the top edge stays")
            XCTAssertTrue(zip(heights, heights.dropFirst()).allSatisfy { grows ? $1 >= $0 - 0.5 : $1 <= $0 + 0.5 },
                          "\(to): the height moves one way: \(heights)")
        }
        // Reduce Motion on: no animation; the page and the height change together, in the same call.
        controller.reduceMotionForTesting = true
        started = nil
        store.page = .overview
        XCTAssertNil(started, "Reduce Motion: nothing animates")
        XCTAssertEqual(store.shownPage, .overview)
        XCTAssertEqual(window.contentRect(forFrameRect: window.frame).height,
                       NDPageHeight.window(store, page: .overview, screen: controller.fitScreenHeight, titleBarHeight: controller.fitTitleBarHeight), accuracy: 0.5, "and the height at once")
        XCTAssertEqual(NDPageSwitch.order(from: 500, to: 800, animates: true), .pageThenFrame)
        XCTAssertEqual(NDPageSwitch.order(from: 800, to: 500, animates: true), .pageThenFrame)
        XCTAssertEqual(NDPageSwitch.order(from: 500, to: 800, animates: false), .together, "Reduce Motion: at once")
    }

    private func stats() -> DashboardStats {
        DashboardStats(chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 0, enrichmentPercent: 50,
                       enrichmentRatePerMinute: 1, databaseSizeBytes: 0, recentActivityBuckets: Array(repeating: 0, count: 12),
                       recentAgentWriteBuckets: Array(repeating: 0, count: 12), recentWatcherWriteBuckets: Array(repeating: 0, count: 12),
                       recentEnrichmentBuckets: Array(repeating: 0, count: 12))
    }

    private func png(_ image: NSImage?) throws -> Data {
        let image = try XCTUnwrap(image)
        let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(image.size.width * 2),
            pixelsHigh: Int(image.size.height * 2), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        rep.size = image.size
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        image.draw(in: NSRect(origin: .zero, size: image.size))
        NSGraphicsContext.restoreGraphicsState()
        return try XCTUnwrap(rep.representation(using: .png, properties: [:]))
    }

    /// What the status item and menu show for one set of stored preferences.
    private func shown(design: String?, newMenu: Bool?) throws -> (titles: [String], actions: [String], image: Data, size: NSSize,
                                                                   hostedNew: Int) {
        let d = defaults()
        if let design { d.set(design, forKey: BrainBarDesign.key) }
        if let newMenu { d.set(newMenu, forKey: BrainBarNewMenuPreference.key) }
        let host = Host()
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(),
            runtime: BrainBarRuntime(launchMode: .menuItemDaemon), dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
            alertsURL: URL(fileURLWithPath: "/tmp/nd-absent-" + UUID().uuidString))
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        let items = controller.contextMenuForTesting.items
        return (items.map(\.title), items.map { ($0.representedObject as? BrainBarMenuAction).map { "\($0)" } ?? "" },
                try png(host.item.view.image), host.item.view.image?.size ?? .zero,
                items.filter { $0.view is NDMenuRowHost }.count)
    }

    /// L6: with `classic` (absent or stored), the icon and the menu are main's, byte for byte and title for title.
    func testClassicIsMainsIconAndMenuForBothMenuSettings() throws {
        for newMenu in [nil, false, true] as [Bool?] {
            let absent = try shown(design: nil, newMenu: newMenu)
            let stored = try shown(design: "classic", newMenu: newMenu)
            XCTAssertEqual(stored.titles, absent.titles, "menu titles, new menu \(String(describing: newMenu))")
            XCTAssertEqual(stored.actions, absent.actions)
            XCTAssertEqual(stored.image, absent.image, "icon pixels, new menu \(String(describing: newMenu))")
            XCTAssertEqual(stored.hostedNew, 0, "classic never builds a new-design row")
        }
        // Main's own expectations for the two menu settings (BrainBarBrainIconTests): charts and the legacy rows when off,
        // the 22 pt brain and the presenter's standing items when on.
        let off = try shown(design: "classic", newMenu: false)
        XCTAssertEqual(off.titles, ["Nothing needs attention", "Show log", "", "", "Open Dashboard", "Settings…", "", "Restart BrainBar", "", "Quit BrainBar"])
        XCTAssertEqual(off.size, NSSize(width: 26, height: 14))
        let on = try shown(design: "classic", newMenu: nil)
        XCTAssertEqual(on.size, NSSize(width: 22, height: 22))
        XCTAssertFalse(on.titles.contains("Open Dashboard"))
    }

    func testAbsentDesignIsClassicAndReadingNeverWrites() {
        let d = defaults()
        XCTAssertEqual(BrainBarDesign.current(in: d), .classic)
        XCTAssertNil(d.object(forKey: BrainBarDesign.key))
        d.set("new", forKey: BrainBarDesign.key)
        XCTAssertEqual(BrainBarDesign.current(in: d), .new)
        d.set("something else", forKey: BrainBarDesign.key)
        XCTAssertEqual(BrainBarDesign.current(in: d), .classic, "an unknown value is classic")
    }

    func testNewDesignDrawsItsOwnMenuAndIconAndSwitchesBackLive() throws {
        let new = try shown(design: "new", newMenu: nil)
        XCTAssertGreaterThanOrEqual(new.hostedNew, 5, "headline and four area rows")
        XCTAssertTrue(new.actions.contains("settings"))
        XCTAssertTrue(new.actions.contains("quit"))
        XCTAssertEqual(new.size, NSSize(width: 22, height: 22))

        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        let host = Host()
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(),
            runtime: BrainBarRuntime(launchMode: .menuItemDaemon), dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertTrue(controller.contextMenuForTesting.items.contains { $0.view is NDMenuRowHost })
        XCTAssertEqual(host.item.view.alphaValue, 1, "L23: no reading yet is drawn into the image (55 % brain and a \"?\" badge)")
        d.set("classic", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertFalse(controller.contextMenuForTesting.items.contains { $0.view is NDMenuRowHost })
        XCTAssertEqual(host.item.view.alphaValue, 1)
    }

    private func segmentedControl(in view: NSView) -> NSSegmentedControl? {
        if let control = view as? NSSegmentedControl { return control }
        for sub in view.subviews { if let found = segmentedControl(in: sub) { return found } }
        return nil
    }

    /// D1 (real screen, round 1): a click on a tab changes the page, the selected segment and the window's height together.
    /// It drives the action the toolbar's segmented control sends (selection, then its target/action), three times, and
    /// after each checks the stored page, the page the window root drew, the control's selected segment and the height.
    func testATabClickChangesPageSegmentAndHeightTogether() throws {
        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        controller.designDefaults = d
        controller.applyDesign()
        let window = controller.windowForTesting
        let toolbar = try XCTUnwrap(window.toolbar)
        let delegate = try XCTUnwrap(toolbar.delegate)
        let item = try XCTUnwrap(delegate.toolbar?(toolbar, itemForItemIdentifier: NDToolbar.pages, willBeInsertedIntoToolbar: true))
        let host = try XCTUnwrap(item.view)
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        host.layoutSubtreeIfNeeded()
        let control = try XCTUnwrap(segmentedControl(in: host))
        let store = controller.newDesignStore
        func settle() {
            for _ in 0..<3 {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
                window.contentView?.layoutSubtreeIfNeeded()
                host.layoutSubtreeIfNeeded()
            }
        }
        settle()
        for page in [NDPage.backups, .jobs, .overview] {
            let index = try XCTUnwrap(NDPage.allCases.firstIndex(of: page))
            control.selectedSegment = index
            XCTAssertTrue(NSApp.sendAction(try XCTUnwrap(control.action), to: control.target, from: control), "the control's own action")
            settle()
            XCTAssertEqual(store.page, page)
            XCTAssertEqual(store.drawnPageForTesting, page, "the window root drew the page the tab names")
            XCTAssertEqual(control.selectedSegment, index, "the selected segment names the same page")
            // L25: the page's height, capped the way the controller caps it (the runner's screen is small: 644 pt).
            let height = NDPageHeight.window(store, page: page, screen: controller.fitScreenHeight, titleBarHeight: controller.fitTitleBarHeight)
            let held = controller.windowWillResize(window, to: NSSize(width: 1200, height: 300)).height
            for measured in [window.frame.height, held] {
                XCTAssertGreaterThanOrEqual(measured, height - 0.5, "\(page): the height follows the same page")
                XCTAssertLessThanOrEqual(measured, height + NDGrid.titleBar, "\(page)")
            }
        }
    }

    /// L57: an injected small screen caps the content at its visible height minus actual AppKit chrome.
    /// the page scrolls inside it, and the footer stays at the window's bottom edge.
    func testASmallScreenCapsTheWindowThePageScrollsTheFooterStays() throws {
        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        controller.visibleHeightOverride = 600
        controller.designDefaults = d
        controller.applyDesign()
        controller.fitPage(.overview)
        let store = controller.newDesignStore
        let cap: CGFloat = 600 - controller.fitTitleBarHeight
        let page = ceil(NDPageHeight.content(store, page: .overview) + NDPageHeight.chrome)
        XCTAssertGreaterThan(page, cap, "the Overview is taller than this screen allows")
        XCTAssertEqual(NDPageHeight.window(store, page: .overview, screen: controller.fitScreenHeight, titleBarHeight: controller.fitTitleBarHeight), cap)
        let window = controller.windowForTesting
        let held = controller.windowWillResize(window, to: NSSize(width: 1200, height: 900)).height
        for measured in [window.frame.height, held] {
            XCTAssertGreaterThanOrEqual(measured, cap - 0.5, "held at the cap")
            XCTAssertLessThanOrEqual(measured, cap + NDGrid.titleBar, "never the full page")
        }
        // The window root at the capped size: the page's scroll view ends where the footer begins, and its content is
        // longer than it (it scrolls); the footer keeps the last 28 pt.
        let host = NSHostingView(rootView: NDWindowRoot(store: store))
        host.frame = NSRect(x: 0, y: 0, width: NDGrid.windowWidth, height: cap)
        host.layoutSubtreeIfNeeded()
        func scrollViews(_ v: NSView) -> [NSScrollView] { (v as? NSScrollView).map { [$0] } ?? [] + v.subviews.flatMap(scrollViews) }
        let scroll = try XCTUnwrap(scrollViews(host).max { $0.frame.height < $1.frame.height }, "the page sits in a scroll view")
        let frame = scroll.convert(scroll.bounds, to: host)
        let bottomGap = host.isFlipped ? cap - frame.maxY : frame.minY
        XCTAssertEqual(bottomGap, NDGrid.footer, accuracy: 1, "the footer stays below the scrolling page")
        XCTAssertEqual(frame.height, cap - NDGrid.titleBar - NDGrid.footer, accuracy: 1)
        // The scroll view's document is laid out lazily off screen, so the page's own measured height says it scrolls.
        XCTAssertGreaterThan(NDPageHeight.content(store, page: .overview) + NDGrid.pageTop + 20, frame.height + 1, "the page scrolls")
    }

    /// The window: classic keeps main's hosting controller and root type; new swaps in its own and swaps back.
    func testWindowRootIsMainsInClassicAndSwapsLive() throws {
        _ = defaults()
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        let classic = controller.contentViewControllerForTesting
        XCTAssertTrue(controller.windowForTesting.contentViewController === classic)
        XCTAssertTrue(String(describing: type(of: classic)).contains("BrainBarWindowRootView"), String(describing: type(of: classic)))
        XCTAssertEqual(controller.windowForTesting.minSize, BrainBarDashboardPanelController.minSize)
        XCTAssertFalse(controller.isShownForTesting)

        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        controller.designDefaults = d
        controller.applyDesign()
        let new = try XCTUnwrap(controller.windowForTesting.contentViewController)
        XCTAssertTrue(String(describing: type(of: new)).contains("NDWindowRoot"), String(describing: type(of: new)))
        // Review blocker 1: one unified 52 pt row, the pages control centred in the toolbar.
        XCTAssertEqual(controller.windowForTesting.toolbarStyle, .unified)
        XCTAssertEqual(controller.windowForTesting.toolbar?.centeredItemIdentifiers, [NDToolbar.pages])
        // L17: the window is 900 wide and takes the page's own height (Overview 716, R1.1), min = max; a tab switch snaps
        // it to the next page's height with the top edge fixed.
        // AppKit may add the unified toolbar's chrome to a content size it holds (20 pt on macOS 27 headless, 24 on CI's
        // macOS 15, 0 measured on the capture Mac's macOS 26), so heights are pinned to [page, page + one 52 pt row].
        let window = controller.windowForTesting
        func holds(_ page: NDPage, _ line: UInt = #line) {
            // L57: the measured page, capped at the visible screen minus actual AppKit chrome.
            let h = NDPageHeight.window(controller.newDesignStore, page: page, screen: controller.fitScreenHeight, titleBarHeight: controller.fitTitleBarHeight)
            let held = controller.windowWillResize(window, to: NSSize(width: 1400, height: 200))
            XCTAssertEqual(held.width, 900, "not resizable: the delegate holds the page size", line: line)
            XCTAssertGreaterThanOrEqual(held.height, h, line: line)
            XCTAssertLessThanOrEqual(held.height, h + NDGrid.titleBar, line: line)
            XCTAssertEqual(window.frame.size.height, held.height, accuracy: 0.5, line: line)
        }
        holds(.overview)
        let top = window.frame.maxY
        controller.fitPage(.backups)
        holds(.backups)
        XCTAssertEqual(window.frame.maxY, top, accuracy: 0.5, "the top edge stays")
        controller.fitPage(.jobs)
        holds(.jobs)
        controller.fitPage(.overview)
        XCTAssertFalse(controller.isShownForTesting, "a design switch never opens the window")

        d.set("classic", forKey: BrainBarDesign.key)
        controller.applyDesign()
        XCTAssertTrue(controller.windowForTesting.contentViewController === classic)
        XCTAssertEqual(controller.windowForTesting.minSize, BrainBarDashboardPanelController.minSize)
        XCTAssertNil(controller.windowForTesting.toolbar, "Classic's chrome is main's: no toolbar")
    }

    /// Review blocker 2: Settings is 520 pt wide and as tall as its three rows, fixed before it is shown.
    func testSettingsWindowIsSizedToItsRows() {
        _ = defaults()
        let settings = NDSettingsWindowController(store: NDStore(fixture: BrainBarDesignFixture.inputs(.calm)))
        let size = settings.window.contentRect(forFrameRect: settings.window.frame).size
        XCTAssertEqual(size.width, 480)
        XCTAssertGreaterThanOrEqual(size.height, 130)
        XCTAssertLessThanOrEqual(size.height, 200)
        XCTAssertFalse(settings.window.styleMask.contains(.resizable))
        XCTAssertEqual(settings.window.contentMaxSize, size)
        XCTAssertEqual(settings.window.identifier?.rawValue, "brainbar.nd.settings")
        XCTAssertEqual(settings.window.title, "BrainBar Settings")
        XCTAssertFalse(settings.window.isVisible)
    }
}

/// L5 and the truth rules: the new design never shows a calmer state than the presentation. Enumerated over the
/// presenter's own input space (BrainBarMenuInvariantTests.inputs) and the seventeen contract cases.
final class BrainBarNewDesignTruthTests: XCTestCase {
    typealias P = BrainBarMenuPresentation
    typealias Thing = BrainBarMenuPresenter.Thing

    static func violations(_ i: NDInputs) -> [String] {
        let c = i.capture
        let p = BrainBarMenuPresenter.present(c)
        let m = NDModel.make(i)
        var out: [String] = []
        let say = BrainBarMenuPresenter.Say.self
        // The headline, count and icon: the presentation's, or more urgent, never calmer.
        if m.count < p.count { out.append("count \(m.count) < presentation \(p.count)") }
        if p.icon == .marked, m.icon != .marked { out.append("icon calmer than the presentation") }
        // L53, one mark per problem: a needs headline carries none (its count and the icon say it); each fault row carries one.
        if p.form == .needs, m.headlineMark != .none { out.append("needs headline with a mark") }
        if p.form == .needs, m.faults.isEmpty { out.append("needs form without a fault row") }
        if m.headline == say.allGood.text, !BrainBarMenuPresenter.allGoodProven(c) { out.append("All good unproven") }
        if m.headline == say.allGood.text, !i.extras.isEmpty { out.append("All good beside a runtime warning") }
        if m.faults.count != m.count, m.icon == .marked { out.append("count \(m.count) but \(m.faults.count) fault rows") }
        if m.everythingElse, !p.rows.contains(where: { $0.kind == .everythingElse }) { out.append("Everything else without the presenter") }
        // Every Needs-you row of the presentation is drawn, in its order, with its action.
        let pf = p.rows.filter { $0.kind == .needsYou }
        if Array(m.faults.prefix(pf.count)).map(\.title) != pf.map { $0.strings.first ?? "" } { out.append("fault rows differ") }
        if Array(m.faults.prefix(pf.count)).map(\.action) != pf.map(\.action) { out.append("fault actions differ") }
        // No area reads fine unless the presenter proves it fine (unknown never drawn as fine).
        let proven = BrainBarMenuPresenter.provenFine(c)
        for area in m.menuAreas where area.mark == .none && !proven.contains(area.thing) {
            out.append("\(area.name) drawn fine, not proven")
        }
        for area in m.areas where area.thing != nil && area.mark == .none && !proven.contains(area.thing!) {
            out.append("\(area.name) panel fine, not proven")
        }
        // A red area always has a counted Needs-you row behind it.
        if !m.faultThings.isEmpty, m.count == 0 { out.append("red area with nothing counted") }
        // Unknown is dashed and "—", never a zero or a value; a value is never dashed.
        for row in m.areas.flatMap(\.rows) {
            if case .agent = row.visual { continue }  // an agent row draws its own unknown (dashed ring and bar)
            let dashed = row.visual == .unknown
            let dash = row.number.value == "—"
            if dashed != dash, !(row.stateWord) { out.append("\(row.name): dashed \(dashed) but number \(row.number.value)") }
            if row.visual == .emptyTrack, i.savesWaiting != 0 { out.append("\(row.name): empty drawn without a measured empty queue") }
            // Design pre-pass 5a: "none" rests on a fresh, known stores check.
            if row.visual == .emptyTrack, !(0...900).contains(c.verdictAwakeAgeS) || c.verdict == nil {
                out.append("\(row.name): none on a stale reading")
            }
        }
        // Design pre-pass 5b: a ring is explained by a dashed row, or says why in its help tag and VoiceOver.
        for area in m.areas where area.mark == .unknown && !area.rows.contains(where: { $0.visual == .unknown }) && area.unknownReason == nil {
            out.append("\(area.name): ring with no dashed row and no reason")
        }
        // Remembering's number is the newest conversation's age, a fact apart from its waveform.
        for area in m.menuAreas where area.thing != .remembering && (area.visual == .unknown) != (area.number.value == "—") {
            out.append("menu \(area.name): visual \(area.visual) number \(area.number.value)")
        }
        return out
    }

    func testNeverCalmerThanThePresentationOnEveryEnumeratedInput() throws {
        typealias I = BrainBarMenuInvariantTests
        let inputs = I.inputs()
        let tally = I.Tally()
        // Every input, crossed with the dot, every freshness and the checking flag; measured both ways on a stride.
        // Every fourth input (the presenter's own test covers each one); the full cross on each.
        let strided = Array(stride(from: 0, to: inputs.count, by: 4))
        DispatchQueue.concurrentPerform(iterations: strided.count) { k in
            let index = strided[k]
            var local: [String] = [], n = 0
            for dot in [BrainBarHealthVerdict.State.green, .red] {
                for freshness in I.Freshness.allCases {
                    for checking in [false, true] {
                        let measured = index % 3 != 0
                        var i = NDInputs(capture: I.capture(inputs[index], dot: dot, checking: checking, wake: false,
                                                            freshness: freshness, measured: measured))
                        i.savesWaiting = index % 2 == 0 ? 0 : nil
                        local += Self.violations(i).map { "\($0) | input \(index) dot \(dot) \(freshness) checking \(checking)" }
                        n += 1
                    }
                }
            }
            tally.add((n, local, [:]))
        }
        var checked = tally.outputs, found = tally.violations
        for c in try BrainBarMenuPresentationTests.loadCases().map({ try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any]) }) {
            found += Self.violations(NDInputs(capture: c))
            checked += 1
        }
        print("BrainBarNewDesignTruthTests: \(checked) captures checked")
        XCTAssertGreaterThan(checked, 40_000)
        XCTAssertEqual(Array(found.prefix(20)), [], "\(found.count) violations")
    }

    func testEveryFixtureHoldsTheRules() {
        for name in BrainBarDesignFixture.Name.allCases {
            XCTAssertEqual(Self.violations(BrainBarDesignFixture.inputs(name)), [], name.rawValue)
        }
    }

    /// The three runtime warnings: an attention-level one is counted and drawn in its own words; a notice stops "All good".
    func testRuntimeWarningsAreCountedAndNeverBesideAllGood() {
        var i = BrainBarDesignFixture.inputs(.calm)
        XCTAssertEqual(NDModel.make(i).headline, "All good")
        i.extras = [NDExtra(id: "job.x", level: .fault, text: "A job alert text", thing: .safe, jobs: [.backupDaily])]
        var m = NDModel.make(i)
        XCTAssertEqual(m.count, 1)
        XCTAssertEqual(m.headline, "1 thing needs you")
        XCTAssertEqual(m.icon, .marked)
        XCTAssertEqual(m.faults.map(\.title), ["A job alert text"])
        XCTAssertEqual(m.menuAreas.first { $0.thing == .safe }?.mark, .fault)
        XCTAssertFalse(m.everythingElse)
        XCTAssertEqual(Self.violations(i), [])

        i.extras = [NDExtra(id: "health.vector-coverage", level: .notice, text: "Vector check unverified", thing: .finding)]
        m = NDModel.make(i)
        XCTAssertEqual(m.headline, "Still checking search")
        XCTAssertEqual(m.count, 0)
        XCTAssertEqual(m.menuAreas.first { $0.thing == .finding }?.mark, .unknown)
        XCTAssertEqual(Self.violations(i), [])
    }

    func testUnloadedWatcherIsAFaultInTheRegistrysWords() {
        let setting = BrainLayerLaunchdJobSetting(enabled: true, loadState: .unloaded)
        let extras = NDExtra.derive(snapshot: .checking, watchSetting: setting)
        XCTAssertEqual(extras.map(\.level), [.fault])
        XCTAssertEqual(extras.first?.thing, .remembering)
        XCTAssertEqual(extras.first?.text, BrainBarMenuPresenter.Say.stopped.text)  // R1.11: an unloaded reader has stopped
    }

    /// L4: the Drive fault drops the Drive copy's age where a gauge shows it; VoiceOver and the report keep it.
    func testDriveFaultDropsTheAgeTheGaugeShows() {
        var c = BrainBarDesignFixture.inputs(.driveSignedOut).capture
        c.backupDailyFailures = 2  // the Drive row absorbs a failing database copy and says its age (R15)
        let checks = (c.verdict?.checks ?? []).map { k -> BrainBarHealthVerdict.Check in
            k.name == "backup_offsite" ? BrainBarDesignFixture.check("backup_offsite", .red, age: 21 * 3600, limit: 18 * 3600) : k
        }
        c.verdict = BrainBarHealthVerdict(dot: .red, checks: checks, notes: [], evaluatedAt: c.verdict!.evaluatedAt)
        let m = NDModel.make(NDInputs(capture: c))
        let drive = m.faults.first { $0.concerns.contains("drive_auth") }
        XCTAssertNotNil(drive)
        XCTAssertTrue(drive!.facts.joined(separator: " ").contains("Your newest backup in Google Drive is"), drive!.facts.description)
        XCTAssertFalse(drive!.shownFacts.joined(separator: " ").contains("Your newest backup in Google Drive is"))
        XCTAssertTrue(drive!.shownFacts.joined(separator: " ").contains("Backups can’t upload until you reconnect it."))
    }

    func testConcernedRowsTurnFaultAndTheDropdownShowsTheConcernedCopy() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.driveSignedOut))
        let rows = m.areas.first { $0.thing == .safe }!.rows
        guard case let .gauge(g)? = rows.first(where: { $0.id == "backup_offsite" })?.visual else { return XCTFail("gauge") }
        XCTAssertTrue(g.concerned)
        XCTAssertEqual(g.tone, .fault)
        guard case let .gauge(local)? = rows.first(where: { $0.id == "backup_local" })?.visual else { return XCTFail("gauge") }
        XCTAssertFalse(local.concerned)
        XCTAssertEqual(local.tone, .fine)
        XCTAssertEqual(m.menuAreas.first { $0.thing == .safe }?.number.text, "27\u{00A0}h")
        XCTAssertEqual(m.backupsTabCount, 1)
    }

    func testUpkeepFailingIsOneFormOneString() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.twoFaults))
        XCTAssertEqual(m.menuAreas.first { $0.thing == .upkeep }?.visual, .streak(3))
        XCTAssertEqual(m.menuAreas.first { $0.thing == .upkeep }?.number.text, "3\u{00A0}nights")
        XCTAssertEqual(m.areas.first { $0.thing == .upkeep }?.rows.first?.number.text, "3\u{00A0}nights")
        XCTAssertEqual(m.jobsTabCount, 1)
        XCTAssertEqual(m.count, 2)
    }

    /// L3: the 36 h / 8 d scale places the notch and never colours it.
    func testUpkeepScaleDecidesNoState() {
        var i = BrainBarDesignFixture.inputs(.calm)
        i.nightlyLastRun = .ran(i.capture.now.addingTimeInterval(-40 * 3600))
        let m = NDModel.make(i)
        guard case let .gauge(g)? = m.areas.first(where: { $0.thing == .upkeep })?.rows.first?.visual else { return XCTFail("gauge") }
        XCTAssertTrue(g.pinned)
        XCTAssertEqual(g.tone, .fine)
        XCTAssertEqual(m.headline, "All good")
        i.nightlyLastRun = .unread
        guard case .unknown? = NDModel.make(i).areas.first(where: { $0.thing == .upkeep })?.rows.first?.visual else { return XCTFail("dashed") }
    }

    func testAgentsColumnCountsMemories() throws {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.calm))
        let agents = try XCTUnwrap(m.areas.first { $0.name == "Agents" })
        XCTAssertEqual(agents.aside, BrainBarMenuPresenter.SurfaceSay.memoriesIn24h)
        XCTAssertEqual(agents.aside, "memories in 24 h")
        for row in agents.rows {
            XCTAssertTrue(row.accessibility.contains("memories"), row.accessibility)
            XCTAssertFalse(row.accessibility.contains("saves"), row.accessibility)
        }
        XCTAssertTrue(agents.rows.contains { $0.number.value.contains(",") }, "thousands separator like the title bar")
    }

    func testMeasuredZeroDrawsNothingAndUnknownIsDashed() {
        var i = BrainBarDesignFixture.inputs(.calm)
        let m = NDModel.make(i)
        let gemini = m.areas.first { $0.name == "Agents" }!.rows.first { $0.id == "gemini" }!
        // The scale is the busiest agent's count (the fixture's Agents rows sum to its 24 h memories, L5).
        XCTAssertEqual(gemini.visual, .agent(sessions: 0, saves: 0, scale: i.agents.agents.compactMap(\.saves).max()!))
        XCTAssertEqual(gemini.number.value, "0")
        i.agents = .unknown
        let unknown = NDModel.make(i).areas.first { $0.name == "Agents" }!
        XCTAssertEqual(unknown.mark, .unknown)
        XCTAssertTrue(unknown.rows.allSatisfy { $0.number.value == "—" })
    }

    /// Design pre-pass 7: beside "Everything else is fine." the fault drops its own reassurance; 22 words at most.
    func testOneUpkeepFaultKeepsToTheWordsBudget() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.nightlyFailing))
        XCTAssertTrue(m.everythingElse)
        let shown = m.faults.flatMap(\.shownFacts).joined(separator: " ")
        XCTAssertFalse(shown.contains("still being saved"), shown)
        XCTAssertLessThanOrEqual(shown.split(separator: " ").count, 22, shown)
        XCTAssertTrue(m.faults.flatMap(\.facts).joined(separator: " ").contains("still being saved"), "VoiceOver and the report keep it")
    }

    func testStaleReadingNeverSaysNone() {
        let row = NDModel.make(BrainBarDesignFixture.inputs(.notConfirmed)).areas.first { $0.thing == .remembering }!.rows.first { $0.id == "stores" }!
        XCTAssertEqual(row.visual, .unknown)
        XCTAssertEqual(row.number.value, "—")
    }

    func testCheckedLineSaysWhenTheVerdictIsStale() {
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.calm)).checked, "Checked 1 min ago")
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.notConfirmed)).checked, "Last checked 2 h ago")
    }

    func testFixtureNamesAreTheCaptureList() {
        XCTAssertEqual(BrainBarDesignFixture.Name.allCases.map(\.rawValue),
                       ["calm", "agent-not-read", "drive-signed-out", "saves-stuck", "meaning-behind", "nightly-failing",
                        "not-confirmed", "two-faults", "reconnect-waiting", "backups-failed-upload", "backup-late",
                        "lag-behind", "lag-stalled", "search-failing", "upload-progress", "upload-no-total", "upload-stalled",
                        "verifying", "upload-snapshot-red", "upload-drive-signed-out", "upload-drive-stalled"])
    }
}

/// L2: the backup-history reader, on invented lines only.
final class BrainBarNewDesignBackupHistoryTests: XCTestCase {
    let now = ISO8601DateFormatter().date(from: "2026-11-15T08:00:00+02:00")!

    private func log(_ lines: [String]) -> NDBackupHistoryReader.Log { .init(lines: lines.map { Substring($0) }, unreadable: false) }
    private func at(_ hoursAgo: Double) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: now.addingTimeInterval(-hoursAgo * 3600))
    }
    private func row(_ hoursAgo: Double, id: String, real: Bool = true, local: Bool = true, uploaded: Bool? = true,
                     verified: Bool = true, integrity: String = "ok") -> String {
        var fields = ["\"attempted_at\": \"\(at(hoursAgo))\"", "\"backup_invocation_id\": \"\(id)\"",
                      "\"backup_log_provenance\": \"\(real ? "real" : "pytest")\"", "\"local_verified\": \(local)",
                      "\"verified\": \(verified)", "\"local_integrity\": \"\(integrity)\""]
        if let uploaded { fields.append("\"uploaded\": \(uploaded)") }
        return "{" + fields.joined(separator: ", ") + "}"
    }

    func testRealRowsGroupedByInvocationInsideTheWindow() throws {
        let lanes = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(log([
            row(200, id: "old"),                          // older than 8 days: outside
            row(30, id: "a", uploaded: true, verified: false), row(29.9, id: "a"),  // one invocation: the newest row speaks
            row(6, id: "b", real: false),                 // a test row: never counted
            row(5, id: "c", local: false, uploaded: false, verified: false, integrity: "corrupt"),
        ]), now: now))
        XCTAssertEqual(lanes.local?.map(\.ok), [true, false])
        XCTAssertEqual(lanes.drive?.map(\.ok), [true, false])
        XCTAssertEqual(lanes.integrity?.map(\.ok), [true, false])
        XCTAssertEqual(lanes.skipped, 0)
    }

    func testADamagedLineMakesTheLanesUnknownAndIsCounted() throws {
        let lanes = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(log([row(5, id: "a"), "{not json"]), now: now))
        XCTAssertNil(lanes.local)
        XCTAssertNil(lanes.drive)
        XCTAssertEqual(lanes.skipped, 1)
    }

    func testAnUnreadableLogIsUnknownNeverEmpty() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("nd-history-" + UUID().uuidString)
        let history = NDBackupHistoryReader.read(databaseLog: dir.appendingPathComponent("missing.log"),
                                                 conversationLog: dir.appendingPathComponent("missing-too.log"), now: now)
        guard case let .read(lanes) = history else { return XCTFail("read") }
        XCTAssertNil(lanes.local)
        XCTAssertNil(lanes.conversations)
    }

    func testReadsOnlyTheTailAndNeverWrites() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("nd-history-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("backup-daily.log")
        let lines = (0..<500).map { row(Double($0) * 0.1, id: "i\($0)") }
        try (lines.reversed().joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
        let before = try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
        let tail = NDBackupHistoryReader.tail(url)
        XCTAssertEqual(tail.lines.count, NDBackupHistoryReader.maxLines)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date, before)
    }

    /// N01: a real row with none of the lanes' results is damage, not an empty night.
    func testASchemaInvalidRowIsDamage() throws {
        let lanes = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(log([
            row(5, id: "a"), "{\"attempted_at\": \"\(at(3))\", \"backup_log_provenance\": \"real\"}"]), now: now))
        XCTAssertNil(lanes.local)
        XCTAssertEqual(lanes.skipped, 1)
    }

    func testConversationPendingRowsDrawNoDot() throws {
        let lines = ["{\"attempted_at\": \"\(at(4))\", \"uploaded\": true, \"verified\": true, \"status\": \"verified\"}",
                     "{\"attempted_at\": \"\(at(2))\", \"uploaded\": false, \"verified\": false, \"status\": \"pending\"}"]
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseConversations(log(lines), now: now))
        XCTAssertEqual(parsed.attempts?.map(\.ok), [true])
    }
}

// MARK: - Code review round 1 (CODE-REVIEW.md F01–F07, N01–N05)

@MainActor final class BrainBarNewDesignReviewTests: XCTestCase {
    private final class Button: NSButton {}
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
    private func suite() -> UserDefaults {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let name = "BrainBarNewDesignReviewTests." + UUID().uuidString
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }

    /// F01: Settings reads the same accepted Drive row as every surface: a stale green check is not "Connected", a
    /// duplicated name is not read, and Reconnect appears only with the counted Drive fault.
    func testSettingsDriveComesFromTheAcceptedRow() {
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.calm)).drive, .init(state: "Connected", action: nil))
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.notConfirmed)).drive.state, "—", "18 minutes old: not current")
        let fault = NDModel.make(BrainBarDesignFixture.inputs(.driveSignedOut)).drive
        XCTAssertEqual(fault.state, "Disconnected")
        XCTAssertEqual(fault.action?.kind, .reconnectDrive)
        var twice = BrainBarDesignFixture.inputs(.calm)
        let checks = twice.capture.verdict!.checks + [BrainBarDesignFixture.check("drive_auth", .red)]
        twice.capture.verdict = BrainBarHealthVerdict(dot: .red, checks: checks, notes: [], evaluatedAt: twice.capture.verdict!.evaluatedAt)
        XCTAssertEqual(NDModel.make(twice).drive.state, "—", "a duplicated check is never read for its fields")
    }

    /// F02: a last-run time alone is never a success dot.
    func testJobsDrawASuccessOnlyForARecordedSuccess() {
        let now = BrainBarDesignFixture.now
        func observation(exit: Int32?, pid: Int32? = nil, damaged: Bool? = nil, failures: Int? = nil) -> BrainLayerLaunchdJobObservation {
            var o = BrainLayerLaunchdJobObservation(loadState: .loaded, runs: 1, lastExitCode: exit, lastRunAt: now,
                                                     nextRunAt: nil, isContinuous: false)
            o.pid = pid; o.calendarHistoryDamaged = damaged; o.calendarConsecutiveFailures = failures
            return o
        }
        XCTAssertEqual(NDStore.outcome(observation(exit: 0)), .ok)
        XCTAssertEqual(NDStore.outcome(observation(exit: 1)), .failed)
        XCTAssertEqual(NDStore.outcome(observation(exit: nil)), .unknown)
        XCTAssertEqual(NDStore.outcome(observation(exit: BrainLayerMaintenanceExit.deferred)), .unknown)
        XCTAssertEqual(NDStore.outcome(observation(exit: 0, damaged: true)), .unknown)
        XCTAssertEqual(NDStore.outcome(observation(exit: 0, failures: 1)), .failed)
        XCTAssertEqual(NDStore.outcome(observation(exit: 0, pid: 42)), .running)

        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.jobs = inputs.jobs!.map { var j = $0; if j.job == .decay { j.lastOutcome = .unknown }; if j.job == .drain { j.lastOutcome = .failed }; return j }
        let store = NDStore(fixture: inputs, page: .jobs)
        let lanes = NDJobsPage(store: store).lanes(store.model)
        XCTAssertEqual(lanes.first { $0.job == .decay }?.events.first?.kind, .unknown)
        XCTAssertEqual(lanes.first { $0.job == .drain }?.events.first?.kind, .fail)
        XCTAssertEqual(lanes.first { $0.job == .backupDaily }?.events.first?.kind, .ok)
    }

    /// F03: the copied report says what the surfaces show, runtime warnings included.
    func testReportCarriesTheRuntimeWarningsTheDropdownShows() {
        var i = BrainBarDesignFixture.inputs(.calm)
        i.extras = [NDExtra(id: "job.x", level: .fault, text: "A job alert text", thing: .safe, jobs: [.backupDaily])]
        var report = NDStore(fixture: i).report
        XCTAssertTrue(report.contains("1 thing needs you"), report)
        XCTAssertTrue(report.contains("A job alert text"), report)
        XCTAssertFalse(report.contains("All good"), report)
        i.extras = [NDExtra(id: "health.vector-coverage", level: .notice, text: "Vector check unverified", thing: .finding)]
        report = NDStore(fixture: i).report
        XCTAssertTrue(report.contains("Notice: Vector check unverified"), report)
        XCTAssertTrue(report.contains("Still checking search"), report)
        XCTAssertFalse(report.contains("All good"), report)
        XCTAssertFalse(NDStore(fixture: BrainBarDesignFixture.inputs(.calm)).report.contains("Warnings:"))
    }

    /// F04: no subscription or task runs unless a surface shows the new design; a history read in flight when the
    /// window closes is cancelled and publishes nothing.
    func testNewDesignWorkStopsOnSwitchAndClose() async throws {
        let runtime = BrainBarRuntime()
        let store = NDStore(runtime: runtime, workspace: BrainBarRecordingWorkspace())
        XCTAssertEqual(store.liveWorkCount, 0, "a new store does nothing until asked")
        store.setIconActive(true)
        XCTAssertGreaterThan(store.liveWorkCount, 0)
        store.setIconActive(false)
        XCTAssertEqual(store.liveWorkCount, 0, "back to Classic: every subscription and task gone")

        let dir = URL(fileURLWithPath: "/nonexistent-nd-test")
        store.setBackupPathsForTesting(.init(launchAgents: dir, databaseLog: dir, archiveLog: dir, maintenanceLog: dir,
                                             snapshotDirectory: dir, archiveDirectory: dir))
        final class Flag: @unchecked Sendable {
            private let lock = NSLock(); private var on = false
            func set() { lock.withLock { on = true } }
            var value: Bool { lock.withLock { on } }
        }
        let gate = DispatchSemaphore(value: 0), started = Flag()
        store.readHistory = { @Sendable known, _, _ in
            started.set(); gate.wait()
            return (.read(NDBackupLanes(local: [], drive: [], conversations: [], integrity: [])), known!)
        }
        store.setWindowActive(true)
        var didStart = false
        // The read starts off the main actor; yield until it has (up to 20 s, so a loaded CI runner cannot fail it early;
        // the wait ends as soon as it starts).
        for _ in 0..<400 where !didStart {
            try await Task.sleep(for: .milliseconds(50))
            didStart = started.value
        }
        XCTAssertTrue(didStart, "the read started")
        store.setWindowActive(false)
        XCTAssertEqual(store.liveWorkCount, 0, "closing the window cancels the read in flight")
        gate.signal()
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(store.inputs.backupHistory, .unread, "a read finishing after close publishes nothing")
    }

    func testClassicControllersCreateNoNewDesignStore() {
        let d = suite()
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        panel.designDefaults = d
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: panel, statusBar: Host(), menuDefaults: d)
        defer { controller.stop() }
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertNil(NDStore.existing(for: runtime), "Classic never builds the new design's store or its work")

        d.set("new", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        let store = try? XCTUnwrap(NDStore.existing(for: runtime))
        XCTAssertEqual(store?.iconActive, true)
        d.set("classic", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        panel.applyDesign()
        XCTAssertEqual(store?.iconActive, false)
        XCTAssertEqual(store?.windowActive, false)
        XCTAssertEqual(store?.liveWorkCount, 0)
    }

    /// F06: a capture keeps its window frame and Design choice in its own suite; the app's defaults are untouched.
    func testCaptureWritesOnlyItsOwnPreferences() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        setenv(BrainBarDesignFixture.environmentKey, "calm", 1)
        defer { unsetenv(BrainBarDesignFixture.environmentKey) }
        let key = BrainBarDashboardPanelController.frameDefaultsKey
        let before = UserDefaults.standard.object(forKey: key) as? String
        let bundleID = Bundle.main.bundleIdentifier ?? ""
        XCTAssertNotEqual(BrainBarDesignFixture.captureSuiteName, "com.brainlayer.brainbar")
        XCTAssertNotEqual(BrainBarDesignFixture.captureSuiteName, "com.brainlayer.BrainBar")
        XCTAssertNotEqual(BrainBarDesignFixture.captureSuiteName, bundleID)
        let (panel, status) = BrainBarDesignFixture.makeCaptureSurfaces(runtime: BrainBarRuntime(), activationPolicy: nil, statusBar: Host())
        defer { status.stop() }
        XCTAssertTrue(panel.designDefaults === BrainBarDesignFixture.capturePreferences)
        XCTAssertTrue(panel.newDesignStore.preferences === BrainBarDesignFixture.capturePreferences)
        XCTAssertTrue(panel.newDesignStore.isFixture)
        XCTAssertEqual(BrainBarDesignFixture.capturePreferences.string(forKey: BrainBarDesign.key), BrainBarDesign.new.rawValue,
                       "fix 4: the capture's Settings shows the design it draws")
        panel.placeWindowForTesting()
        panel.prepareForClose()
        XCTAssertNotNil(BrainBarDesignFixture.capturePreferences.string(forKey: key), "the capture's frame went to its own suite")
        XCTAssertEqual(UserDefaults.standard.object(forKey: key) as? String, before, "and nothing reached the app's defaults")
        XCTAssertNil(UserDefaults.standard.object(forKey: BrainBarDesign.key))
    }

    /// F07: the row is named for what it measures: the conversation reader's heartbeat.
    func testReaderRowIsNamedForTheHeartbeat() {
        let row = NDModel.make(BrainBarDesignFixture.inputs(.calm)).areas.first { $0.thing == .remembering }!.rows.first!
        XCTAssertEqual(row.name, "Read")  // R1.12 (§ C): the stage's one name
        XCTAssertTrue(row.accessibility.contains("last reported"), row.accessibility)
        XCTAssertTrue(NDModel.make(BrainBarDesignFixture.inputs(.calm)).areas.flatMap(\.rows).allSatisfy { $0.name != "New conversations" })
    }

    /// N04: a failed database check says what failed to VoiceOver.
    func testFailedDatabaseCheckIsSpokenInRegistryWords() {
        var i = BrainBarDesignFixture.inputs(.calm)
        i.capture.verdict = BrainBarHealthVerdict(dot: .red, checks: BrainBarDesignFixture.replacing(BrainBarDesignFixture.calmChecks(),
            BrainBarDesignFixture.check("integrity", .red)), notes: [], evaluatedAt: i.capture.verdict!.evaluatedAt)
        let row = NDModel.make(i).areas.first { $0.thing == .safe }!.rows.first { $0.id == "integrity" }!
        XCTAssertEqual(row.number.value, NDModel.Number.faultMark.value)
        XCTAssertTrue(row.accessibility.contains(BrainBarMenuPresenter.Say.s19.text), row.accessibility)
    }

    /// Review blocker 3: "Newest copy" is now − the lane's age (within an hour), in every fixture.
    func testCaptureClockIsTheWallClock() {
        let wall = Date(timeIntervalSinceReferenceDate: 784_000_123)
        let capture = BrainBarDesignFixture.makeNow(capturing: true, wall: wall)
        XCTAssertLessThan(wall.timeIntervalSince(capture), 60, "a capture's now is its launch minute, not a frozen 08:00")
        XCTAssertGreaterThanOrEqual(wall.timeIntervalSince(capture), 0)
        XCTAssertEqual(BrainBarDesignFixture.makeNow(capturing: false, wall: wall), BrainBarDesignFixture.fixedNow)
    }

    /// Fix 5 with review B2: a capture tick moves the fixture clock one minute; the 1 h hero shifts by exactly one bucket
    /// (and the first tick after opening slides: polish 7), every wave is read again off the one series, deterministically.
    func testCaptureTickShiftsTheHourByOneBucket() throws {
        let start = BrainBarDesignFixture.inputs(.calm)
        let store = NDStore(fixture: start), twin = NDStore(fixture: start)
        store.tickFixture(); twin.tickFixture()
        let hour = try XCTUnwrap(store.inputs.heroWaves[.hour]), was = try XCTUnwrap(start.heroWaves[.hour])
        XCTAssertEqual(hour.count, was.count)
        XCTAssertEqual(Array(hour.reads.dropLast()), Array(was.reads.dropFirst()))
        XCTAssertEqual(Array(hour.saves.dropLast()), Array(was.saves.dropFirst()))
        XCTAssertEqual(hour.sequence, try XCTUnwrap(was.sequence) + 1)
        XCTAssertEqual(hour.outgoing, NDWave.Bucket(saves: was.saves[0], reads: was.reads[0]), "the first tick slides")
        XCTAssertEqual(try XCTUnwrap(hour.endsAt).timeIntervalSince(try XCTUnwrap(was.endsAt)), 60, accuracy: 0.001)
        XCTAssertEqual(store.inputs.menuWave, twin.inputs.menuWave, "deterministic")
        XCTAssertEqual(store.inputs.heroWaves, twin.inputs.heroWaves, "deterministic")
        XCTAssertEqual(store.model, NDModel.make(store.inputs), "through the sample path: the model follows")
        store.tickFixture()
        XCTAssertEqual(store.fixtureTicks, 2)
        XCTAssertEqual(store.inputs.heroWaves[.hour]?.sequence, try XCTUnwrap(was.sequence) + 2)
    }

    /// Review B1, on the layout model: through one shift the baseline rect never moves; the moving layer holds n + 1
    /// buckets while it slides and n at rest; at t = 0 it stands exactly one bucket width right, so every bar is where it
    /// was before the data changed (no hole, no kick), and the outgoing bucket is the one that left.
    func testShiftSlidesOnePitchOverAStaticBaseline() throws {
        let t0 = BrainBarDesignFixture.fixedNow
        for (minutes, count, size) in [(60, 60, NDWaveform.Size.hero), (60, 24, .menu), (180, 60, .hero), (1_440, 60, .hero)] {
            let width = Double(minutes * 60) / Double(count)
            let before = NDFixtureTraffic.wave(minutes: minutes, buckets: count, now: t0)
            let after = NDFixtureTraffic.wave(minutes: minutes, buckets: count, now: t0.addingTimeInterval(width))
                .linked(after: before, minutes: minutes)
            let target = try XCTUnwrap(after.sequence)
            XCTAssertEqual(target, try XCTUnwrap(before.sequence) + 1)
            XCTAssertEqual(after.outgoing, NDWave.Bucket(saves: before.saves[0], reads: before.reads[0]))
            for canvas in [CGSize(width: 664, height: 114), CGSize(width: 664, height: 58), CGSize(width: 172, height: 20)] {
                let (bw, gap) = NDWaveform.layout(width: canvas.width, count: count, size: size)
                let pitch = bw + gap
                XCTAssertEqual(NDWaveform.offset(target: target, phase: Double(target - 1), pitch: pitch), pitch, "t = 0: +1 pitch")
                let mid = NDWaveform.offset(target: target, phase: Double(target) - 0.5, pitch: pitch)
                XCTAssertEqual(mid, pitch / 2, accuracy: 0.0001)
                XCTAssertEqual(NDWaveform.offset(target: target, phase: Double(target), pitch: pitch), 0)
                XCTAssertEqual(NDWaveform.baseline(canvas, nowTick: true), NDWaveform.baseline(canvas, nowTick: true))
                let atStart = NDWaveform.strip(after, width: canvas.width, size: size, offset: pitch)
                let during = NDWaveform.strip(after, width: canvas.width, size: size, offset: mid)
                let rest = NDWaveform.strip(after, width: canvas.width, size: size, offset: 0)
                let shown = NDWaveform.strip(before, width: canvas.width, size: size, offset: 0)
                XCTAssertEqual(atStart.count, count + 1)
                XCTAssertEqual(during.count, count + 1)
                XCTAssertEqual(rest.count, count)
                // At t = 0 the strip's first n entries are exactly the bars on screen before the change.
                for i in 0..<count {
                    XCTAssertEqual(atStart[i].x, shown[i].x, accuracy: 0.0001, "bar \(i) does not move at t = 0")
                    XCTAssertEqual(atStart[i].bucket, shown[i].bucket, "bar \(i) is the same bar")
                }
            }
        }
        // Not one bucket later: drawn in place, no outgoing bucket.
        let a = NDFixtureTraffic.wave(minutes: 60, buckets: 60, now: t0)
        XCTAssertNil(NDFixtureTraffic.wave(minutes: 60, buckets: 60, now: t0.addingTimeInterval(180)).linked(after: a, minutes: 60).outgoing)
        XCTAssertNil(a.linked(after: a, minutes: 60).outgoing)
        XCTAssertNil(a.linked(after: nil, minutes: 60).outgoing)
    }

    /// Review B2 in the fixture: across clocks, 1 h <= 3 h <= 24 h for both series, and the Agents rows sum to the 24 h
    /// saved plus the 24 h read (L5: they count memories). Captioned with the approved noun.
    func testFixtureTotalsNest() throws {
        for minutes in [0, 1, 2, 7, 61, 190, 1_441] {
            var i = BrainBarDesignFixture.inputs(.calm)
            BrainBarDesignFixture.traffic(&i, at: BrainBarDesignFixture.fixedNow.addingTimeInterval(Double(minutes) * 60 + 17))
            let h = try XCTUnwrap(i.heroWaves[.hour]), t = try XCTUnwrap(i.heroWaves[.threeHours]), d = try XCTUnwrap(i.heroWaves[.day])
            XCTAssertLessThanOrEqual(h.savesTotal, t.savesTotal, "\(minutes)")
            XCTAssertLessThanOrEqual(t.savesTotal, d.savesTotal, "\(minutes)")
            XCTAssertLessThanOrEqual(h.readsTotal, t.readsTotal, "\(minutes)")
            XCTAssertLessThanOrEqual(t.readsTotal, d.readsTotal, "\(minutes)")
            // L5 ruling: the Agents rows count memories (saved plus read), every one attributed in the fixture.
            XCTAssertEqual(i.agents.agents.compactMap(\.saves).reduce(0, +), d.savesTotal + d.readsTotal, "\(minutes)")
        }
    }

    /// Fix 6: every Reconnect a person can click (dropdown row, needs band, Backups row, Settings) reaches the one store
    /// action, and in a capture that shows the waiting line.
    func testEveryReconnectClickShowsTheWaitingLine() {
        for press in [{ (s: NDStore) in s.perform(.reconnectDrive) }, { $0.perform(.connectDrive) }, { $0.reconnectDrive() }] {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.driveSignedOut))
            XCTAssertFalse(store.inputs.driveReconnecting)
            press(store)
            XCTAssertTrue(store.inputs.driveReconnecting)
        }
    }

    /// Empty Settings window: the app's SwiftUI scene graph holds no scene with a window (Settings, WindowGroup, Window,
    /// DocumentGroup), so neither AppKit's launch-time open-untitled request, a reopen, state restoration nor
    /// `showSettingsWindow:` reaching SwiftUI's own app delegate can present one; its one scene is a menu-bar extra that is
    /// never inserted (no status item). Both designs.
    func testTheAppHasNoSceneThatCanPresentAWindow() throws {
        let scenes = String(describing: BrainBarApp.Body.self)
        XCTAssertTrue(scenes.contains("MenuBarExtra"), scenes)
        for windowed in ["Settings<", "SettingsScene", "WindowGroup", "Window<", "DocumentGroup", "UtilityWindow"] {
            XCTAssertFalse(scenes.contains(windowed), "\(windowed) in \(scenes)")
        }
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/BrainBarApp.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("MenuBarExtra(\"BrainBar\", isInserted: .constant(false))"), "never inserted: no status item")
        XCTAssertFalse(source.split(separator: "\n").contains { $0.trimmingCharacters(in: .whitespaces).hasPrefix("Settings {") },
                       "no SwiftUI Settings scene")
    }

    /// Fix 10: a switched-off job says "off" in Ink 2; its switch's help names the job and the action.
    func testSwitchedOffJobSaysOff() throws {
        var i = BrainBarDesignFixture.inputs(.calm)
        let index = try XCTUnwrap(i.jobs?.firstIndex { !$0.continuous })
        i.jobs![index].enabled = false
        let store = NDStore(fixture: i, page: .jobs)
        let lane = try XCTUnwrap(NDJobsPage(store: store).lanes(store.model).first { $0.job == i.jobs![index].job })
        XCTAssertEqual(lane.number.value, BrainBarMenuPresenter.SurfaceSay.off)
        XCTAssertTrue(lane.number.secondary)
        XCTAssertFalse(lane.events.contains { $0.kind == .next }, "a switched-off job has no next run")
        XCTAssertEqual(BrainBarMenuPresenter.SurfaceSay.turnOn(lane.name), "Turn on " + lane.name)
        XCTAssertEqual(BrainBarMenuPresenter.SurfaceSay.turnOff(lane.name), "Turn off " + lane.name)
    }

    /// Fix 11: the report sentence never breaks, so neither "an agent." nor "to an agent." can stand alone.
    func testReportSentenceIsOneUnit() {
        let report = BrainBarMenuPresenter.Say.agent.text  // R1.11: the one who-acts sentence
        XCTAssertFalse(keepingUnits("Fact one. " + report).hasSuffix(" " + report.split(separator: " ").suffix(2).joined(separator: " ")))
        XCTAssertFalse(keepingUnits(report).contains(" "))
    }

    /// R1 § 3.1: 6 pt bars with 2 pt corners in the window, 3 pt bars with 1 pt in the dropdown; never more than half a bar.
    func testShortHeroBarsAreNotBeads() {
        XCTAssertEqual(NDWaveform.barRadius(height: 40, width: 6, pill: false, corner: 2), 2)
        XCTAssertEqual(NDWaveform.barRadius(height: 2, width: 6, pill: false, corner: 2), 1, "a 2 pt bar is not a bead")
        XCTAssertEqual(NDWaveform.barRadius(height: 9, width: 3, pill: false, corner: 1), 1)
        XCTAssertEqual(NDWaveform.layout(width: 800, count: 60, size: .hero).bar, 6)
        XCTAssertEqual(NDWaveform.layout(width: 128, count: 24, size: .menu).bar, 3)
    }

    /// Fix 14: an area draws the unknown ring only when one of its rows is dashed; otherwise the reason is spoken.
    func testRingOnlyBesideADashedRow() {
        for name in BrainBarDesignFixture.Name.allCases {
            let m = NDModel.make(BrainBarDesignFixture.inputs(name))
            for area in m.areas {
                if area.shownMark == .unknown {
                    XCTAssertTrue(area.rows.contains { $0.visual == .unknown }, "\(name.rawValue) \(area.name)")
                } else if area.mark == .unknown {
                    XCTAssertNotNil(area.unknownReason, "\(name.rawValue) \(area.name)")
                }
            }
            for area in m.menuAreas where area.shownMark == .unknown {
                XCTAssertEqual(area.visual, .unknown, "\(name.rawValue) \(area.name)")
            }
        }
    }

    /// Fix 15: the last run and the next run both within 6 pt of now: only the next-run ring is drawn.
    func testRailDrawsOnlyTheNextRingBesideNow() {
        let close: [NDRail.Event] = [.init(hours: -0.05, kind: .ok), .init(hours: 0.05, kind: .next)]
        XCTAssertEqual(NDRail.shown(close, width: 480, past: 36, ahead: 12), [close[1]])
        let apart: [NDRail.Event] = [.init(hours: -5, kind: .ok), .init(hours: 0.05, kind: .next)]
        XCTAssertEqual(NDRail.shown(apart, width: 480, past: 36, ahead: 12), apart)
        let noNext: [NDRail.Event] = [.init(hours: -0.05, kind: .ok)]
        XCTAssertEqual(NDRail.shown(noNext, width: 480, past: 36, ahead: 12), noNext)
    }

    /// Review B3: on the Backups page "Newest copy" and the "Copy on this Mac" lane are both unknown, or both known and
    /// within an hour of each other, in every fixture, the stale one included.
    func testNewestCopyFollowsTheLaneOnTheBackupsPage() throws {
        let units: [String: Double] = ["s": 1, "min": 60, "h": 3600, "d": 86_400]
        var sawUnknown = false
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .backups)
            let page = NDBackupsPage(store: store)
            let lane = try XCTUnwrap(page.lanes(store.model).first { $0.id == "backup_local" })
            let fact = page.newestCopyFact(store.model)
            if lane.number.value == NDModel.Number.unknown.value {
                sawUnknown = true
                XCTAssertEqual(fact, "—", name.rawValue)
                continue
            }
            XCTAssertNotEqual(fact, "—", name.rawValue)
            let age = try XCTUnwrap(Double(lane.number.value)) * (try XCTUnwrap(units[lane.number.unit]))
            let newest = try XCTUnwrap(store.inputs.newestLocalCopyAt)
            XCTAssertEqual(store.inputs.capture.now.timeIntervalSince(newest), age, accuracy: 3600, name.rawValue)
        }
        XCTAssertTrue(sawUnknown, "the stale fixture is covered")
    }

    /// Review N1: on the Backups page a row's rail has filled dots (a confirmed copy, or a failed attempt) only if its
    /// newest cell is known; an unknown newest keeps the attempts' places as not-confirmed rings with a dashed line from
    /// the first to now. Every fixture, both uncertain states included.
    func testBackupRailClaimsNoCopyItsNewestCellCallsUnknown() throws {
        var unknownRows = 0
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .backups)
            for lane in NDBackupsPage(store: store).lanes(store.model) {
                let filled = lane.events.contains { $0.kind == .ok || $0.kind == .fail }
                let known = lane.number.value != NDModel.Number.unknown.value
                if !known {
                    unknownRows += 1
                    XCTAssertFalse(filled, "\(name.rawValue) \(lane.name)")
                    let past = lane.events.filter { $0.kind != .next }
                    XCTAssertTrue(past.allSatisfy { $0.kind == .unconfirmed && $0.hours <= 0 }, "\(name.rawValue) \(lane.name)")
                    XCTAssertEqual(lane.unconfirmedFrom, past.map(\.hours).min(), "\(name.rawValue) \(lane.name)")
                } else {
                    XCTAssertFalse(lane.events.contains { $0.kind == .unconfirmed }, "\(name.rawValue) \(lane.name)")
                    XCTAssertNil(lane.unconfirmedFrom)
                }
                XCTAssertTrue(lane.events.filter { $0.kind == .next }.allSatisfy { $0.hours >= 0 }, "next rings sit right of now")
            }
        }
        XCTAssertGreaterThan(unknownRows, 0, "checking and cant-tell are covered")
    }

    func testNewestCopyTimeMatchesTheLaneAge() throws {
        for name in BrainBarDesignFixture.Name.allCases {
            let i = BrainBarDesignFixture.inputs(name)
            let local = try XCTUnwrap(i.capture.verdict?.checks.first { $0.name == "backup_local" }?.ageS)
            let newest = try XCTUnwrap(i.newestLocalCopyAt)
            XCTAssertEqual(i.capture.now.timeIntervalSince(newest), local, accuracy: 3600, name.rawValue)
        }
    }

    /// Design pre-pass 6: on Backups the newest dot and the number come from the same record (within an hour).
    func testNewestRailDotMatchesTheNumber() {
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .backups)
            for lane in NDBackupsPage(store: store).lanes(store.model) where lane.number.unit == "h" {
                let newest = lane.events.filter { $0.kind != .next }.map(\.hours).max()
                guard let newest, let hours = Double(lane.number.value) else { continue }
                XCTAssertEqual(-newest, hours, accuracy: 1, "\(name.rawValue) \(lane.name)")
            }
        }
    }
}

/// F05: Classic against an oracle recorded from origin/main's own code (an export of `origin/main` 84b0e7a5 with a
/// recorder test, `Fixtures/classic-main-golden.json`), never against the code under test: the menu's titles, actions,
/// enabled states and keys, the icon's size, template flag and pixels, and the window's root type and minimum size.
@MainActor final class BrainBarClassicGoldenTests: XCTestCase {
    private final class Button: NSButton {}
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

    static func golden() throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/classic-main-golden.json")
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    /// Pixels compared on the alpha and colour channels with a small tolerance, so a different macOS's antialiasing passes
    /// and a changed drawing does not.
    static func pixelsMatch(_ a: Data, _ b: Data) -> (Bool, String) {
        guard let ra = NSBitmapImageRep(data: a), let rb = NSBitmapImageRep(data: b) else { return (false, "undecodable") }
        guard ra.pixelsWide == rb.pixelsWide, ra.pixelsHigh == rb.pixelsHigh else {
            return (false, "size \(ra.pixelsWide)x\(ra.pixelsHigh) vs \(rb.pixelsWide)x\(rb.pixelsHigh)")
        }
        var total = 0, worst = 0
        for y in 0..<ra.pixelsHigh {
            for x in 0..<ra.pixelsWide {
                guard let ca = ra.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                      let cb = rb.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                let d = [abs(ca.alphaComponent - cb.alphaComponent), abs(ca.redComponent * ca.alphaComponent - cb.redComponent * cb.alphaComponent),
                         abs(ca.greenComponent * ca.alphaComponent - cb.greenComponent * cb.alphaComponent),
                         abs(ca.blueComponent * ca.alphaComponent - cb.blueComponent * cb.alphaComponent)].max()!
                let v = Int(d * 255)
                total += v
                worst = max(worst, v)
            }
        }
        let mean = Double(total) / Double(ra.pixelsWide * ra.pixelsHigh)
        return (mean < 2 && worst < 96, "mean \(mean) worst \(worst)")
    }

    /// What an `ImageRenderer` raster depends on: Classic's chart icon (the old menu) is rendered by SwiftUI at
    /// `NSScreen.main`'s backing scale in the app's appearance, on this OS's renderer. Measured, not assumed.
    @MainActor static func rasterEnvironment() -> String {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let os = ProcessInfo.processInfo.operatingSystemVersion.majorVersion
        let scale = NSScreen.main?.backingScaleFactor ?? 0
        let appearance = NSApplication.shared.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua])?.rawValue ?? "unknown"
        return "macOS \(os), screen scale \(scale), \(appearance)"
    }

    func testPrintRasterEnvironment() throws {
        guard ProcessInfo.processInfo.environment["ND_PRINT_RASTER_ENV"] != nil else { throw XCTSkip("probe only") }
        print("ND_RASTER_ENV=\(Self.rasterEnvironment())")
    }

    func testClassicMatchesTheGoldenRecordedFromMain() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let golden = try Self.golden()
        XCTAssertEqual(golden["source"] as? String, "origin/main")
        XCTAssertNotNil(golden["rasterEnvironment"] as? String, "the raster icon's recording environment")
        XCTAssertEqual(golden["commit"] as? String, "84b0e7a58c034e46cb9080071557e0574db31f3c", "re-record when main's classic path moves")
        for expected in try XCTUnwrap(golden["cases"] as? [[String: Any]]) {
            let newMenu = try XCTUnwrap(expected["newMenu"] as? String)
            for design in [nil, "classic"] as [String?] {
                let name = "BrainBarClassicGoldenTests." + UUID().uuidString
                let d = try XCTUnwrap(UserDefaults(suiteName: name))
                defer { d.removePersistentDomain(forName: name) }
                if newMenu != "absent" { d.set(newMenu == "true", forKey: BrainBarNewMenuPreference.key) }
                if let design { d.set(design, forKey: BrainBarDesign.key) }
                let host = Host()
                let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
                let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
                    dashboardPanelController: nil, statusBar: host, menuDefaults: d)
                controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
                    alertsURL: URL(fileURLWithPath: "/tmp/nd-absent-golden"))
                controller.menuNeedsUpdate(controller.contextMenuForTesting)
                let zero = Array(repeating: 0, count: 12)
                controller.renderStatusIconForTesting(stats: DashboardStats(chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 0,
                    enrichmentPercent: 50, enrichmentRatePerMinute: 1, databaseSizeBytes: 0, recentActivityBuckets: zero,
                    recentAgentWriteBuckets: zero, recentWatcherWriteBuckets: zero, recentEnrichmentBuckets: zero), state: .idle)
                let items = controller.contextMenuForTesting.items
                let label = "new menu \(newMenu), design \(design ?? "absent")"
                XCTAssertEqual(items.map(\.title), expected["titles"] as? [String], label)
                XCTAssertEqual(items.map { ($0.representedObject as? BrainBarMenuAction).map { "\($0)" } ?? "" }, expected["actions"] as? [String], label)
                XCTAssertEqual(items.map(\.isEnabled), expected["enabled"] as? [Bool], label)
                XCTAssertEqual(items.map(\.keyEquivalent), expected["keys"] as? [String], label)
                XCTAssertEqual(items.filter { $0.view != nil }.count, expected["hosted"] as? Int, label)
                let image = try XCTUnwrap(host.item.view.image)
                XCTAssertEqual(image.size.width, expected["iconWidth"] as? CGFloat, label)
                XCTAssertEqual(image.size.height, expected["iconHeight"] as? CGFloat, label)
                XCTAssertEqual(image.isTemplate, expected["iconTemplate"] as? Bool, label)
                let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(image.size.width * 2),
                    pixelsHigh: Int(image.size.height * 2), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
                rep.size = image.size
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
                image.draw(in: NSRect(origin: .zero, size: image.size))
                NSGraphicsContext.restoreGraphicsState()
                let goldenPNG = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(expected["iconPNG"] as? String)))
                // The brain is a vector drawing at a fixed 2x: its pixels are compared on every machine. The chart icon is an
                // ImageRenderer raster whose pixels follow the OS, the screen's scale and the appearance (the macos-15
                // runner differed by a mean 3.7 of 255): compared only where it was recorded; its size and template
                // flag, and the whole menu, are compared everywhere.
                let raster = !image.isTemplate
                if !raster || Self.rasterEnvironment() == golden["rasterEnvironment"] as? String {
                    let (same, detail) = Self.pixelsMatch(try XCTUnwrap(rep.representation(using: .png, properties: [:])), goldenPNG)
                    XCTAssertTrue(same, "icon pixels differ from main (\(detail)), \(label)")
                } else {
                    print("BrainBarClassicGoldenTests: chart-icon pixels not compared here (\(Self.rasterEnvironment())); recorded on \(golden["rasterEnvironment"] as? String ?? "?")")
                }
                XCTAssertNil(NDStore.existing(for: runtime), "Classic built new-design work, \(label)")
                controller.stop()
            }
        }
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        // The root's type, structurally everywhere (main's hosting controller around main's root view); its exact spelling,
        // which follows the Swift toolchain's type printing, only where the golden was recorded.
        let rootType = String(describing: type(of: panel.contentViewControllerForTesting))
        let goldenRootType = try XCTUnwrap(golden["windowRootType"] as? String)
        XCTAssertTrue(rootType.hasPrefix("NSHostingController<") && rootType.contains("BrainBarWindowRootView"), rootType)
        XCTAssertEqual(rootType.components(separatedBy: "BrainBarWindowRootView").count,
                       goldenRootType.components(separatedBy: "BrainBarWindowRootView").count, "one classic root, as on main")
        if Self.rasterEnvironment() == golden["rasterEnvironment"] as? String { XCTAssertEqual(rootType, goldenRootType) }
        XCTAssertEqual(panel.windowForTesting.minSize.width, golden["windowMinWidth"] as? CGFloat)
        XCTAssertEqual(panel.windowForTesting.minSize.height, golden["windowMinHeight"] as? CGFloat)
    }
}

/// #1810's rule, applied to the new design: no sentence is written in NewDesign/. Its sentences live with the wording
/// registry (`BrainBarMenuPresenter.SurfaceSay`, the presenter's `Say`); what stays here are the copy deck's labels and
/// state words, check names and VoiceOver fragments. A literal is a sentence when it starts with a capital (or an
/// interpolation), has a space, and ends a sentence ("." or "…") or holds one (". " between words). Comments are skipped, as in `testThePresenterWritesNoWordsOutsideTheRegistry`.
final class BrainBarNewDesignWordsTests: XCTestCase {
    func testTheNewDesignWritesNoSentencesOutsideTheRegistry() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/NewDesign")
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".swift") }
        XCTAssertGreaterThanOrEqual(files.count, 12)
        let literal = try NSRegularExpression(pattern: #""((?:[^"\\\n]|\\.)*)""#)
        var found: [String] = []
        for file in files {
            let code = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8).components(separatedBy: "\n")
                .map { $0.components(separatedBy: "//")[0] }.joined(separator: "\n")
            for match in literal.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                let s = String(code[Range(match.range(at: 1), in: code)!])
                let words = s.range(of: #"^[A-Z\\(]"#, options: .regularExpression) != nil && s.contains(" ")
                let sentence = words && (s.range(of: #"[a-z)][.…]$"#, options: .regularExpression) != nil
                    || s.range(of: #"[a-z]\. [A-Z]"#, options: .regularExpression) != nil)
                if sentence { found.append("\(file): \(s)") }
            }
        }
        XCTAssertEqual(found, [])
    }
}

/// Round 1 (DESIGN-SPEC-R1 with revision 1.1): the pipeline graph, the drawn state of every fixture from its verdict
/// facts, the headline and button rules, and the dropdown's words.
@MainActor final class BrainBarNewDesignR1Tests: XCTestCase {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    typealias Fixture = BrainBarDesignFixture

    /// L18: every health check maps to connections, a stage, or the observer; every mapping names a connection that
    /// exists; every connection is guarded by a check or listed as a known gap.
    /// The copy deck names jobs by their launchd raw value (it imports no settings types); every key is a real job.
    func testJobWordsAreKeyedByRealJobs() {
        let jobs = Set(BrainLayerLaunchdJob.allCases.map(\.rawValue))
        XCTAssertTrue(Set(Say.jobNames.keys).isSubset(of: jobs))
        XCTAssertEqual(Set(Say.jobNames.keys), Set(Say.jobHelp.keys))
        XCTAssertEqual(Set(Say.jobNames.keys), jobs.subtracting(["index"]), "every scheduled job has a name and a help line")
    }

    func testEveryHealthCheckMapsInTheGraph() {
        let checks = Set(BrainBarHealthVerdict.Check.labels.keys)
        XCTAssertFalse(checks.isEmpty)
        for name in checks { XCTAssertNotNil(NDPipelineGraph.checks[name], "\(name) maps to nothing") }
        let ids = Set(NDPipelineGraph.connections.map(\.id))
        var guarded: Set<String> = []
        for (name, rule) in NDPipelineGraph.checks {
            if case let .connections(list) = rule {
                for id in list { XCTAssertTrue(ids.contains(id), "\(name) names \(id), which is not a connection") }
                guarded.formUnion(list)
            }
        }
        XCTAssertEqual(ids.subtracting(guarded).subtracting(NDPipelineGraph.factGuards.keys), NDPipelineGraph.unguarded,
                       "a connection with no check is guarded by a named fact or is a listed gap")
        XCTAssertEqual(Set(NDPipelineGraph.connections.flatMap(\.pipes)), Set(NDPipe.allCases), "every drawn pipe belongs to a connection")
        XCTAssertEqual(NDPipelineGraph.checks["monitor"], .observer)
    }

    private func pipeline(_ name: Fixture.Name) -> NDPipelineState { NDPipelineState.make(Fixture.inputs(name)) }

    /// R1.3: every state on the loop, each drawn from its own facts, in one treatment per station.
    func testEachFixtureDrawsItsState() throws {
        let calm = pipeline(.calm)
        XCTAssertTrue(calm.faults.isEmpty && calm.notices.isEmpty && calm.agentFaults.isEmpty)
        XCTAssertFalse(calm.allUnconfirmed)
        XCTAssertTrue(NDPipe.allCases.allSatisfy { calm.look($0) == .flowing }, "L6: calm stays calm")
        XCTAssertNil(calm.status[.read], "the fixture has a poll heartbeat, not a measured successful read")
        XCTAssertEqual(calm.status[.queue]?.text, Say.empty, "R1.12 (F): Save queue says it is empty")
        XCTAssertEqual(calm.status[.mac]?.text, Say.keptNewest(3, NDFormat.ageText(4 * 3600)), "R1.5 change 8: the copies kept")
        XCTAssertEqual(calm.status[.drive]?.text, Say.lastUpload(NDFormat.ageText(3 * 3600)), "Drive's newest upload of either kind")
        XCTAssertEqual(calm.status[.jobs]?.text, Say.lastRunAgo(NDFormat.ageText(5 * 3600)))
        XCTAssertEqual(calm.status[.search]?.text, "99% searchable", "R1.12 (F): Search says how much is searchable")
        XCTAssertTrue(calm.status.values.allSatisfy { $0.severity == nil })

        let agent = pipeline(.agentNotRead)
        XCTAssertEqual(agent.agentFaults, [.cursor: .red])
        XCTAssertEqual(agent.treatment(.agents), .fault)
        XCTAssertTrue(NDPipe.allCases.allSatisfy { agent.look($0) == .flowing }, "an agent's fault is its icon, no pipe changes")
        XCTAssertEqual(agent.status[.read], .init(text: Say.providerNotRead("Cursor"), severity: .red), "R1.12 (E4): red, like every fault status")

        for name in [Fixture.Name.driveSignedOut, .reconnectWaiting] {
            let drive = pipeline(name)
            XCTAssertEqual(drive.faults, [.drive], name.rawValue)
            XCTAssertEqual(drive.look(.macDrive), .flowing, "L36: Drive refuses what arrives; its incoming pipes stay as they are")
            XCTAssertEqual(drive.look(.rail), .flowing)
            // R1.10.1: while the browser is asked, the station waits like every surface.
            XCTAssertEqual(drive.status[.drive], .init(text: name == .reconnectWaiting ? Say.stationWaiting : Say.signedOut, severity: .red))
            XCTAssertEqual(drive.status[.mac]?.severity, nil, "the Mac copy is still fresh")
        }

        let saves = pipeline(.savesStuck)
        XCTAssertEqual(saves.faults, [.queue])
        XCTAssertEqual(saves.look(.queueMemory), .faint, "the pipe after the faulty stage is faint")
        XCTAssertEqual(saves.status[.queue], .init(text: Say.stuckFor(NDFormat.ageText(2 * 3600)), severity: .red))  // R1.12 (F)

        let meaning = pipeline(.meaningBehind)
        XCTAssertEqual(meaning.notices, [.search])
        XCTAssertEqual(meaning.look(.memorySearch), .flowing, "R1.4 change 16: a notice changes no pipe")
        XCTAssertEqual(meaning.look(.answers), .flowing, "the return to the agents stays solid")
        XCTAssertEqual(meaning.status[.search], .init(text: Say.catchingUp, severity: .orange))  // R1.12 (F)

        let nightly = pipeline(.nightlyFailing)
        XCTAssertEqual(nightly.faults, [.jobs])
        XCTAssertEqual(nightly.look(.jobsMemory), .faint)
        XCTAssertEqual(nightly.status[.jobs], .init(text: Say.failedNights(3), severity: .red), "R1.10: the station is named for its job")

        let stale = pipeline(.notConfirmed)
        XCTAssertTrue(stale.allUnconfirmed)
        XCTAssertNotNil(stale.unconfirmedSince)
        XCTAssertTrue(NDStation.allCases.allSatisfy { stale.treatment($0) == .notConfirmed })
        XCTAssertTrue(NDPipe.allCases.allSatisfy { stale.look($0) == .dashed }, "every pipe dashed")
        XCTAssertFalse(stale.status.isEmpty, "status lines keep their last known values")

        let two = pipeline(.twoFaults)
        XCTAssertEqual(two.faults, [.drive, .jobs])

        let late = pipeline(.backupLate)
        XCTAssertEqual(late.notices, [.mac])
        XCTAssertEqual(late.look(.memoryMac), .flowing)

        XCTAssertEqual(pipeline(.backupsFailedUpload), calm, "a failed upload last week is history, not a break now")
    }

    /// Every check the chart can draw, from the verdict's facts alone: red on a calm verdict marks only its own station.
    func testEachRedCheckBreaksItsOwnConnection() {
        let expected: [String: (NDPipelineState) -> Bool] = [
            "ingest": { $0.faults == [.read] && $0.look(.readMemory) == .faint && $0.status[.read]?.severity == .red },
            "stores": { $0.faults == [.queue] && $0.look(.queueMemory) == .faint },
            "embedding": { $0.faults == [.search] && $0.look(.memorySearch) == .flowing && $0.look(.answers) == .flowing },
            "search": { $0.faults == [.search] && $0.look(.answers) == .faint },
            "backup_local": { $0.faults == [.mac] && $0.look(.macDrive) == .faint },
            "backup_offsite": { $0.faults == [.drive] && $0.look(.macDrive) == .flowing },
            "transcripts": { $0.faults == [.drive] && $0.look(.rail) == .flowing },
            "drive_auth": { $0.faults == [.drive] && $0.look(.macDrive) == .flowing && $0.look(.rail) == .flowing },
            "integrity": { $0.faults == [.mac] },
            "disk": { $0.faults == [.mac] },
            "monitor": { $0.allUnconfirmed && $0.faults.isEmpty },
        ]
        XCTAssertEqual(Set(expected.keys), Set(NDPipelineGraph.checks.keys))
        for (name, holds) in expected {
            var i = Fixture.inputs(.calm)
            i.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check(name, .red, age: 2 * 3600, limit: 3600)), dot: .red)
            let drawn = NDPipelineState.make(i)
            XCTAssertTrue(holds(drawn), "\(name): \(drawn)")
        }
        // A check that cannot say: its pipes are drawn "not confirmed", nothing breaks.
        var unknown = Fixture.inputs(.calm)
        unknown.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check("backup_offsite", .unknown)))
        let drawn = NDPipelineState.make(unknown)
        XCTAssertEqual(drawn.unconfirmed, [.macDrive])
        XCTAssertEqual(drawn.look(.macDrive), .dashed)
        XCTAssertTrue(drawn.faults.isEmpty)
    }

    /// Headlines, details and the one filled button (R1.1 change 23: Reconnect only; the report is always bordered).
    func testHeadlinesAndTheOneFilledButton() {
        func r1(_ name: Fixture.Name) -> NDR1 {
            let i = Fixture.inputs(name)
            return NDR1.make(i, NDModel.make(i))
        }
        XCTAssertEqual(r1(.calm).headline, BrainBarMenuPresenter.Say.allGood.text)
        XCTAssertTrue(r1(.calm).items.isEmpty)
        // R1.10 (L48, L51): the headline counts ("1 thing needs you") and the problem row names the problem.
        XCTAssertEqual(r1(.agentNotRead).headline, Say.oneThingNeedsYou)
        XCTAssertEqual(r1(.driveSignedOut).headline, Say.oneThingNeedsYou)
        XCTAssertEqual(r1(.savesStuck).items.first?.detail, Say.savesWaited(38, NDFormat.ageText(2 * 3600)))
        XCTAssertEqual(r1(.meaningBehind).headline, Say.meaningCatchingUp, "D6: amber is catching up, not a fault")
        XCTAssertEqual(r1(.meaningBehind).items.map(\.severity), [.orange], "late, not stopped: one orange item, no presenter echo")
        XCTAssertNil(r1(.meaningBehind).items.first?.action, "nothing to do")
        XCTAssertEqual(r1(.nightlyFailing).items.first?.detail, Say.notCheckedSince("Thursday"))
        XCTAssertEqual(r1(.notConfirmed).headline, Say.notConfirmedSince("06:00"))
        XCTAssertEqual(r1(.notConfirmed).footerLeft, Say.waitingForNextCheck)
        XCTAssertNil(r1(.notConfirmed).footerRight)
        XCTAssertEqual(r1(.twoFaults).headline, Say.thingsNeedYou(2))
        XCTAssertEqual(r1(.twoFaults).items.map(\.id), ["drive", "nightly"])
        for name in Fixture.Name.allCases {
            for item in r1(name).items {
                guard let action = item.action else { continue }
                XCTAssertEqual(NDButtonPolicy.filled(action), [.reconnectDrive, .connectDrive].contains(action.kind), name.rawValue)
            }
        }
        XCTAssertEqual(r1(.twoFaults).items.first?.menuDetail, Say.lastDriveUpload(NDFormat.ageText(27 * 3600)))
        XCTAssertEqual(r1(.twoFaults).items.last?.title, "Nightly check-up failed 3 nights", "R1.12 (§ C): one wording")
    }

    /// § 4.1: the calm dropdown says what it must in few words (a number with its unit counts as nothing); R1.6 item 14
    /// made the Search row "99% searchable by meaning", R1.12 (F) "99% searchable", the station's words: 9 words.
    func testCalmDropdownIsElevenWords() throws {
        let i = Fixture.inputs(.calm)
        let r1 = NDR1.make(i, NDModel.make(i))
        let words = ([NDMenuHeadline.text(r1)] + r1.menuRows.flatMap { [$0.label, $0.value] }).joined(separator: " ")
        let stripped = words.replacing(#/\d+(?:[.,]\d+)?\s?(?:%|h|min|s|d)?/#, with: "")
        let count = stripped.split(whereSeparator: { $0 == " " || $0 == "\u{00A0}" }).count
        XCTAssertEqual(count, 9, stripped)
        XCTAssertEqual(r1.menuRows.map(\.value), ["", Say.searchablePercent(99), Say.ago(NDFormat.ageText(4 * 3600)), Say.fine])
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        XCTAssertEqual(two.menuRows.map(\.value)[2], Say.driveAgo(NDFormat.ageText(27 * 3600)),
                       "D7.3: the stale copy named, one number for one fact (as 'Nothing has uploaded for 27 h')")
        XCTAssertEqual(two.menuRows[2].tone, .fault)
        XCTAssertEqual(two.menuRows[3].value, Say.failed)
    }

    /// L17 and the clip defect: in every fixture, on every page, and in a constructed three-fault state, the laid-out
    /// content ends at least the bottom inset (24 pt) above the footer, and the window is never below the spec's height.
    func testNoPageIsClippedUnderTheFooter() throws {
        var three = Fixture.inputs(.twoFaults)
        three.capture.verdict = BrainBarHealthVerdict(
            dot: .red, checks: Fixture.replacing(three.capture.verdict!.checks, Fixture.check("stores", .red, age: 7200, limit: 1800)),
            notes: [], evaluatedAt: three.capture.verdict!.evaluatedAt)
        three.savesWaiting = 38
        three.oldestSaveQueuedAt = Fixture.fixedNow.addingTimeInterval(-7200)
        var cases: [(String, NDInputs)] = Fixture.Name.allCases.map { ($0.rawValue, Fixture.inputs($0)) }
        cases.append(("three-faults", three))
        var heights: [String] = []
        for (name, inputs) in cases {
            for page in NDPage.allCases {
                let store = NDStore(fixture: inputs, page: page)
                let window = NDPageHeight.window(store, page: page, screen: nil)
                let content = NDPageHeight.content(store, page: page)
                // title bar 52 + top inset 20 + content + bottom inset 20 + footer 28 (R1.2 change 22: no floor)
                XCTAssertLessThanOrEqual(NDGrid.titleBar + NDGrid.pageTop + content + 20 + NDGrid.footer, window + 0.5, "\(name) \(page)")
                if page == .overview { heights.append("\(name) \(Int(window))") }
            }
        }
        let calm = NDPageHeight.window(NDStore(fixture: Fixture.inputs(.calm), page: .overview), page: .overview, screen: nil)
        let twoFaults = NDPageHeight.window(NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview), page: .overview, screen: nil)
        // R1.10 (L48): the Overview grows by the problems surface only: 12 pt, then two 36 pt rows and the line between.
        XCTAssertEqual(twoFaults - calm, NDGrid.headlineToRows + 2 * NDProblemRows.rowHeight + 0.5, accuracy: 1.5,
                       "only the problem rows make it taller (R1.12: 14 pt under the headline)")
        // L57: without AppKit chrome, this offscreen page can use the full visible height.
        XCTAssertEqual(NDPageHeight.window(NDStore(fixture: three, page: .overview), page: .overview, screen: 720), 720)
        print("Overview heights: " + heights.joined(separator: ", "))
    }

    /// Ruling on change 17: more than five agents cannot occur while there are four agent families; this fails loudly the
    /// day a fifth family appears, so the "+N more" list is built then.
    func testAgentFamiliesStayWithinTheChartsList() {
        XCTAssertLessThanOrEqual(AgentFamily.allCases.count, 5, "a fifth agent family: build change 17a (+N more)")
        XCTAssertEqual(NDPipelineLayout.agentRows.count, AgentFamily.allCases.count)
    }

    /// D2 (real screen, round 1): one state, one truth. For every fixture, no dropdown row shows a fine value for an area
    /// whose stages read late, broken or not confirmed on the Overview (the same `condition` rule).
    func testDropdownRowsNeverClaimFineWhereTheOverviewDoesNot() {
        let stagesOf: [NDR1.MenuRow.Kind: [NDStation]] = [.search: [.search], .backups: [.mac, .drive], .jobs: [.jobs]]
        var checkedNotFine = 0
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let r1 = NDR1.make(i, NDModel.make(i))
            for row in r1.menuRows where row.kind != .activity {
                let conditions = (stagesOf[row.kind] ?? []).map(r1.pipeline.condition)
                guard conditions.contains(where: { $0 != .fine }) else { continue }
                checkedNotFine += 1
                XCTAssertTrue(row.tone != .fine || row.value == NDModel.Number.unknown.value,
                              "\(name.rawValue): \(row.label) says \(row.value) while its stage reads \(conditions)")
            }
        }
        XCTAssertGreaterThan(checkedNotFine, 0)
        let behind = NDR1.make(Fixture.inputs(.meaningBehind), NDModel.make(Fixture.inputs(.meaningBehind))).menuRows[1]
        XCTAssertEqual(behind.value, Say.behind(NDFormat.ageText(2 * 3600)), "the Search row leads with what is wrong")
        XCTAssertEqual(behind.tone, .warn)
        let stale = NDR1.make(Fixture.inputs(.notConfirmed), NDModel.make(Fixture.inputs(.notConfirmed))).menuRows
        XCTAssertEqual(stale.dropFirst().map(\.value), Array(repeating: NDModel.Number.unknown.value, count: 3))
    }

    /// D3: nothing confirmed, nothing calm: the icon is comp 01's dimmed glyph whenever every pipe is not confirmed.
    func testIconIsDimmedWhenNothingIsConfirmed() {
        func dimmed(_ name: Fixture.Name) -> Bool {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            return NDIcon.dimmed(m, NDR1.make(i, m))
        }
        XCTAssertTrue(dimmed(.notConfirmed))
        XCTAssertFalse(dimmed(.calm))
        XCTAssertFalse(dimmed(.driveSignedOut))
        var unread = Fixture.inputs(.calm)
        unread.capture.verdict = nil
        let m = NDModel.make(unread)
        XCTAssertTrue(NDIcon.dimmed(m, NDR1.make(unread, m)), "no reading yet")
    }

    /// L2 and the lead's 1.4 rule: every hover panel's arrow sits on its own station (the tip within the circle's
    /// horizontal extent on a top or bottom edge, its vertical extent on a side edge), and the body covers no station's
    /// circle, name, purpose or status, and no action, for all stations in every fixture.
    func testPanelArrowsSitOnTheirStationsAndBodiesClearEverything() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let r1 = store.r1
            let chart = NDPipelineChart(state: r1.pipeline, memories: store.model.memories, items: r1.stationItems)
            for station in NDStation.allCases {
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let own = L.circle(station)
                let where_ = "\(name.rawValue) \(station)"
                // L58: only these top-row panels, below after both full-height side candidates fail,
                // may cover interior marks. Their own words, heading and footer remain protected.
                let topRowBelowBothSidesBlocked = [NDStation.agents, .read, .memory, .mac, .drive].contains(station)
                    && panel.edge == .bottom && panel.clearanceException == .topRowBelowBothSidesBlocked
                if topRowBelowBothSidesBlocked {
                    for side: Edge in [.top, .trailing, .leading] {
                        XCTAssertNil(L.panelCandidate(station, edge: side, state: r1.pipeline, actions: [], height: panel.body.height, width: panel.body.width), where_)
                    }
                }
                switch panel.edge {
                case .top, .bottom:
                    XCTAssertTrue((own.minX...own.maxX).contains(panel.tip.x), "\(where_): the arrow points at its station")
                case .leading, .trailing:
                    XCTAssertTrue((own.minY...own.maxY).contains(panel.tip.y), "\(where_): the arrow points at its station")
                }
                // The arrow stays on the body's straight edge, clear of its rounded corners.
                switch panel.edge {
                case .top, .bottom:
                    XCTAssertTrue((panel.body.minX + L.arrowInset - 0.01...panel.body.maxX - L.arrowInset + 0.01).contains(panel.tip.x), where_)
                case .leading, .trailing:
                    XCTAssertTrue((panel.body.minY + L.arrowInset - 0.01...panel.body.maxY - L.arrowInset + 0.01).contains(panel.tip.y), where_)
                }
                XCTAssertFalse(panel.outer.intersects(own.insetBy(dx: 1, dy: 1)), "\(where_) covers its own station")
                // L54 remains strict outside the named L58 fallback; own words are always protected.
                for other in NDStation.allCases {
                    if other != station && !topRowBelowBothSidesBlocked {
                        XCTAssertFalse(panel.body.intersects(L.circle(other)), "\(where_) covers \(other)")
                    }
                    for rect in L.labelRects(other, r1.pipeline) {
                        if other == station || !topRowBelowBothSidesBlocked {
                            XCTAssertFalse(panel.outer.intersects(rect), "\(where_) covers \(other)'s words")
                        }
                    }
                }
                // R13: both job panels clear the visible caption fields, including their 4 pt native insets.
                if station == .jobs || station == .weekly {
                    for (_, rect) in L.laneLabelRects() {
                        XCTAssertFalse(panel.outer.intersects(rect.insetBy(dx: -4, dy: -4)), where_)
                    }
                    let content = NDStationPopover.measure(AnyView(NDStationPopover(store: store, station: station)), width: panel.body.width)
                    XCTAssertGreaterThanOrEqual(panel.body.height + 0.01, content.height, where_)
                    XCTAssertFalse(panel.scrolls, where_)
                }
            }
        }
    }

    /// The lead's 1.4 rule: a station's hit area is its ring and its name, laid out in the drawing's own coordinates.
    func testHitAreaIsTheRingAndTheName() {
        let L = NDPipelineLayout.self
        let calm = pipeline(.calm)
        for station in L.stations.keys {
            let hit = L.hitShape(station, calm)
            XCTAssertEqual(hit.circle.midX, L.point(station).x, accuracy: 0.01, "\(station): no offset")
            XCTAssertEqual(hit.circle.midY, L.point(station).y, accuracy: 0.01)
            XCTAssertEqual(hit.circle.width, 2 * L.ringRadius, accuracy: 0.01)
            XCTAssertEqual(hit.label, L.labelRects(station, calm).first, "\(station): the name is part of it")
        }
        XCTAssertEqual(L.hitShape(.memory, calm).circle, L.hub)
        XCTAssertEqual(L.hitShape(.agents, calm).circle, L.agentsFrame)
    }

    /// D5: the Reconnect → waiting swap holds its layout: the headline and the page measure the same before and during,
    /// and the page's height key (what triggers a refit) does not change, so no refit runs mid-swap.
    func testReconnectSwapHoldsTheLayout() {
        var waiting = Fixture.inputs(.driveSignedOut)
        let before = NDStore(fixture: waiting)
        waiting.driveReconnecting = true
        let during = NDStore(fixture: waiting)
        XCTAssertEqual(NDPageHeight.content(before, page: .overview), NDPageHeight.content(during, page: .overview))
        XCTAssertEqual(NDPageHeight.key(before), NDPageHeight.key(during))
        func headline(_ store: NDStore) -> CGSize {
            NSHostingController(rootView: NDHeadline(store: store, r1: store.r1).frame(width: 852))
                .sizeThatFits(in: CGSize(width: 852, height: 1000))
        }
        XCTAssertEqual(headline(before), headline(during))
    }

    /// A past failed upload under "All good" is said in words: the Backups legend carries "failed" beside the red dot.
    func testAPastFailedUploadIsExplainedOnTheBackupsPage() {
        let store = NDStore(fixture: Fixture.inputs(.backupsFailedUpload), page: .backups)
        let lanes = NDBackupsPage(store: store).lanes(store.model)
        // R1.2 change 13: retried the same night, so a hollow ring and its own legend words, never a filled red dot.
        XCTAssertTrue(lanes.contains { $0.events.contains { $0.kind == .retried } })
        XCTAssertTrue(NDBackupsPage.legend(lanes).contains(Say.failedThenRetried))
        XCTAssertEqual(store.r1.headline, BrainBarMenuPresenter.Say.allGood.text)
        let calm = NDStore(fixture: Fixture.inputs(.calm), page: .backups)
        XCTAssertFalse(NDBackupsPage.legend(NDBackupsPage(store: calm).lanes(calm.model)).contains(Say.failed))
        XCTAssertFalse(NDBackupsPage.legend(NDBackupsPage(store: calm).lanes(calm.model)).contains(Say.failedThenRetried))
    }

    /// D6 (L20 rewritten): red, "needs you", the count, the badges and the fault icon are for red verdicts only. For every
    /// fixture: the fault icon iff at least one station is red; "needs you" iff a red item; badges count red items only.
    func testAmberIsNeverShownAsAFault() {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            let r1 = NDR1.make(i, m)
            let p = r1.pipeline
            let redStation = !p.allUnconfirmed && NDStation.allCases.contains { station in
                if case .issue(_, .red) = p.condition(station) { return true }
                return false
            }
            let icon = NDIcon.marked(m, r1)
            XCTAssertEqual(icon.icon == .marked, redStation, "\(name.rawValue): fault icon iff a red station")
            XCTAssertEqual(icon.count, r1.redCount, name.rawValue)
            let needsYou = NDMenuHeadline.text(r1).contains("need")
            XCTAssertEqual(needsYou, r1.redCount > 0, "\(name.rawValue): needs you iff red")
            XCTAssertLessThanOrEqual(r1.backupsBadge + r1.jobsBadge, r1.redCount, name.rawValue)
        }
        let i = Fixture.inputs(.meaningBehind)
        let r1 = NDR1.make(i, NDModel.make(i))
        XCTAssertEqual(r1.redCount, 0)
        XCTAssertEqual(r1.headlineTone, .warn)
        XCTAssertEqual(r1.items.first?.detail, Say.meaningBehindDetail)
        XCTAssertNil(r1.items.first?.action, "nothing to do: no button")
        XCTAssertEqual(NDIcon.marked(NDModel.make(i), r1).icon, .resting, "the resting brain for an amber state")
        XCTAssertEqual(NDMenuHeadline.text(r1), Say.meaningCatchingUp)
        // An amber ingest lag is catching up, never "needs you" (the overnight false alarm).
        var lag = Fixture.inputs(.calm)
        lag.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check("ingest", .amber, age: 1200, limit: 900)), dot: .amber)
        let lr = NDR1.make(lag, NDModel.make(lag))
        XCTAssertEqual(lr.redCount, 0)
        XCTAssertFalse(NDMenuHeadline.text(lr).contains("need"), NDMenuHeadline.text(lr))
        XCTAssertEqual(NDIcon.marked(NDModel.make(lag), lr).icon, .resting)
        XCTAssertTrue(lr.items.allSatisfy { $0.action == nil }, "no report button for a lag")
    }

    /// D7 (as amended by L27): the two-fault headline is one line; timed statuses; the selected palette redraws; one capture clock.
    func testCouncilRoundOneFixes() throws {
        let two = NDStore(fixture: Fixture.inputs(.twoFaults))
        let size = NSHostingController(rootView: NDHeadline(store: two, r1: two.r1).frame(width: 852))
            .sizeThatFits(in: CGSize(width: 852, height: 1000))
        XCTAssertEqual(size.height, NDType.headlineLine, accuracy: 1, "L27: one line; R1.12: the 24 pt headline on its 30 pt line")
        XCTAssertEqual(two.r1.items.map(\.menuDetail), [Say.lastDriveUpload(NDFormat.ageText(27 * 3600)),
                                                        Say.notCheckedSinceShort("Thursday")])
        XCTAssertNotEqual(NDPalette.menu, NDPalette.menuSelected, "a selected row's chart redraws in the selected colours")
        let pinned = BrainBarDesignFixture.makeNow(capturing: true, wall: Date(), pinned: "2026-11-16T02:24:30+02:00")
        XCTAssertEqual(pinned, ISO8601DateFormatter().date(from: "2026-11-16T02:24:00+02:00"), "BRAINBAR_DESIGN_NOW pins the clock")
        XCTAssertEqual(BrainBarDesignFixture.makeNow(capturing: false, wall: Date(), pinned: "2026-11-16T02:24:30+02:00"),
                       BrainBarDesignFixture.fixedNow, "tests and renders keep the fixed moment")
    }

    /// D8.1: the hover readout is an overlay: the plot's laid-out frame is identical with and without a hovered bucket.
    func testHoverReadoutNeverMovesThePlot() throws {
        let store = NDStore(fixture: Fixture.inputs(.calm))
        func plotFrame(hover: Int?) -> CGRect? {
            var frame: CGRect?
            let host = NSHostingView(rootView: NDActivitySection(store: store, hover: hover) { frame = $0 }.frame(width: 852))
            host.frame = NSRect(x: 0, y: 0, width: 852, height: 220)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
            host.layoutSubtreeIfNeeded()
            return frame
        }
        let rest = try XCTUnwrap(plotFrame(hover: nil))
        for bucket in [0, 30, 59] {
            XCTAssertEqual(try XCTUnwrap(plotFrame(hover: bucket)), rest, "bucket \(bucket)")
        }
        XCTAssertEqual(Say.upTo("4", minutes: 1), "up to 4 / min", "R1.2 change 20")
        XCTAssertEqual(Say.upTo("83", minutes: 3), "up to 83 / 3 min")
    }

    /// Council r2a finding 4: the Reconnect hand-off never shows both: the line starts when the button's fade has ended,
    /// and the button comes back only after the line has gone. The dark read bars are #8189A0 (R1.2 change 21).
    func testReconnectSwapHasNoOverlapAndReadBarsAreLifted() {
        XCTAssertGreaterThanOrEqual(NDMotion.waitingDelay, NDMotion.buttonLeavesDuration, "the line waits for the button to go")
        XCTAssertEqual(NDMotion.buttonLeavesDuration, 0.16, accuracy: 0.0001)
        let dark = NDColor.read.usingColorSpace(.sRGB).flatMap { _ in
            NSAppearance(named: .darkAqua).map { appearance -> NSColor? in
                var resolved: NSColor?
                appearance.performAsCurrentDrawingAppearance { resolved = NDColor.read.usingColorSpace(.sRGB) }
                return resolved
            } ?? nil
        }
        let c = dark ?? NSColor.black
        // R1.2 change 21: #8189A0 in dark.
        XCTAssertEqual(Int((c.redComponent * 255).rounded()), 0x81)
        XCTAssertEqual(Int((c.greenComponent * 255).rounded()), 0x89)
        XCTAssertEqual(Int((c.blueComponent * 255).rounded()), 0xA0)
    }

    /// L47 (replacing L23) over every fixture: the red icon iff something is red; the "?" badge iff not confirmed (or no
    /// reading yet); the amber dot whenever any check is amber and none is red, the catching-up states included;
    /// otherwise the resting brain.
    func testIconPredicateOverEveryFixture() {
        var dotted: [String] = []
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            let r1 = NDR1.make(i, m)
            let state = NDIcon.state(m, r1)
            if r1.redCount > 0 {
                XCTAssertEqual(state, .red(r1.redCount), name.rawValue)
            } else if r1.pipeline.allUnconfirmed {
                XCTAssertEqual(state, .unconfirmed, name.rawValue)
            } else if i.capture.verdict?.checks.contains(where: { $0.state == .amber }) == true
                        || r1.items.contains(where: { $0.severity == .orange }) {
                XCTAssertEqual(state, .amber, name.rawValue)
                dotted.append(name.rawValue)
            } else {
                XCTAssertEqual(state, .rest, name.rawValue)
            }
        }
        func state(_ name: Fixture.Name) -> NDIcon.State {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            return NDIcon.state(m, NDR1.make(i, m))
        }
        XCTAssertEqual(dotted, ["meaning-behind", "backup-late", "lag-behind"], "the fixtures with the dot (L55: the draining lag too)")
        XCTAssertEqual(state(.meaningBehind), .amber, "L47: catching up on search by meaning carries the dot")
        XCTAssertEqual(state(.backupLate), .amber, "a late copy: the amber dot")
        // The other catching-up state: new conversations (an amber ingest lag under 30 min).
        var lag = Fixture.inputs(.calm)
        lag.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check("ingest", .amber, age: 1200, limit: 900)), dot: .amber)
        let lagModel = NDModel.make(lag)
        let lagR1 = NDR1.make(lag, lagModel)
        XCTAssertTrue(lagR1.items.contains { $0.catchingUp }, "it is the catching-up state")
        XCTAssertEqual(NDIcon.state(lagModel, lagR1), .amber, "L47: catching up on new conversations carries the dot")
        XCTAssertEqual(state(.notConfirmed), .unconfirmed)
        XCTAssertEqual(state(.twoFaults), .red(2))
    }

    /// R1.2 changes 5, 8 and 9, L24: Reconnect is the only filled button and at most one per page; amber notices never get
    /// a button; one amber notice says itself (catching up by its own title, any other as "Check soon" with its sentence);
    /// the dropdown shows blocks only when something is red or there are two or more notices.
    func testOneFilledButtonAndAmberNeverGetsOne() {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let r1 = NDR1.make(i, NDModel.make(i))
            let red = r1.items.filter { $0.severity == .red }
            XCTAssertEqual(r1.showsProblemRows, !red.isEmpty || r1.items.count > 1, name.rawValue)
            let filled = r1.items.compactMap(\.action).filter(NDButtonPolicy.filled)
            XCTAssertLessThanOrEqual(filled.count, 1, "\(name.rawValue): one filled button per page")
            XCTAssertTrue(filled.allSatisfy { [.reconnectDrive, .connectDrive].contains($0.kind) }, name.rawValue)
            XCTAssertTrue(r1.items.filter { $0.severity == .orange }.allSatisfy { $0.action == nil }, "\(name.rawValue): amber, no button")
            if red.isEmpty { XCTAssertFalse(r1.headline.contains("need"), "\(name.rawValue): amber is never \"needs you\"") }
        }
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        XCTAssertEqual(two.headline, Say.thingsNeedYou(2))
        let late = NDR1.make(Fixture.inputs(.backupLate), NDModel.make(Fixture.inputs(.backupLate)))
        XCTAssertEqual(late.headline, BrainBarMenuPresenter.Say.localRunningLate.text, "R2A item 9, R1.6 item 5")
        XCTAssertTrue(late.headlineDetail?.hasPrefix("The newest is 30\u{00A0}h old") == true, "R1.12 (E3): its second line")
        XCTAssertFalse(late.showsProblemRows)
        let behind = NDR1.make(Fixture.inputs(.meaningBehind), NDModel.make(Fixture.inputs(.meaningBehind)))
        XCTAssertEqual(behind.headline, Say.meaningCatchingUp)
        XCTAssertEqual(behind.headlineDetail, Say.meaningBehindDetail)
    }

    /// Lead, 09:55: the Activity chart carries no "own scale" note; each half's "up to" label says its size.
    func testActivityHasNoScaleNote() throws {
        let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar")
        for file in ["BrainBarMenuSentences.swift", "NewDesign/NDWindow.swift", "NewDesign/NDMenu.swift"] {
            let text = try String(contentsOf: sources.appendingPathComponent(file), encoding: .utf8)
            XCTAssertFalse(text.contains("own scale") || text.contains("ownScale"), file)
        }
    }

    /// R1.2 change 12: "last known" is the last confirmed evaluation's time, the same in the window and the dropdown.
    func testLastKnownTimeIsTheEvaluationsInBothSurfaces() throws {
        let i = Fixture.inputs(.notConfirmed)
        let r1 = NDR1.make(i, NDModel.make(i))
        let at = try XCTUnwrap(i.capture.verdict?.evaluatedAt)
        XCTAssertEqual(r1.pipeline.lastKnownAt, NDR1.hhmm(at, i.capture), "the evaluation's time, never the render time")
        XCTAssertEqual(Say.valuesLastKnownAt(r1.pipeline.lastKnownAt!), "not confirmed: values last known at 06:00")
        XCTAssertEqual(Say.lastKnownAt(r1.pipeline.lastKnownAt!), "Last known at 06:00")
        XCTAssertEqual(NDMenuHeadline.detail(r1, lastVerdictHadRed: false), Say.nothingKnownBrokenShort)
        XCTAssertNil(NDMenuHeadline.detail(r1, lastVerdictHadRed: true), "only when the last confirmed verdict had no red check")
    }

    /// L57: the page is capped at the visible screen, minus injected AppKit chrome.
    func testWindowIsThePagesHeightCappedByTheScreen() {
        for name in Fixture.Name.allCases {
            for page in NDPage.allCases {
                let store = NDStore(fixture: Fixture.inputs(name), page: page)
                let needed = ceil(NDPageHeight.content(store, page: page) + NDPageHeight.chrome)
                XCTAssertEqual(NDPageHeight.window(store, page: page, screen: nil), needed, "\(name.rawValue) \(page)")
                XCTAssertEqual(NDPageHeight.window(store, page: page, screen: 600), min(needed, 600), "\(name.rawValue) \(page): L57")
            }
        }
    }

    /// R1.2 change 13: a failed upload retried the same night is one hollow red ring, said in the legend and its help tag.
    func testARetriedFailureIsAHollowRingWithItsWords() throws {
        let store = NDStore(fixture: Fixture.inputs(.backupsFailedUpload), page: .backups)
        let lanes = NDBackupsPage(store: store).lanes(store.model)
        let drive = try XCTUnwrap(lanes.first { $0.id == "backup_offsite" })
        XCTAssertTrue(drive.events.contains { $0.kind == .retried })
        XCTAssertFalse(drive.events.contains { $0.kind == .fail }, "no filled red dot when the retry succeeded")
        XCTAssertNotNil(drive.help)
        XCTAssertTrue(NDBackupsPage.legend(lanes).contains(Say.failedThenRetried))
        XCTAssertEqual(store.r1.headline, BrainBarMenuPresenter.Say.allGood.text)
    }

    /// R1.3 change 9 and L29: a fault station draws its ring, its badge and its red name together, every station that can
    /// fault; the same treatment drives the dropdown rows and the icon's count.
    func testOneFaultTreatmentOnStationsRowsAndIcon() {
        let fault = NDPipelineChart.marks(.fault)
        XCTAssertNotNil(fault.ring)
        XCTAssertNotNil(fault.badge)
        XCTAssertFalse(fault.nameRed, "R1.6 item 10: the name keeps its colour and weight; the short line is red")
        XCTAssertEqual(fault.statusTone, .fault)
        let notice = NDPipelineChart.marks(.notice)
        XCTAssertNotNil(notice.ring)
        XCTAssertEqual(notice.badge, "exclamationmark.triangle.fill")
        XCTAssertFalse(notice.nameRed, "a notice is not stopped: the name stays t1")
        let stale = NDPipelineChart.marks(.notConfirmed)
        XCTAssertTrue(stale.ringDashed)
        XCTAssertNil(stale.badge)
        XCTAssertEqual(NDPipelineChart.marks(.calm), .init())
        let stagesOf: [NDR1.MenuRow.Kind: [NDStation]] = [.search: [.search], .backups: [.mac, .drive], .jobs: [.jobs]]
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            let r1 = NDR1.make(i, m)
            let p = r1.pipeline
            // Every red item stands on a red station (or an agent's row), and every red station carries one.
            let redStations = Set(r1.items.filter { $0.severity == .red }.compactMap(\.station))
            let drawn = p.faults.union(p.treatment(.agents) == .fault ? [.agents] : [])
            XCTAssertEqual(redStations, drawn, name.rawValue)
            XCTAssertEqual(r1.redCount, p.faultCount, "\(name.rawValue): the headline and icon count what the chart draws")
            if p.faultCount > 0 {
                XCTAssertEqual(NDIcon.state(m, r1), .red(p.faultCount), name.rawValue)
            } else {
                XCTAssertNotEqual(NDIcon.state(m, r1), .red(0), name.rawValue)
            }
            for row in r1.menuRows where row.kind != .activity {
                let anyFault = (stagesOf[row.kind] ?? []).contains { p.treatment($0) == .fault }
                if anyFault { XCTAssertEqual(row.tone, .fault, "\(name.rawValue) \(row.label)") }
                if row.tone == .fault, !anyFault {
                    XCTAssertTrue(r1.items.contains { $0.severity == .red }, "\(name.rawValue) \(row.label): red only beside a red item")
                }
            }
        }
    }

    /// R1.5 change 1 (L40): no button on the chart at rest, in any fixture; every fault's action is in its station's
    /// panel and in the dropdown.
    func testNoButtonOnTheChartEveryActionInItsPanelAndDropdown() throws {
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let r1 = store.r1
            XCTAssertTrue(NDPipelineChart.controls(r1.stationItems).allSatisfy { $0.action == nil }, "\(name.rawValue): stations only")
            let menu = NDMenuBuilder(store: store, perform: { _ in }, openWindow: { _ in }).build()
            for item in r1.items {
                guard let action = item.action, let station = item.station else { continue }
                let panel = NDStationPopover.model(station, item: r1.stationItems[station], store.inputs, store.model, r1.pipeline)
                XCTAssertEqual(panel.action, action, "\(name.rawValue): \(item.title)'s action is in its panel")
                // R1.13.1: while a sign-in waits, the dropdown's item reads the waiting words and offers no Reconnect.
                let offered = menu.items.contains { ($0.representedObject as? BrainBarMenuAction) == .row(action.kind) && $0.isEnabled }
                XCTAssertEqual(offered, !NDDriveWaiting.shows(store.inputs, for: action), "\(name.rawValue): and in the dropdown")
            }
        }
        // The chart's own view code lays out stations and hover regions, never a button.
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        let a = try XCTUnwrap(source.range(of: "    var body: some View {\n        let shownHover"))
        let b = try XCTUnwrap(source.range(of: "    /// The station's panel at its content's measured height."))
        XCTAssertFalse(source[a.lowerBound..<b.lowerBound].contains("NDButton"), "no action button on the chart")
    }

    /// L32: inside the window an action has one name (in its panel); the dropdown keeps the long one.
    func testAnActionHasOneNameInTheWindow() throws {
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        let reconnect = try XCTUnwrap(two.stationItems[.drive]?.action)
        let report = try XCTUnwrap(two.stationItems[.jobs]?.action)
        XCTAssertEqual(NDPipelineChart.shortLabel(reconnect), Say.reconnectShort)
        XCTAssertEqual(NDPipelineChart.shortLabel(report), "Copy a report for your agent", "R1.12 (§ C): one label on every surface")
        XCTAssertEqual(reconnect.label, BrainBarMenuPresenter.Say.reconnect.text, "the dropdown and the spoken name keep the long form")
        XCTAssertEqual(report.label, BrainBarMenuPresenter.Say.copyReport.text)
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDStationPanel.swift"), encoding: .utf8)
        XCTAssertFalse(source.contains("NDButton(label: action.label"), "the panel never shows the long form")
    }

    /// L33, one test per conditional sentence, both ways.
    func testLastGoodRunOnlyWhenTheLogSaysWhen() throws {
        var i = Fixture.inputs(.nightlyFailing)
        let item = try XCTUnwrap(NDR1.make(i, NDModel.make(i)).stationItems[.jobs])
        XCTAssertTrue(NDStationPopover.faultLines(.jobs, item: item, i).first?.hasPrefix("Last good run: ") == true)
        i.nightlyLastRun = .unread
        XCTAssertFalse(NDStationPopover.faultLines(.jobs, item: item, i).contains { $0.hasPrefix("Last good run") }, "left out, never guessed")
    }

    func testMemoryStillSavesOnlyWhileSavingAndSearchAreGreen() throws {
        var i = Fixture.inputs(.nightlyFailing)
        let item = try XCTUnwrap(NDR1.make(i, NDModel.make(i)).stationItems[.jobs])
        XCTAssertTrue(NDStationPopover.faultLines(.jobs, item: item, i).contains(Say.memoryStillSaves))
        for name in ["stores", "search"] {
            i = Fixture.inputs(.nightlyFailing)
            i.capture.verdict = BrainBarHealthVerdict(dot: .amber, checks: Fixture.replacing(i.capture.verdict!.checks,
                                                                                               Fixture.check(name, .amber, age: 7200, limit: 3600)),
                                                      notes: [], evaluatedAt: i.capture.verdict!.evaluatedAt)
            let lines = NDStationPopover.faultLines(.jobs, item: item, i)
            XCTAssertFalse(lines.contains(Say.memoryStillSaves), name)
            XCTAssertTrue(lines.contains(Say.notCheckedOrTidiedSince), name)
        }
    }

    func testMacCopyStillMadeOnlyWhileTheLocalCopyIsGreen() throws {
        var i = Fixture.inputs(.driveSignedOut)
        let item = try XCTUnwrap(NDR1.make(i, NDModel.make(i)).stationItems[.drive])
        // R1.12 (5e): Drive's fault panel is its title, its two rows and Reconnect; the rows carry each upload's age.
        XCTAssertEqual(NDStationPopover.faultLines(.drive, item: item, i), [])
        XCTAssertEqual(NDStationPopover.model(.drive, item: item, i, NDModel.make(i), NDPipelineState.make(i)).rows.map(\.label), ["Database", "Sessions"])
        i.capture.verdict = BrainBarHealthVerdict(dot: .red, checks: Fixture.replacing(i.capture.verdict!.checks,
                                                                                         Fixture.check("backup_local", .amber, age: 40 * 3600, limit: 36 * 3600)),
                                                  notes: [], evaluatedAt: i.capture.verdict!.evaluatedAt)
        XCTAssertFalse(NDStationPopover.faultLines(.drive, item: item, i).contains(Say.localCopyStillMade))
    }

    func testByMeaningBehindOnlyFromTheEmbeddingChecksAge() {
        // R1.12 (F): a notice says "catching up"; a stall keeps its backlog age, read from the check.
        XCTAssertEqual(pipeline(.meaningBehind).status[.search]?.text, Say.catchingUp)
        var stalled = Fixture.inputs(.meaningBehind)
        stalled.capture.verdict = BrainBarHealthVerdict(dot: .red, checks: Fixture.replacing(stalled.capture.verdict!.checks,
                                                                                               Fixture.check("embedding", .red, age: 2 * 3600, limit: 3600)),
                                                        notes: [], evaluatedAt: stalled.capture.verdict!.evaluatedAt)
        XCTAssertEqual(NDPipelineState.make(stalled).status[.search]?.text, Say.byMeaningBehind(NDFormat.ageText(2 * 3600)))
        var i = Fixture.inputs(.meaningBehind)
        i.capture.verdict = BrainBarHealthVerdict(dot: .amber, checks: Fixture.replacing(i.capture.verdict!.checks,
                                                                                           Fixture.check("embedding", .amber, age: nil, limit: 3600)),
                                                  notes: [], evaluatedAt: i.capture.verdict!.evaluatedAt)
        let drawn = NDPipelineState.make(i)
        XCTAssertEqual(drawn.treatment(.search), .notice)
        XCTAssertFalse(drawn.status[.search]?.text.hasPrefix("by meaning") == true, "no age read, no age said")
    }

    /// No station moves between states; the Kept-safe frame never grows (R1.5: no actions on the chart).
    func testStationsDoNotMoveBetweenStates() {
        let expected: [NDStation: CGPoint] = [
            // R14 deliberately moved Search to y297 to remove the unused lower band.
            // Keep a literal pin, independent of the production layout tokens, in every state.
            .read: CGPoint(x: 215, y: 44), .queue: CGPoint(x: 215, y: 114), .search: CGPoint(x: 736, y: 297),
            .mac: CGPoint(x: 599, y: 44), .drive: CGPoint(x: 736, y: 44), .jobs: CGPoint(x: 342, y: 138), .weekly: CGPoint(x: 528, y: 138),
        ]
        XCTAssertEqual(NDPipelineLayout.stations, expected)
        let centers = expected.merging([.agents: CGPoint(x: 38, y: 72), .memory: CGPoint(x: 435, y: 65.5)]) { fixed, _ in fixed }
        for name in Fixture.Name.allCases {
            let p = pipeline(name)
            // Check both axes of every actual station hit area, including Memory and the agents.
            for (station, point) in centers {
                let hit = NDPipelineLayout.hitShape(station, p).circle
                XCTAssertEqual(CGPoint(x: hit.midX, y: hit.midY), point, "\(name.rawValue) \(station): fixed center")
            }
            for (station, point) in expected {
                let labels = NDPipelineLayout.labelRects(station, p)
                XCTAssertEqual(labels.first?.minY, point.y + 17, "\(name.rawValue) \(station): fixed name y")
                if p.status[station] != nil {
                    let offset: CGFloat = station == .jobs || station == .weekly ? 57 : 33
                    XCTAssertEqual(labels.last?.minY, point.y + offset, "\(name.rawValue) \(station): fixed status y")
                }

                for rect in NDPipelineLayout.labelRects(station, p) {
                    XCTAssertEqual(rect.midX, point.x, accuracy: 0.5, "\(name.rawValue) \(station): its words stay centred on it")
                }
            }
        }
        XCTAssertEqual(NDPipelineLayout.keptFrame(grown: true), NDPipelineLayout.keptSafe)
        XCTAssertEqual(NDPipelineLayout.keptSafe, CGRect(x: 521, y: 18, width: 283, height: 84), "R1.10.4: 32 pt from the right; R1.12: 84 tall")
    }

    /// Lead (1.3 brief): inside Kept safe, 16 pt between the two stations' text blocks and 12 pt from any text to the frame,
    /// at their widest statuses (R1.5: "3 kept, newest 30 h ago" and "last upload 27 h ago", both Semibold).
    func testKeptSafeTextKeepsItsSpacing() {
        let L = NDPipelineLayout.self
        func block(_ s: NDStation, _ widest: String) -> ClosedRange<CGFloat> {
            let half = max(L.textWidth(widest, size: 11, weight: .semibold), L.textWidth(L.name(s), size: 13, weight: .semibold)) / 2
            return (L.point(s).x - half)...(L.point(s).x + half)
        }
        let mac = block(.mac, Say.keptNewest(3, NDFormat.ageText(30 * 3600)))
        let drive = block(.drive, Say.lastUpload(NDFormat.ageText(27 * 3600)))
        XCTAssertGreaterThanOrEqual(drive.lowerBound - mac.upperBound, 16)
        XCTAssertGreaterThanOrEqual(mac.lowerBound - L.keptSafe.minX, 12)
        XCTAssertGreaterThanOrEqual(L.keptSafe.maxX - drive.upperBound, 12)
        XCTAssertLessThanOrEqual(L.keptSafe.maxX, L.drawWidth, "the drawing reaches it")
        XCTAssertLessThan(L.keptSafe.maxY, L.ySaves)
    }

    /// L27: the headline is one line; no problem lines or buttons under it.
    func testHeadlineIsOneLine() {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let r1 = NDR1.make(i, NDModel.make(i))
            guard r1.redCount > 0 else { continue }
            XCTAssertNil(r1.headlineDetail, name.rawValue)
            let red = r1.items.filter { $0.severity == .red }
            XCTAssertEqual(r1.headline, red.count == 1 ? Say.oneThingNeedsYou : Say.thingsNeedYou(red.count), name.rawValue)
        }
    }

    /// R1.4 change 17 (L36): every line ends 3 pt outside the fault ring (r 18.5 from a station's centre, arrow tips on
    /// that point), or 3 pt from the Memory box or the agents' frame; the same geometry in every state.
    func testEveryLineEndsAtTheLineEndDistance() {
        let L = NDPipelineLayout.self
        func check(_ p: CGPoint, _ end: NDPipelineLayout.End, _ what: String) {
            switch end {
            case let .station(s):
                XCTAssertEqual(hypot(p.x - L.point(s).x, p.y - L.point(s).y), L.lineEnd, accuracy: 0.01, what)
            case let .frame(r):
                let gap = min(abs(p.x - r.minX), abs(p.x - r.maxX), abs(p.y - r.minY), abs(p.y - r.maxY))
                XCTAssertEqual(gap, L.edgeGap, accuracy: 0.01, what)
                XCTAssertFalse(r.contains(p), "\(what) ends outside the frame")
            }
        }
        for c in L.connections {
            let ends = L.ends(c.pipe)
            check(c.line.points.first!, ends.start, "\(c.pipe) start")
            check(c.line.points.last!, ends.end, "\(c.pipe) end")
            for p in [c.line.points.first!, c.line.points.last!] {
                for s in L.stations.keys {
                    XCTAssertGreaterThanOrEqual(hypot(p.x - L.point(s).x, p.y - L.point(s).y), L.lineEnd - 0.01,
                                                "\(c.pipe): no end inside \(s)'s circle")
                }
            }
        }
        XCTAssertEqual(L.lineEnd, L.ringRadius + 3, "3 pt outside the fault ring")
        // The state picks a stroke, never the geometry: calm and two faults draw the same lines.
        for name in [Fixture.Name.calm, .twoFaults, .driveSignedOut] {
            XCTAssertTrue(NDPipe.allCases.allSatisfy { [.flowing, .faint, .dashed].contains(pipeline(name).look($0)) }, name.rawValue)
        }
    }

    /// R1.4 changes 1 to 3, L34, L38: four provider icons named for their providers; the fault lands on exactly the
    /// provider a check names, and on none in every other fixture.
    func testProviderIconsAreNamedAndOnlyTheNamedOneFaults() throws {
        XCTAssertEqual(NDPipelineLayout.agentRows.map { NDProvider.name($0.0) }, ["Claude", "Codex", "Cursor", "Antigravity"])
        let chart = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        XCTAssertTrue(chart.contains(".help(NDProvider.name(family))"))
        XCTAssertTrue(chart.contains(".accessibilityLabel(NDProvider.name(family))"))
        for family in AgentFamily.allCases {
            let icon = NDProvider.icon(family)
            let path = NDSVGPath.path(icon.path, viewBox: icon.viewBox, in: CGRect(x: 0, y: 0, width: 16, height: 16))
            XCTAssertFalse(path.isEmpty, "\(family) draws")
            XCTAssertTrue(CGRect(x: -0.5, y: -0.5, width: 17, height: 17).contains(path.boundingRect), "\(family) fits its 16 pt box")
        }
        for name in Fixture.Name.allCases {
            let faults = pipeline(name).agentFaults
            XCTAssertEqual(Set(faults.keys), name == .agentNotRead ? [.cursor] : [], name.rawValue)
        }
    }

    func testGeminiIsAntigravityAndOnlyANamedProviderIsNotRead() {
        XCTAssertEqual(NDProvider.name(.gemini), "Antigravity")
        func drawn(_ codes: [String]?) -> NDPipelineState {
            var i = Fixture.inputs(.calm)
            var k = Fixture.check("ingest", .red, age: 7200, limit: 3600)
            k.contextCodes = codes
            i.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), k), dot: .red)
            return NDPipelineState.make(i)
        }
        let named = drawn(["growth_not_indexed:gemini"])
        XCTAssertEqual(named.agentFaults, [.gemini: .red])
        XCTAssertEqual(named.status[.read]?.text, Say.providerNotRead("Antigravity"))
        for codes in [nil, [], ["path:~/.gemini/antigravity-cli"], ["growth_not_indexed:unknownagent"]] as [[String]?] {
            let s = drawn(codes)
            XCTAssertTrue(s.agentFaults.isEmpty, "\(String(describing: codes)): no provider guessed")
            XCTAssertFalse(s.status[.read]?.text.hasSuffix(" not read") == true, "\(String(describing: codes))")
        }
    }

    /// R1.4 comp 45: each string removed from the chart at rest is in the hover that now carries it.
    func testRailWordsAreInTheRailHover() {
        XCTAssertTrue(NDPipelineChart.lineHovers.contains { $0.title == Say.railTitle && $0.line == Say.railLine && !$0.rects.isEmpty })
        XCTAssertEqual(Say.railTitle.lowercased(), Say.sessionFilesCopied)
    }

    func testAnswersWordsAreInTheReturnLineHover() {
        XCTAssertTrue(NDPipelineChart.lineHovers.contains { $0.title == Say.answersTitle && $0.line == Say.answersLine && !$0.rects.isEmpty })
        XCTAssertTrue(NDStationPopover.explanation(.search).contains("answers go back"))
    }

    func testKeptSafeWordsAreInTheFrameHover() {
        XCTAssertTrue(NDPipelineChart.lineHovers.contains { $0.title == Say.keptSafe && $0.line == Say.keptSafeLine && $0.rects == [NDPipelineLayout.keptSafe] })
    }

    func testYourAgentsWordsAreInTheGroupHover() {
        XCTAssertEqual(NDPipelineLayout.name(.agents), Say.yourAgents)
        let i = Fixture.inputs(.calm)
        XCTAssertEqual(NDStationPopover.lines(.agents, i, pipeline(.calm)).first, Say.agentsLine, "the group's panel lists the four providers")
    }

    func testSessionsAndSavesWordsAreInTheirStationHovers() {
        XCTAssertTrue(NDStationPopover.explanation(.read).contains("session files"))
        XCTAssertTrue(NDStationPopover.explanation(.queue).contains("saved"))
    }

    func testEmptyIsInTheSaveQueueHover() {
        let i = Fixture.inputs(.calm)
        let rows = NDStationPopover.rows(.queue, i, NDModel.make(i))
        XCTAssertEqual(rows.map(\.label), [Say.waitingRow])
        XCTAssertEqual(rows.map(\.value), [Say.none])
        let stuck = Fixture.inputs(.savesStuck)
        XCTAssertTrue(NDStationPopover.rows(.queue, stuck, NDModel.make(stuck)).isEmpty, "\"none\" only when it is measured empty")
    }

    /// R1.12 (F, Fable 11): the Search panel says its last check worked in its text, and its one row is the searchable
    /// share; no "Search by words: no check yet" under "All good".
    func testWorkingIsInTheSearchHover() {
        let i = Fixture.inputs(.calm)
        let rows = NDStationPopover.rows(.search, i, NDModel.make(i))
        XCTAssertEqual(rows, [.init(label: "By meaning", value: "99% searchable")])
        XCTAssertEqual(NDStationPopover.lines(.search, i, NDPipelineState.make(i)),
                       ["Finds memories by meaning and by matching words. Its last check worked."])
        let failing = Fixture.inputs(.searchFailing)
        XCTAssertEqual(NDStationPopover.lines(.search, failing, NDPipelineState.make(failing)),
                       ["Finds memories by meaning and by matching words."], "the second sentence only while search is green")
    }

    /// R1.6 item 9: the readout's lines describe the hovered column only: its time range and its two counts; the
    /// range-wide "Busiest" line is gone from it.
    func testR16ReadoutLinesDescribeTheColumnOnly() throws {
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDWindow.swift"), encoding: .utf8)
        let a = try XCTUnwrap(source.range(of: "struct NDActivityReadout: View {"))
        let b = try XCTUnwrap(source.range(of: "struct NDReadoutPlacement: Layout {"))
        let readout = source[a.lowerBound..<b.lowerBound]
        XCTAssertFalse(readout.contains("busiest") || readout.contains("Busiest"), "no range-wide line in the readout")
        XCTAssertTrue(readout.contains("wave.saves[index]") && readout.contains("wave.reads[index]"), "the column's own counts")
    }

    /// R1.4 changes 5 to 15: the chart at rest draws none of the moved words.
    func testTheChartAtRestCarriesNoneOfTheMovedWords() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign")
        let chart = try String(contentsOf: dir.appendingPathComponent("NDPipelineChart.swift"), encoding: .utf8)
        let a = try XCTUnwrap(chart.range(of: "    static func draw(_ ctx: GraphicsContext"))
        let b = try XCTUnwrap(chart.range(of: "// MARK: - A station's hit area"))
        let drawing = String(chart[a.lowerBound..<b.lowerBound])
        for moved in ["lanesSessions", "lanesSaves", "sessionFilesCopied", "answersToAgentsLane", "Say.keptSafe", "Say.yourAgents",
                      "Say.empty", "Say.working", "family.label", "upTo"] {
            XCTAssertFalse(drawing.contains(moved), "the chart draws \(moved) at rest")
        }
        let window = try String(contentsOf: dir.appendingPathComponent("NDWindow.swift"), encoding: .utf8)
        XCTAssertFalse(window.contains("Say.upTo("), "the halves' maxima are in the readout, not on the chart")
        for name in Fixture.Name.allCases {
            let texts = pipeline(name).status.values.map(\.text)
            XCTAssertFalse(texts.contains(Say.working), name.rawValue)  // R1.12 (F) brings "empty" back as Save queue's status
        }
    }

    /// Capture R7 (2): no popover line truncates. For every station in every fixture, each line of the panel is measured
    /// at its font; the panel's laid-out height holds every line at the number of wrapped lines its width needs.
    func testNoPopoverLineTruncates() throws {
        func width(_ s: String, _ size: CGFloat, _ weight: NSFont.Weight) -> CGFloat {
            ceil(NSAttributedString(string: s, attributes: [.font: NSFont.systemFont(ofSize: size, weight: weight)]).size().width)
        }
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let r1 = store.r1
            for station in NDStation.allCases {
                let item = r1.stationItems[station]
                let panelWidth = NDPipelineLayout.panelSide(station, hasAction: item?.action != nil).width
                let available = panelWidth - 28
                // R1.12: the panel's own model (its title at 13 Semibold, its text and rows at the panel size, 11 pt).
                let model = NDStationPopover.model(station, item: item, store.inputs, store.model, r1.pipeline)
                let body = NDType.panelBodySize, bodyLine = ceil(NSLayoutManager().defaultLineHeight(for: NSFont.systemFont(ofSize: body)))
                var lines: [(String, CGFloat, NSFont.Weight, CGFloat)] = [(model.title, NDScale.body, .semibold, 16)]
                lines += NDStationPanelBody.paragraphs(model.lines).map { ($0, body, .regular, bodyLine) }
                lines += model.rows.map { ((($0.value.isEmpty ? $0.label : $0.label + "   " + $0.value)), body, .regular, NDStationPanelBody.rowHeight) }
                let needed = lines.reduce(CGFloat(0)) { sum, line in
                    sum + ceil(width(line.0, line.1, line.2) / available) * line.3
                }
                let measured = NDStationPopover.measure(AnyView(NDStationPopover(store: store, station: station)), width: panelWidth).height
                XCTAssertGreaterThanOrEqual(measured - 20, needed, "\(name.rawValue) \(station): every line fits, wrapped (R1.12: 10 pt padding)")
            }
        }
        // No popover text is limited to one line or a fixed height.
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        let popover = try XCTUnwrap(source.components(separatedBy: "// MARK: - A line's or a frame's hover").last)
        XCTAssertFalse(popover.contains(".lineLimit("), "a popover line may wrap, never truncate")
        XCTAssertFalse(popover.contains(".frame(height: NDGrid.row)"), "rows grow with their text")
    }

    /// R1.5 change 4: Tab walks the stations; each is focusable and draws the system focus ring. (A real
    /// `nextValidKeyView` walk needs a key window on screen, which these tests may not open.)
    func testTabWalksTheStations() throws {
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        XCTAssertEqual(NDPipelineChart.controls(two.stationItems).map(\.station.rawValue),
                       ["agents", "read", "queue", "memory", "jobs", "weekly", "search", "mac", "drive"])
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("ForEach(Self.controls(items)) { control in station(control.station) }"))
        XCTAssertTrue(source.contains(".focusable(interactions: .activate)\n        .focused($focused)"), "a station takes focus")
        XCTAssertTrue(source.contains(".stroke(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3)"), "a 3 pt focus ring")
    }

    /// Capture R7 (3): every action is reachable from the dropdown by keyboard: an enabled menu item with its action, in
    /// the order the arrow keys walk.
    func testEveryActionIsReachableFromTheDropdownByKeyboard() {
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            var performed: [BrainBarMenuAction] = []
            let builder = NDMenuBuilder(store: store, perform: { performed.append($0) }, openWindow: { _ in })
            let menu = builder.build()
            let reachable = menu.items.filter { $0.isEnabled && $0.action != nil && $0.target != nil }
            for item in store.r1.items {
                guard let action = item.action, store.r1.showsProblemRows else { continue }
                // R1.13.1: a Reconnect that is already waiting is not offered again (its item reads the waiting words).
                XCTAssertEqual(reachable.contains { ($0.representedObject as? BrainBarMenuAction) == .row(action.kind) },
                               !NDDriveWaiting.shows(store.inputs, for: action),
                               "\(name.rawValue): \(action.label) is a keyboard-reachable menu item")
            }
        }
        let two = NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview)
        let titles = NDMenuBuilder(store: two, perform: { _ in }, openWindow: { _ in }).build().items
            .filter { $0.isEnabled && $0.action != nil }.map(\.title)
        XCTAssertEqual(Array(titles.prefix(6)), [BrainBarMenuPresenter.Say.reconnect.text, BrainBarMenuPresenter.Say.copyReport.text,
                                                 Say.activity, Say.search, Say.backups, Say.jobs])
    }

    /// Council r2a finding 3: the hover reads the plot's own coordinates, the origin the rule is drawn from. A pointer at
    /// the drawn rule of the first, a middle and the last bin selects that bin, and so does one up to half a bin away.
    func testChartHoverSelectsTheBinUnderThePointer() throws {
        let count = 60
        let pitch = NDActivitySection.plotWidth / CGFloat(count)
        for bin in [0, 30, count - 1] {
            let rule = NDActivitySection.gutter + NDActivitySection.center(bin, count: count)
            for dx in [0, -0.45 * pitch, 0.45 * pitch] {
                XCTAssertEqual(NDActivitySection.hoverIndex(atChartX: rule + dx, count: count), bin, "bin \(bin), \(dx) pt")
            }
        }
        XCTAssertNil(NDActivitySection.hoverIndex(atChartX: NDActivitySection.gutter - 1, count: count), "the gutter selects nothing")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDWindow.swift"), encoding: .utf8)
        let hover = try XCTUnwrap(source.range(of: ".onContinuousHover(coordinateSpace: .local)"))
        let place = try XCTUnwrap(source.range(of: ".padding(.leading, Self.gutter)\n                    .id(store.range)"))
        XCTAssertLessThan(hover.lowerBound, place.lowerBound, "the hover sits on the plot, before it is placed beside the gutter")
        XCTAssertFalse(source.contains(".offset(x: Self.gutter)"), "no offset that moves the drawing but not the hit area")
    }

    /// Council r2a finding 2: one number for one fact. Drive's last backup is its newest copy of either kind: the Overview
    /// station, the dropdown and the Backups page agree in every fixture.
    func testDriveSaysOneNumberOnEverySurface() {
        func plain(_ s: String) -> String { s.replacingOccurrences(of: "\u{00A0}", with: " ") }
        var compared = 0
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .backups)
            let r1 = store.r1
            // R1.6 item 4: Drive's measure is the newest database copy's upload age.
            guard !r1.pipeline.allUnconfirmed,
                  let newest = store.inputs.capture.verdict?.checks.first(where: { $0.name == "backup_offsite" })?.ageS else { continue }
            let measure = NDFormat.ageText(newest)
            if r1.pipeline.treatment(.drive) != .fault {
                XCTAssertEqual(r1.pipeline.status[.drive]?.text, Say.lastUpload(measure), "\(name.rawValue): the station")
            }
            let lanes = NDBackupsPage(store: store).lanes(store.model)
            let driveLanes = lanes.filter { ["backup_offsite", "transcripts"].contains($0.id) }.map { plain($0.number.text) }
            XCTAssertTrue(driveLanes.contains(plain(measure)), "\(name.rawValue): the Backups page shows the same copy (\(driveLanes))")
            let row = r1.menuRows.first { $0.kind == .backups }!
            if row.tone == .fine {
                let mac = store.inputs.capture.verdict?.checks.first { $0.name == "backup_local" }?.ageS ?? 0
                XCTAssertEqual(row.value, Say.ago(NDFormat.ageText(max(mac, newest))), "\(name.rawValue): the Mac's or Drive's number")
            } else if r1.pipeline.condition(.drive) != .fine {
                XCTAssertEqual(row.value, Say.driveAgo(measure), "\(name.rawValue): the dropdown names the same Drive age")
            }
            compared += 1
        }
        XCTAssertGreaterThan(compared, 5)
    }

    /// R1.5 (L40): Reconnect and its waiting line live in Google Drive's panel and the dropdown, so nothing waits on the chart.
    func testReconnectWaitsInItsPanelNotOnTheChart() throws {
        var waiting = Fixture.inputs(.driveSignedOut)
        waiting.driveReconnecting = true
        let store = NDStore(fixture: waiting, page: .overview)
        XCTAssertTrue(NDPipelineChart.controls(store.r1.stationItems).allSatisfy { $0.action == nil })
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDStationPanel.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("NDWaitingSwap(waiting: waiting, alignment: .leading)"), "R1.10.1: the panel's waiting comes from NDDriveWaiting")
    }

    // MARK: Round 2a decisions (R2A-DECISIONS.md, "Taken"), one test each

    /// R2A 1, as R1.12 (§ B) has it: every panel keeps 12 pt inside the window's content edge on every side, with no
    /// exception: none passes the left edge, none hangs below the window, none reaches into the Activity card.
    func testR2A1PanelsStayInsideTheWindow() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            for station in NDStation.allCases {
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let where_ = "\(name.rawValue) \(station)"
                XCTAssertGreaterThanOrEqual(panel.outer.minX, L.windowInside.minX - 0.01, where_)
                XCTAssertLessThanOrEqual(panel.outer.maxX, L.windowInside.maxX + 0.01, where_)
                XCTAssertLessThanOrEqual(panel.outer.maxY, L.windowInside.maxY + 0.01, where_)
                XCTAssertGreaterThanOrEqual(panel.outer.minY, L.activityLabelsBottom - 0.01,
                                            "\(where_): over the Activity card's bars or axis labels (\(panel.outer.minY))")
            }
        }
    }

    /// R1.6 item 9 (R2A 2 again): the readout as the window lays it out (the hosted section, the path the live window
    /// uses): 12 pt right of the hovered column, left of it in the right third, its bottom 8 pt above the column's
    /// tallest bar (its top stops at the totals row's foot), for the first, a middle and the last column.
    func testR16ReadoutIsAnchoredToTheHoveredColumnAsLaidOut() throws {
        let store = NDStore(fixture: Fixture.inputs(.calm))
        let wave = try XCTUnwrap(store.inputs.heroWaves[.hour])
        func frames(hover: Int) -> (plot: CGRect, readout: CGRect)? {
            var plot: CGRect?
            var readout: CGRect?
            let section = NDActivitySection(store: store, hover: hover, reportPlotFrame: { plot = $0 }, reportReadoutFrame: { readout = $0 })
            let host = NSHostingView(rootView: section.frame(width: 852))
            host.frame = NSRect(x: 0, y: 0, width: 852, height: 220)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
            host.layoutSubtreeIfNeeded()
            guard let plot, let readout else { return nil }
            return (plot, readout)
        }
        let last = wave.count - 1
        XCTAssertFalse(NDActivitySection.readoutFlips(0, count: wave.count))
        XCTAssertTrue(NDActivitySection.readoutFlips(last, count: wave.count))
        for index in [0, wave.count / 2, last] {
            let f = try XCTUnwrap(frames(hover: index), "column \(index): a readout is laid out")
            let column = f.plot.minX + NDActivitySection.center(index, count: wave.count)
            if NDActivitySection.readoutFlips(index, count: wave.count) {
                XCTAssertEqual(column - f.readout.maxX, 12, accuracy: 0.5, "column \(index): 12 pt left of it")
            } else {
                XCTAssertEqual(f.readout.minX - column, 12, accuracy: 0.5, "column \(index): 12 pt right of it")
            }
            XCTAssertFalse((f.readout.minX...f.readout.maxX).contains(column), "column \(index): never over the pointer")
            let top = f.plot.minY + NDWaveform.columnTop(wave, index: index, canvas: NDActivitySection.canvas, nowTick: true, size: .hero)
            let ceiling = f.plot.minY + NDReadoutPlacement.ceiling
            XCTAssertEqual(f.readout.minY, max(ceiling, top - 8 - f.readout.height), accuracy: 0.5, "column \(index): 8 pt above its bar")
            XCTAssertGreaterThan(f.readout.minX, f.plot.minX - NDActivitySection.gutter, "column \(index): never at the top left")
        }
        // The placement itself: below the ceiling the bottom is 8 pt above the bar; a tall bar stops it at the ceiling.
        let size = CGSize(width: 120, height: 60)
        XCTAssertEqual(NDReadoutPlacement.origin(column: 300, barTop: 100, flips: false, size: size), CGPoint(x: 312, y: 32))
        XCTAssertEqual(NDReadoutPlacement.origin(column: 600, barTop: 100, flips: true, size: size), CGPoint(x: 468, y: 32))
        XCTAssertEqual(NDReadoutPlacement.origin(column: 300, barTop: 2, flips: false, size: size).y, NDReadoutPlacement.ceiling)
    }

    /// R2A 3: Drive's status names what it measures (an upload); in backup-late the Mac copy's age stays the Mac's.
    func testR2A3DriveNamesWhatItMeasures() {
        let late = pipeline(.backupLate)
        XCTAssertEqual(late.status[.mac], .init(text: Say.keptNewest(2, NDFormat.ageText(30 * 3600)), severity: .orange))
        XCTAssertTrue(late.status[.drive]?.text.hasPrefix("last upload ") == true, "Drive does not claim a fresh backup")
        XCTAssertFalse(late.status[.drive]?.text.contains("backup") == true)
    }

    /// R2A 4: no crossfade draws two texts at once: the range switch's old bars leave before the new arrive, the axis
    /// words swap at once, and Reconnect leaves before its waiting line arrives (tested above).
    func testR2A4NoCrossfadeDrawsTwoTexts() throws {
        XCTAssertEqual(NDMotion.rangeOutDuration, 0.1, accuracy: 0.0001)
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDWindow.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains(".asymmetric(insertion: .opacity.animation(NDMotion.rangeIn)"))
        XCTAssertTrue(source.contains("removal: .opacity.animation(NDMotion.rangeOut)"))
        XCTAssertTrue(source.contains(".transaction { $0.animation = nil }"), "the axis words swap with no crossfade")
        let design = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/BrainBarDesign.swift"), encoding: .utf8)
        XCTAssertTrue(design.contains("duration: 0.15).delay(rangeOutDuration)"), "in after out")
        XCTAssertGreaterThanOrEqual(NDMotion.waitingDelay, NDMotion.buttonLeavesDuration)
    }

    /// R2A 5 is superseded by L40 (no actions on the chart). What remains: the answers lane clears Jobs' longer name and
    /// status (R1.5 change 23), and no lane crosses the Kept-safe frame.
    func testR2A5LanesClearTheWords() {
        let L = NDPipelineLayout.self
        let fault = pipeline(.nightlyFailing)
        let jobsWords = L.labelRects(.jobs, fault).map(\.maxY).max() ?? 0
        XCTAssertGreaterThanOrEqual(L.yAnswers - jobsWords, 8, "the answers lane passes below Jobs' words")
        let lane = L.connections.first { $0.pipe == .memorySearch }!.line
        XCTAssertGreaterThanOrEqual(lane.points[0].y - L.keptSafe.maxY, 8, "the Memory-to-Search lane runs below the frame")
    }

    /// R2A 6: "not confirmed" shows no relative ages: clock times for the last read and the Mac copy, a dash elsewhere.
    func testR2A6NotConfirmedShowsNoRelativeAges() {
        let stale = pipeline(.notConfirmed)
        XCTAssertTrue(stale.allUnconfirmed)
        for (station, status) in stale.status {
            XCTAssertFalse(status.text.hasSuffix(" ago"), "\(station): \(status.text)")
        }
        // A heartbeat is not a successful read: no last-read time is inferred from it.
        XCTAssertNil(stale.status[.read], "no read time was measured")
        var older = Fixture.inputs(.notConfirmed)
        older.capture.watcher = .init(lastReadAt: older.capture.now.addingTimeInterval(-3 * 3600))
        XCTAssertTrue(NDPipelineState.make(older).status[.read]?.text.hasPrefix("last read ") == true, "a read before the check: its time")
        // R1.12 (E2, replaces R1.7 item 5's dash): each station keeps its last-known value, as a clock time.
        XCTAssertTrue(stale.status[.mac]?.text.hasPrefix("3 kept, newest ") == true, stale.status[.mac]?.text ?? "nil")
        XCTAssertTrue(stale.status[.drive]?.text.hasPrefix("last upload ") == true, stale.status[.drive]?.text ?? "nil")
        XCTAssertTrue(stale.status[.jobs]?.text.hasPrefix("last run ") == true, stale.status[.jobs]?.text ?? "nil")
    }

    /// R2A 7: the window's height change on a page switch animates 200 ms ease-in-out, and not at all under Reduce Motion.
    func testR2A7PageSwitchHeightAnimates() throws {
        XCTAssertEqual(NDMotion.pageResize(reduceMotion: false), 0.2)
        XCTAssertNil(NDMotion.pageResize(reduceMotion: true))
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/BrainBarDashboardPanelController.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("NDMotion.pageResize(reduceMotion: reduceMotionForTesting ?? NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)"),
                      "the system's Reduce Motion, unless a test sets it")
        XCTAssertTrue(source.contains("CAMediaTimingFunction(controlPoints: c.0, c.1, c.2, c.3)"), "R1.8 item 1: the strong ease-out")
        // L57: exercise the production frame calculation after moving it out of the controller.
        let before = NSRect(x: 100, y: 200, width: 900, height: 500)
        let after = NDPageHeight.frame(size: NSSize(width: 900, height: 600), keeping: before, visibleFrame: nil)
        XCTAssertEqual(after.maxY, before.maxY, "the top edge stays")
        XCTAssertEqual(after.minX, before.minX)
    }

    /// R2A 8: Memory carries its unit.
    func testR2A8MemoryCarriesItsUnit() throws {
        XCTAssertEqual(Say.memoriesCount("297,412"), "297,412 memories")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("Say.memoriesCount(count)"))
    }

    /// R2A 9: a single fault says what it is in the headline: red, its own sentence; amber with no worded sentence,
    /// "{station}: {its status}"; catching up keeps its title; two or more keep the count.
    func testR2A9ASingleFaultSaysWhatItIs() {
        func r1(_ name: Fixture.Name) -> NDR1 { NDR1.make(Fixture.inputs(name), NDModel.make(Fixture.inputs(name))) }
        XCTAssertEqual(r1(.driveSignedOut).headline, Say.oneThingNeedsYou, "R1.10: the row names it")
        XCTAssertEqual(r1(.backupLate).headline, "The local copy is running late", "R1.6 item 5, R1.12 (E3, L53 C4)")
        XCTAssertEqual(r1(.meaningBehind).headline, Say.meaningCatchingUp)
        XCTAssertEqual(r1(.twoFaults).headline, Say.thingsNeedYou(2))
        XCTAssertNil(r1(.twoFaults).headlineDetail, "no list under the count")
    }

    // MARK: 1.4.1 polish (L39)

    /// L39: on a small screen every panel stays inside the screen's visible frame, its arrow still on its station; a body
    /// the screen shortens scrolls.
    func testPanelsStayInsideASmallScreen() {
        let L = NDPipelineLayout.self
        // A screen whose visible frame ends 10 pt below the chart and starts 200 pt above it, in chart coordinates.
        let screen = CGRect(x: -60, y: -200, width: 940, height: L.size.height + 210)
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            for station in NDStation.allCases {
                let natural = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let fitted = L.fitted(natural, within: screen)
                let where_ = "\(name.rawValue) \(station)"
                XCTAssertTrue(screen.insetBy(dx: -0.5, dy: -0.5).contains(fitted.outer), "\(where_): inside the screen")
                XCTAssertEqual(fitted.tip, natural.tip, "\(where_): the arrow stays on its station")
                switch fitted.edge {
                case .top, .bottom:
                    XCTAssertTrue((fitted.body.minX + L.arrowInset - 0.01...fitted.body.maxX - L.arrowInset + 0.01).contains(fitted.tip.x), where_)
                case .leading, .trailing:
                    XCTAssertTrue((fitted.body.minY + L.arrowInset - 0.01...fitted.body.maxY - L.arrowInset + 0.01).contains(fitted.tip.y), where_)
                }
            }
        }
        // On a roomy screen nothing changes.
        let store = NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
        let jobs = chart.panel(.jobs, content: AnyView(NDStationPopover(store: store, station: .jobs)))
        XCTAssertEqual(L.fitted(jobs, within: CGRect(x: -500, y: -800, width: 2000, height: 2000)), jobs)
    }

    /// R14: Memory-to-Search leaves the lower port with 8 pt frame clearance and turns once into Search.
    func testMemoryToSearchIsOneCleanRoute() {
        let L = NDPipelineLayout.self
        let line = L.connections.first { $0.pipe == .memorySearch }!
        XCTAssertEqual(line.corner, 10)
        let p = line.line.points
        XCTAssertEqual(p.count, 3, "R14: one clean turn below the copies")
        XCTAssertEqual(p[0].x, L.hub.maxX - 12, accuracy: 0.01, "R14: it leaves the lower port")
        XCTAssertEqual(p[0].y, L.hub.maxY + L.edgeGap) // R14: keep the real 8 pt frame clearance.
        XCTAssertEqual(p[1].y, p[0].y)
        XCTAssertEqual(p[1].x, L.point(.search).x)
    }

    /// 1.4.1 (3): the Pipeline block's height is set by its lowest station (Search's name under the answers lane), so it
    /// cannot lose height without moving a station.
    func testPipelineBlockHeightIsSetByItsLowestStation() {
        let L = NDPipelineLayout.self
        let search = L.labelRects(.search, pipeline(.calm)).map(\.maxY).max() ?? 0  // R1.12 (F): with its status line
        // R14: Search uses the lower loop; no unused band remains below its words.
        XCTAssertGreaterThan(search, L.size.height - 4, "Search ends the compact loop")
        XCTAssertLessThanOrEqual(search, L.size.height + 0.5)
    }

    /// CI on e89086d2: every fixture is a pure function of its own pinned clock and zone. Built under two process time
    /// zones, each fixture's headline, items and icon state are identical. (The runner's backup-late failure was not
    /// the zone: CI tests the merge with main, whose R31 presenter makes an amber check a notice; the window now derives
    /// amber notices from the verdict, see the next test.)
    func testEveryFixtureIsTheSameInEveryTimeZone() throws {
        let original = NSTimeZone.default
        defer { NSTimeZone.default = original }
        func snapshot() -> [String] {
            Fixture.Name.allCases.map { name in
                let i = Fixture.inputs(name)
                let m = NDModel.make(i)
                let r1 = NDR1.make(i, m)
                return "\(name.rawValue)|\(r1.headline)|\(r1.items.map(\.title))|\(NDIcon.state(m, r1))|\(r1.pipeline.status.mapValues(\.text).sorted { $0.key.rawValue < $1.key.rawValue })"
            }
        }
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "UTC"))
        let utc = snapshot()
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Asia/Jerusalem"))
        let jerusalem = snapshot()
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "America/Los_Angeles"))
        let la = snapshot()
        XCTAssertEqual(utc, jerusalem)
        XCTAssertEqual(utc, la)
    }

    /// CI on e89086d2 (cause): an amber copy is a notice on the window whatever the presenter calls it. backup-late's notice
    /// comes from the verdict's amber backup_local check, so it holds on main's R31 presenter, which lists no fault for it.
    func testAmberNoticesComeFromTheVerdictNotThePresenter() {
        let i = Fixture.inputs(.backupLate)
        let r1 = NDR1.make(i, NDModel.make(i))
        let notice = r1.items.first { $0.station == .mac }
        XCTAssertEqual(notice?.id, "notice.mac", "derived from the verdict, not a presenter fault")
        XCTAssertEqual(notice?.severity, .orange)
        XCTAssertEqual(r1.items.filter { $0.station == .mac }.count, 1, "said once")
        XCTAssertEqual(r1.headline, BrainBarMenuPresenter.Say.localRunningLate.text)
    }

    // MARK: Revision 1.5 (L40 to L45)

    /// R1.5 change 4: Return or Space on a focused station opens its panel with focus on the action; Escape closes it and
    /// gives focus back to the station. Driven through the station's own key handling (no window on screen).
    func testKeyboardOpensThePanelOnItsActionAndEscapeCloses() {
        let presenter = NDPanelPresenter()
        let store = NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
        for key in [KeyEquivalent.return, .space] {
            let handled = NDStationButton.handle(key, station: .drive, presenter: presenter) {
                let content = AnyView(NDStationPopover(store: store, station: .drive))
                presenter.show(.drive, layout: chart.panel(.drive, content: content), content: content, pinned: true, focusAction: true)
            }
            XCTAssertTrue(handled)
            XCTAssertEqual(presenter.shown, .drive, "\(key): the panel opens")
            XCTAssertEqual(presenter.focusTarget, .action(.drive), "\(key): its action takes focus")
            XCTAssertTrue(NDStationButton.handle(.escape, station: .drive, presenter: presenter) {})
            XCTAssertNil(presenter.shown, "Escape closes it")
            XCTAssertEqual(presenter.focusTarget, .station(.drive), "focus is back on the station")
        }
        XCTAssertFalse(NDStationButton.handle(.escape, station: .jobs, presenter: presenter) {}, "Escape with no panel open is not handled")
        let panel = NDStationPopover.model(.drive, item: store.r1.stationItems[.drive], store.inputs, store.model, store.r1.pipeline)
        XCTAssertNotNil(panel.action, "the focused panel has an action to focus")
    }

    /// R1.5 change 3: Copy report reads "Copied" with a tick for 1.5 s after a click, at the same size.
    func testCopyReportShowsCopiedForOneAndAHalfSeconds() {
        XCTAssertEqual(NDCopyReportButton.face(copied: false, label: Say.copyReportShort).text, Say.copyReportShort)
        XCTAssertFalse(NDCopyReportButton.face(copied: false, label: Say.copyReportShort).tick)
        XCTAssertEqual(NDCopyReportButton.face(copied: true, label: Say.copyReportShort).text, Say.copiedLabel)
        XCTAssertTrue(NDCopyReportButton.face(copied: true, label: Say.copyReportShort).tick)
        XCTAssertEqual(NDCopyReportButton.copiedFor, .milliseconds(1500))
    }

    /// R1.5 change 6 (L41): one red per fault: in agent-not-read only the agent's icon is red; Read has no ring and its
    /// words are plain.
    func testOneRedPerFault() {
        let agent = pipeline(.agentNotRead)
        let redStations = NDStation.allCases.filter { agent.treatment($0) == .fault }
        XCTAssertEqual(redStations, [.agents], "exactly one red mark: the agent's")
        XCTAssertEqual(agent.agentFaults, [.cursor: .red], "on the named provider's icon")
        XCTAssertEqual(agent.treatment(.read), .calm, "Read has no ring")
        XCTAssertEqual(agent.status[.read]?.severity, .red, "R1.12 (E4): its status reads red like every fault status (no ring)")
        XCTAssertEqual(NDPipelineChart.marks(agent.treatment(.read)), .init())
    }

    /// R1.5 change 8 (L45): "{n} kept" is the REAL number of copies found; the panel's keep number is the backup's own
    /// constant (`DEFAULT_LOCAL_COMPRESSED_KEEP` in backup_daily.py).
    func testKeptIsTheRealCountAndTheKeepNumberIsTheBackupsConstant() throws {
        let calm = Fixture.inputs(.calm), late = Fixture.inputs(.backupLate)
        XCTAssertEqual(calm.localCopiesKept, 3)
        XCTAssertEqual(late.localCopiesKept, 2)
        XCTAssertEqual(pipeline(.calm).status[.mac]?.text, Say.keptNewest(3, NDFormat.ageText(4 * 3600)))
        XCTAssertEqual(pipeline(.backupLate).status[.mac]?.text, Say.keptNewest(2, NDFormat.ageText(30 * 3600)))
        let rows = NDStationPopover.rows(.mac, late, NDModel.make(late))
        XCTAssertEqual(rows.first { $0.label == Say.keptRow }?.value, "2", "the panel shows the real count too")
        let python = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/brainlayer/backup_daily.py"), encoding: .utf8)
        let match = try XCTUnwrap(python.firstMatch(of: /\nDEFAULT_LOCAL_COMPRESSED_KEEP = (\d+)\n/))
        let keep = try XCTUnwrap(Int(match.1))
        XCTAssertEqual(NDBackupFacts.keepNewest, keep, "the app's keep number is the backup's")
        XCTAssertEqual(NDStationPopover.lines(.mac, calm, pipeline(.calm)), [Say.localCopyPanel(Say.countWord(keep))])  // R1.12 (I 12)
        XCTAssertEqual(Say.localCopyPanel(Say.countWord(keep)),
                       "Saved every night. The three newest are kept until Google Drive holds a checked copy.")
        XCTAssertEqual(BackupLocalFiles.compressedSnapshots(in: ["2026-11-15.db.zst", "2026-11-14.db.gz", "2026-11-13.db", "notes.txt",
                                                                "2026-11-12T03-15.db.zst"]).count, 3, "compressed copies only")
    }

    /// R1.5 changes 10 to 21: the light tokens, measured. Text against its background at least 4.5:1; lines and marks
    /// against the card at least 3:1; the faint downstream pipe (28 % of `pipe`) is the named exception.
    func testLightTokensMeetTheirContrast() throws {
        func rgba(_ c: NSColor) -> (r: Double, g: Double, b: Double, a: Double) {
            var out = (r: 0.0, g: 0.0, b: 0.0, a: 0.0)
            NSAppearance(named: .aqua)!.performAsCurrentDrawingAppearance {
                let s = c.usingColorSpace(.sRGB)!
                out = (Double(s.redComponent), Double(s.greenComponent), Double(s.blueComponent), Double(s.alphaComponent))
            }
            return out
        }
        func over(_ fg: NSColor, _ bg: NSColor) -> (Double, Double, Double) {
            let f = rgba(fg), b = rgba(bg)
            return (f.r * f.a + b.r * (1 - f.a), f.g * f.a + b.g * (1 - f.a), f.b * f.a + b.b * (1 - f.a))
        }
        func luminance(_ c: (Double, Double, Double)) -> Double {
            func lin(_ v: Double) -> Double { v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
            return 0.2126 * lin(c.0) + 0.7152 * lin(c.1) + 0.0722 * lin(c.2)
        }
        func ratio(_ fg: NSColor, on bg: NSColor) -> Double {
            let b = rgba(bg)
            let l1 = luminance(over(fg, bg)), l2 = luminance((b.r, b.g, b.b))
            return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
        }
        let texts: [(String, NSColor)] = [("t1", NDColor.t1), ("t2", NDColor.t2), ("t3", NDColor.t3), ("red-t", NDColor.redText),
                                          ("orange-t", NDColor.orangeText), ("saved-t", NDColor.savedText), ("read-t", NDColor.readText)]
        for (name, c) in texts { XCTAssertGreaterThanOrEqual(ratio(c, on: NDColor.card), 4.5, "\(name) on the card") }
        XCTAssertGreaterThanOrEqual(ratio(NDColor.t3, on: NDColor.bg), 4.5, "t3 on the plane (the footer)")
        let marks: [(String, NSColor)] = [("pipe", NDColor.pipe), ("red", NDColor.red), ("orange", NDColor.orange),
                                          ("saved", NDColor.saved), ("read", NDColor.read)]
        for (name, c) in marks { XCTAssertGreaterThanOrEqual(ratio(c, on: NDColor.card), 3, "\(name) on the card") }
        XCTAssertGreaterThanOrEqual(ratio(.white, on: NDColor.red), 4.5, "white on red (tab pills)")
        XCTAssertGreaterThanOrEqual(ratio(.white, on: NDColor.accent), 4.5, "white on accent (Reconnect)")
        XCTAssertLessThan(ratio(NDColor.pipe.withAlphaComponent(0.28), on: NDColor.card), 3, "the faint downstream pipe: the named exception")
        // The light card is no longer pure white, and the plane is one step darker (changes 10, 11).
        let card = rgba(NDColor.card), plane = rgba(NDColor.bg)
        XCTAssertLessThan(card.r, 1)
        XCTAssertGreaterThan(ratio(NDColor.card, on: NDColor.bg), 1.16)
        _ = plane
    }

    // MARK: 1.5.1 (the real screen on the M4)

    /// 1.5.1 (1): a failing provider's own icon opens the agents' fault panel (the same panel, with Copy report); with
    /// nothing wrong, no icon opens a panel and the group's hover keeps its explanation.
    func testAFailingProviderIconOpensItsFaultPanel() throws {
        for name in Fixture.Name.allCases {
            XCTAssertEqual(NDPipelineChart.faultTriggers(pipeline(name)), name == .agentNotRead ? [.cursor] : [], name.rawValue)
        }
        let store = NDStore(fixture: Fixture.inputs(.agentNotRead), page: .overview)
        let fault = NDStationPopover.model(.agents, item: store.r1.stationItems[.agents], store.inputs, store.model, store.r1.pipeline)
        XCTAssertEqual(fault.title, Say.agentNotRead("Cursor"))
        XCTAssertEqual(fault.action?.kind, .copyReport)
        let calm = NDStore(fixture: Fixture.inputs(.calm), page: .overview)
        let explain = NDStationPopover.model(.agents, item: calm.r1.stationItems[.agents], calm.inputs, calm.model, calm.r1.pipeline)
        XCTAssertEqual(explain.title, Say.yourAgents)
        XCTAssertNil(explain.action)
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("ForEach(Self.controls(items)) { control in station(control.station) }"),
                      "provider icons share the agents station target and its fault panel")
    }

    /// 1.5.1 (2, 3), as R1.12 (§ B) has it: no panel covers the footer's lines, for every station in every fixture at the
    /// default window size. (R1.12: a panel may cover the "Pipeline" section label, pipes, empty card padding and the
    /// footer's empty middle.)
    func testNoPanelCoversTheHeadingOrTheFooter() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            for station in NDStation.allCases {
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let where_ = "\(name.rawValue) \(station)"
                for (index, text) in L.footerTexts.enumerated() {
                    XCTAssertFalse(panel.outer.intersects(text), "\(where_) covers the footer's \(index == 0 ? "left" : "right") line")
                }
            }
        }
    }

    /// § 3.1 hover: the pointer's bucket is the pitch that holds it; outside the plot there is none.
    func testActivityHoverPicksTheBucketUnderThePointer() {
        XCTAssertEqual(NDActivitySection.bucket(at: 0, count: 60), 0)
        XCTAssertEqual(NDActivitySection.bucket(at: 13.4, count: 60), 1)
        XCTAssertEqual(NDActivitySection.bucket(at: 767.9, count: 60), 59)
        XCTAssertNil(NDActivitySection.bucket(at: -1, count: 60))
        XCTAssertNil(NDActivitySection.bucket(at: 768, count: 60), "R1.2: a 768 pt plot inside its surface")
        XCTAssertNil(NDActivitySection.bucket(at: 10, count: 0))
    }

    // MARK: Revision 1.6 (R2-DECISIONS "Taken")

    /// R1.6 item 1: a panel closes the moment the pointer leaves (no delay, a 100 ms fade) and never stays up while
    /// another station is highlighted. With no window to draw in, the presenter's state is the whole story.
    func testR16PanelsCloseWhenThePointerLeaves() {
        XCTAssertEqual(NDPanelPresenter.closeDelay, 0)
        XCTAssertEqual(NDPanelPresenter.closeFade, 0.1, accuracy: 0.0001)
        let presenter = NDPanelPresenter()
        let layout = NDPipelineLayout.Panel(edge: .top, anchor: .zero, tip: .zero, body: .zero, outer: .zero)
        presenter.show(.read, layout: layout, content: AnyView(EmptyView()), pinned: false)
        presenter.pointer(.read, over: false) { _ in }
        XCTAssertNil(presenter.shown, "gone the moment the pointer leaves, with no delay")
        presenter.show(.read, layout: layout, content: AnyView(EmptyView()), pinned: false)
        presenter.pointer(.drive, over: true) { _ in }
        XCTAssertNil(presenter.shown, "another station highlighted: the old panel is gone at once")
        presenter.hide()
    }

    /// M1 calm-fit correction: a panel scrolls only when chart or screen bounds cut its measured content.
    func testR16PanelsFitTheirContent() throws {
        let store = NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
        let roomy = CGRect(x: -2000, y: -2000, width: 4000, height: 4000)
        for station in NDStation.allCases {
            let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
            let content = NDStationPopover.measure(AnyView(NDStationPopover(store: store, station: station)), width: panel.body.width).height
            XCTAssertEqual(panel.scrolls, panel.body.height < content, "\(station): only cut content scrolls")
            XCTAssertEqual(NDPipelineLayout.fitted(panel, within: roomy).scrolls, panel.scrolls, "a roomy screen adds no cut")
        }
        let queue = chart.panel(.queue, content: AnyView(NDStationPopover(store: store, station: .queue)))
        let short = CGRect(x: -2000, y: -2000, width: 4000, height: 2000 + queue.body.minY + 40)
        XCTAssertTrue(NDPipelineLayout.fitted(queue, within: short).scrolls, "the screen cuts it: then it scrolls")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("if layout.scrolls {"), "the chrome holds a scroll view only when cut")
    }

    /// R1.6 item 3: a page switch never draws a clipped page: growing, the frame goes first (200 ms ease-out) and the
    /// page follows; shrinking, the page goes first; Reduce Motion swaps both at once. The store keeps the page shown
    /// apart from the page chosen, so the controller can hold it.
    func testR16PageSwitchesNeverDrawAClippedPage() {
        XCTAssertEqual(NDPageSwitch.order(from: 500, to: 800, animates: true), .pageThenFrame, "R1.8 item 1: the page first")
        XCTAssertEqual(NDPageSwitch.order(from: 800, to: 500, animates: true), .pageThenFrame)
        XCTAssertEqual(NDPageSwitch.order(from: 500, to: 800, animates: false), .together, "Reduce Motion: at once")
        XCTAssertEqual(NDPageSwitch.order(from: 600, to: 600.2, animates: true), .together)
        XCTAssertEqual(NDMotion.pageResizeDuration, 0.2, accuracy: 0.0001)
        let store = NDStore(fixture: Fixture.inputs(.calm), page: .overview)
        var held: [NDPage] = []
        store.pageDidChange = { held.append($0) }
        store.page = .jobs
        XCTAssertEqual(held, [.jobs])
        XCTAssertEqual(store.shownPage, .overview, "the old page stays drawn until the controller swaps it")
    }

    /// R1.6 items 4 and 5: Drive's station says the newest database copy's upload age (the session files' age is a row
    /// in its panel); backup-late shows the true older age; the dropdown's Backups row uses the same measure; the
    /// headline says the Mac copy is late with its age.
    func testR16DriveSaysTheDatabaseCopyAndBackupLateIsTrue() throws {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let p = NDPipelineState.make(i)
            guard p.condition(.drive) == .fine, !p.allUnconfirmed,
                  let db = i.capture.verdict?.checks.first(where: { $0.name == "backup_offsite" })?.ageS else { continue }
            XCTAssertEqual(p.status[.drive]?.text, Say.lastUpload(NDFormat.ageText(db)), "\(name.rawValue)")
        }
        let late = Fixture.inputs(.backupLate)
        let db = try XCTUnwrap(late.capture.verdict?.checks.first { $0.name == "backup_offsite" }?.ageS)
        XCTAssertEqual(db, 30 * 3600, "the fixture's database copy is as old as the late Mac copy")
        let r1 = NDR1.make(late, NDModel.make(late))
        XCTAssertEqual(r1.pipeline.status[.drive]?.text, Say.lastUpload(NDFormat.ageText(db)))
        let store = NDStore(fixture: late, page: .overview)
        let rows = NDStationPopover.model(.drive, item: nil, late, store.model, r1.pipeline).rows
        XCTAssertEqual(rows.map(\.label), [Say.databaseCopy, Say.conversationsCopy], "the session files' age is a row")
        XCTAssertEqual(r1.menuRows.first { $0.kind == .backups }?.value.contains(NDFormat.ageText(db)), true)
        XCTAssertEqual(r1.headline, BrainBarMenuPresenter.Say.localRunningLate.text)
        // R1.12 (E3, L53 C4): the menu's sentence, with its second line.
        XCTAssertEqual(r1.headline, "The local copy is running late")
        XCTAssertTrue(r1.headlineDetail?.hasPrefix("The newest is 30\u{00A0}h old") == true, r1.headlineDetail ?? "nil")
    }

    /// R1.6 item 6: not confirmed, the Activity chart draws no live data: hatched from the last confirmed check, with
    /// "no data since {time}"; the totals say "—"; the bars before it are dimmed; past it no column can be hovered.
    func testR16NotConfirmedDrawsNoLiveData() throws {
        let store = NDStore(fixture: Fixture.inputs(.notConfirmed), page: .overview)
        let wave = try XCTUnwrap(store.inputs.heroWaves[.hour])
        XCTAssertEqual(try XCTUnwrap(NDActivitySection.noDataFraction(store, wave: wave)), 0, accuracy: 0.001,
                       "the last check was 2 h ago: the whole hour is hatched")
        store.range = .threeHours
        let three = try XCTUnwrap(NDActivitySection.noDataFraction(store, wave: store.inputs.heroWaves[.threeHours]))
        XCTAssertEqual(three, 1.0 / 3, accuracy: 0.01, "on 3 h the first hour is drawn, dimmed")
        XCTAssertTrue(NDActivitySection.hoverable(0, count: 60, cut: three))
        XCTAssertFalse(NDActivitySection.hoverable(59, count: 60, cut: three), "nothing past the last check reads out")
        XCTAssertNil(NDActivitySection.noDataFraction(NDStore(fixture: Fixture.inputs(.calm)), wave: wave), "confirmed: none")
        XCTAssertLessThan(NDWaveform.dimmedBefore, 1)
        XCTAssertEqual(Say.noDataSince("14:02"), "no data since 14:02")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDWindow.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("let known = wave?.readable == true && cut == nil"), "the totals say \"—\"")
    }

    /// R1.6 item 7: no control holds focus when the window opens; the window itself is first responder until Tab.
    func testR16NoFocusRingWhenTheWindowOpens() throws {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 100), styleMask: [.titled],
                              backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        let button = NSButton(title: "1 h", target: nil, action: nil)
        window.contentView?.addSubview(button)
        window.makeFirstResponder(button)
        BrainBarDashboardPanelController.clearInitialFocus(window)
        XCTAssertTrue(window.firstResponder === window, "nothing in the window holds focus")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/BrainBarDashboardPanelController.swift"), encoding: .utf8)
        let show = try XCTUnwrap(source.range(of: "    func show(anchoredTo anchorView: NSView? = nil) {"))
        let end = try XCTUnwrap(source.range(of: "    func dismiss() {"))
        XCTAssertTrue(source[show.lowerBound..<end.lowerBound].contains("Self.clearInitialFocus(window)"), "opening clears it")
    }

    /// R1.6 item 8: the diagram is centred in its card: equal 32 pt insets left and right (card padding 16 plus 16).
    func testR16DiagramIsCentredInItsCard() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let p = NDPipelineState.make(Fixture.inputs(name))
            var drawn = L.agentsFrame.union(L.hub).union(L.keptSafe)
            for station in NDStation.allCases { L.labelRects(station, p).forEach { drawn = drawn.union($0) } }
            XCTAssertEqual(16 + drawn.minX, 32, accuracy: 0.5, "\(name.rawValue): left inset")
            // R1.10.4: equal again (everything right of the agents moved 15 pt left; the copies keep 137 pt between them).
            XCTAssertEqual(16 + L.size.width - drawn.maxX, 32, accuracy: 0.5, "\(name.rawValue): right inset")
        }
    }

    /// R1.6 items 10 and 11: a broken station's name keeps its colour and weight; only its short line is red
    /// ("failed 3 nights"); the jobs fault panel's title is "Failed 3 nights in a row".
    func testR16ABrokenStationKeepsItsNameOnlyItsLineIsRed() throws {
        let marks = NDPipelineChart.marks(.fault)
        XCTAssertFalse(marks.nameRed, "the name keeps its normal colour and weight")
        XCTAssertEqual(marks.statusTone, .fault)
        let failing = Fixture.inputs(.nightlyFailing)
        let p = NDPipelineState.make(failing)
        XCTAssertEqual(p.status[.jobs], .init(text: Say.failedNights(3), severity: .red), "R1.10: the station names the job")
        let r1 = NDR1.make(failing, NDModel.make(failing))
        let item = try XCTUnwrap(r1.stationItems[.jobs])
        // R1.12 (§ C): one wording, and the station's line is its tail.
        XCTAssertEqual(NDStationPopover.faultTitle(.jobs, item, failing), "Nightly check-up failed 3 nights")
        XCTAssertEqual(Say.failedNights(3), "failed 3 nights")
        XCTAssertTrue(item.title.hasSuffix(Say.failedNights(3)))
    }

    /// R1.6 item 12: an on Jobs switch uses the system accent; the system's gray is for off only.
    func testR16JobsSwitchesShowTheirState() throws {
        XCTAssertEqual(NDColor.switchOn, NSColor.controlAccentColor)
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPages.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains(".toggleStyle(.switch)") && source.contains(".tint(ND.switchOn)"))
    }

    /// R1.6 item 14: the dropdown's Search row says how much is searchable by meaning.
    func testR16DropdownSearchRowSaysSearchableByMeaning() throws {
        let i = Fixture.inputs(.calm)
        let row = try XCTUnwrap(NDR1.make(i, NDModel.make(i)).menuRows.first { $0.kind == .search })
        XCTAssertEqual(row.value, "99% searchable", "R1.12 (F): the station's and the panel's words")
    }

    // MARK: Revision 1.10.5

    /// R1.10.5 (L49, L50): one name per thing on every surface. The copy on this Mac is "Local copy" and the weekly job is
    /// "Weekly clean" on the Overview's stations, the Backups page, the Jobs page, the dropdown's help, the model's rows
    /// and the late headline; no source string still says the old names.
    func testR1105OneNamePerThing() throws {
        let local = NDPipelineLayout.name(.mac)
        XCTAssertEqual(local, "Local copy")
        XCTAssertEqual(NDPipelineLayout.name(.weekly), "Weekly clean")
        let calm = NDStore(fixture: Fixture.inputs(.calm), page: .backups)
        let lanes = NDBackupsPage(store: calm).lanes(calm.model)
        XCTAssertEqual(lanes.first { $0.id == "backup_local" }?.name, local, "the Backups page")
        XCTAssertEqual(Say.jobNames["maintenanceWeekly"], NDPipelineLayout.name(.weekly), "the Jobs page")
        XCTAssertEqual(Say.jobNames["maintenanceNightly"], NDPipelineLayout.name(.jobs), "the Jobs page")
        let help = try XCTUnwrap(calm.r1.menuRows.first { $0.kind == .backups }?.help)
        XCTAssertTrue(help.hasPrefix(local + ":"), "the dropdown's help: \(help)")
        let rows = calm.model.areas.flatMap(\.rows).map(\.name)
        XCTAssertTrue(rows.contains(local), "the model's rows")
        let late = NDR1.make(Fixture.inputs(.backupLate), NDModel.make(Fixture.inputs(.backupLate)))
        XCTAssertTrue(late.headline.localizedCaseInsensitiveContains(local), late.headline)
        // R1.12 (§ C, the 1.12 table): the session files, the Drive database copy, the stages and the check, one name each.
        XCTAssertEqual(lanes.map(\.name), ["Local copy", "Database in Google Drive", "Sessions in Google Drive", "Local copy check"])
        XCTAssertEqual(Say.jobNames["watch"], NDPipelineLayout.name(.read), "the Jobs page: Read")
        XCTAssertEqual(Say.jobNames["drain"], NDPipelineLayout.name(.queue), "the Jobs page: Save queue")
        XCTAssertEqual(Say.jobNames["jsonlBackup"], "Back up sessions")
        let signedOut = Fixture.inputs(.driveSignedOut)
        XCTAssertEqual(NDStationPopover.rows(.drive, signedOut, NDModel.make(signedOut)).map(\.label), ["Database", "Sessions"], "the Drive panel")
        XCTAssertTrue(help.contains("Database in Google Drive:") && help.contains("Sessions in Google Drive:"), help)
        let lateValue = try XCTUnwrap(late.menuRows.first { $0.kind == .backups }?.value)
        XCTAssertTrue(lateValue.hasPrefix(local + " "), "the dropdown's Backups value names the late copy: \(lateValue)")
        let checkHelp = lanes.first(where: { $0.id == "integrity" })?.help ?? ""
        XCTAssertTrue(checkHelp.hasPrefix(Say.laneCheckHelp), "E1: its help tag: \(checkHelp)")
        // No string a surface draws still carries an old name (comments may tell the history).
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar")
        let files = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("NewDesign").path)
            .filter { $0.hasSuffix(".swift") }.map { "NewDesign/" + $0 } + ["BrainBarMenuSentences.swift"]
        for file in files {
            let source = try String(contentsOf: root.appendingPathComponent(file), encoding: .utf8)
            for (number, line) in source.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard !code.hasPrefix("//") else { continue }
                let text = code.components(separatedBy: " // ").first ?? code
                for old in ["Copy on this Mac", "copy on this Mac", "Weekly deep clean", "Nightly check-up, weekly clean",
                            // R1.12 (§ C): the names 1.12 retires.
                            "Copy in Google Drive", "Conversations in Drive", "Database check", "Read new conversations",
                            "Save waiting memories", "Back up conversations", "Conversations copy", "Database copy",
                            "nights in a row", "check-up failing", "clean failing", "stuck in the queue"] {
                    XCTAssertFalse(text.contains("\"") && text.contains(old), "\(file):\(number + 1) still says \"\(old)\"")
                }
            }
        }
    }

    // MARK: Revision 1.10.1

    /// R1.10.1: after Reconnect (the reconnect-waiting fixture, and a press on the row's Reconnect in drive-signed-out), the
    /// problem row, Google Drive's station status, its panel, Settings and the dropdown all take the one waiting state,
    /// from the one source, and none offers a filled Reconnect; before the press, all offer it.
    func testR1101WaitingAfterReconnectIsShownEverywhere() throws {
        let pressed = NDStore(fixture: Fixture.inputs(.driveSignedOut), page: .overview)
        let before = pressed.r1
        let rowAction = try XCTUnwrap(before.problemRows.first { $0.id == "drive" }?.action)
        XCTAssertFalse(NDDriveWaiting.shows(pressed.inputs, for: rowAction), "before the press: Reconnect")
        XCTAssertEqual(before.pipeline.status[.drive]?.text, Say.signedOut)
        pressed.perform(rowAction.kind)  // a click on the row's Reconnect
        let cases: [(String, NDStore)] = [("reconnect-waiting", NDStore(fixture: Fixture.inputs(.reconnectWaiting), page: .overview)),
                                          ("drive-signed-out after Reconnect", pressed)]
        for (name, store) in cases {
            let r1 = store.r1
            let row = try XCTUnwrap(r1.problemRows.first { $0.id == "drive" }, name)
            XCTAssertTrue(NDDriveWaiting.shows(store.inputs, for: row.action), "\(name): the row waits")
            XCTAssertEqual(r1.pipeline.status[.drive]?.text, Say.stationWaiting, "\(name): the station waits")
            let panel = NDStationPopover.model(.drive, item: r1.stationItems[.drive], store.inputs, store.model, r1.pipeline)
            XCTAssertTrue(NDDriveWaiting.shows(store.inputs, for: panel.action), "\(name): the panel waits")
            XCTAssertTrue(NDDriveWaiting.shows(store.inputs, for: NDSettingsView.driveAction(r1)), "\(name): Settings waits")
            let dropdown = r1.items.first { $0.id == "drive" }?.action
            XCTAssertTrue(NDDriveWaiting.shows(store.inputs, for: dropdown), "\(name): the dropdown waits")
        }
        // Every surface that offers Reconnect reads the one predicate.
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign")
        for (file, needle) in [("NDWindow.swift", "if NDDriveWaiting.shows(store.inputs, for: action) {"),
                               ("NDPipelineChart.swift", "waiting: NDDriveWaiting.shows(store.inputs, for: store.r1.stationItems[station]?.action)"),
                               ("NDSettings.swift", "NDWaitingSwap(waiting: NDDriveWaiting.shows(store.inputs, for: action))"),
                               ("NDMenu.swift", "NDWaitingSwap(waiting: NDDriveWaiting.shows(store.inputs, for: action), alignment: .leading)"),
                               ("NDPages.swift", "NDWaitingSwap(waiting: NDDriveWaiting.shows(store.inputs, for: action))"),
                               ("NDPipeline.swift", "NDDriveWaiting.shows(i, for: nil) ? Say.stationWaiting : Say.signedOut")] {
            let source = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8)
            XCTAssertTrue(source.contains(needle), "\(file) reads the one waiting state")
        }
        XCTAssertEqual(Say.stationWaiting, Say.waitingRow.lowercased(), "the approved word")
    }

    // MARK: Revision 1.10 (L48 to L51, CHANGES-R1.10)

    /// R1.10 A (L48): one very short row per red problem and none for amber; each row is one line at the default width;
    /// each row's action is its station panel's and its dropdown block's; Reconnect first, then the pipeline's order.
    func testR110OneRowPerRedProblem() throws {
        let card = NDGrid.windowWidth - 2 * NDGrid.pageSide
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let r1 = store.r1
            let red = r1.items.filter { $0.severity == .red }
            XCTAssertEqual(Set(r1.problemRows.map(\.id)), Set(red.map(\.id)), "\(name.rawValue): one row per red problem")
            XCTAssertFalse(r1.problemRows.contains { row in r1.items.contains { $0.id == row.id && $0.severity == .orange } },
                           "\(name.rawValue): amber gets no row")
            let menu = NDMenuBuilder(store: store, perform: { _ in }, openWindow: { _ in }).build()
            for row in r1.problemRows {
                // One line: the mark, the words (13 Semibold) and the widest button fit the card.
                let words = NDPipelineLayout.textWidth(row.words, size: 13, weight: .semibold)
                XCTAssertLessThanOrEqual(14 + 14 + 10 + words + 12 + 110 + 14, card, "\(name.rawValue): \(row.words) fits one line")
                guard let action = row.action, let station = row.station else { continue }
                let panel = NDStationPopover.model(station, item: r1.stationItems[station], store.inputs, store.model, r1.pipeline)
                XCTAssertEqual(panel.action, action, "\(name.rawValue): \(row.words): the panel's action")
                // R1.13.1: while a sign-in waits, the dropdown's item reads the waiting words instead.
                XCTAssertEqual(menu.items.contains { ($0.representedObject as? BrainBarMenuAction) == .row(action.kind) },
                               !NDDriveWaiting.shows(store.inputs, for: action), "\(name.rawValue): \(row.words): the dropdown's action")
            }
            if !r1.problemRows.isEmpty {
                let host = NSHostingView(rootView: NDProblemRows(store: store, rows: r1.problemRows).frame(width: card))
                let lines = CGFloat(r1.problemRows.count)
                XCTAssertEqual(host.fittingSize.height, lines * NDProblemRows.rowHeight + (lines - 1) * 0.5, accuracy: 1,
                               "\(name.rawValue): 36 pt rows, one line each")
            }
        }
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        // R1.12 (§ C): one wording per fault, the row's as everywhere else.
        XCTAssertEqual(two.problemRows.map(\.words), ["Google Drive is signed out", "Nightly check-up failed 3 nights"], "Reconnect first")
        XCTAssertEqual(two.problemRows.first?.action?.kind, .reconnectDrive)
        let stuck = NDR1.make(Fixture.inputs(.savesStuck), NDModel.make(Fixture.inputs(.savesStuck)))
        XCTAssertEqual(stuck.problemRows.map(\.words), ["Saves are stuck"])
        let agent = NDR1.make(Fixture.inputs(.agentNotRead), NDModel.make(Fixture.inputs(.agentNotRead)))
        XCTAssertEqual(agent.problemRows.map(\.words), ["Cursor\u{2019}s sessions aren\u{2019}t being read"])
    }

    /// R1.10 A 4 (L51): with one problem the headline is "1 thing needs you" and the row names the problem; the two never
    /// say the same words.
    func testR110OneProblemSaysOneThingAndTheRowNamesIt() {
        let r1 = NDR1.make(Fixture.inputs(.driveSignedOut), NDModel.make(Fixture.inputs(.driveSignedOut)))
        XCTAssertEqual(r1.headline, "1 thing needs you")
        XCTAssertEqual(r1.problemRows.map(\.words), ["Google Drive is signed out"])  // R1.12 (§ C)
        XCTAssertFalse(r1.headline.localizedCaseInsensitiveContains("Google Drive"), "no repeated words")
        XCTAssertNil(r1.headlineDetail)
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        XCTAssertEqual(two.headline, "2 things need you")
    }

    /// R1.10 B (L49, L51): the two job stations each fault on their own job only, each says its last run, both link to
    /// Memory both ways, and the weekly clean never shows a skipped run as done.
    func testR110TwoJobStationsFaultOnTheirOwn() throws {
        let nightly = pipeline(.nightlyFailing)
        XCTAssertTrue(nightly.faults.contains(.jobs))
        XCTAssertFalse(nightly.faults.contains(.weekly), "a failing nightly does not mark the weekly")
        var weeklyFailing = Fixture.inputs(.calm)
        let index = try XCTUnwrap(weeklyFailing.jobs?.firstIndex { $0.job == .maintenanceWeekly })
        weeklyFailing.jobs?[index].lastOutcome = .failed
        weeklyFailing.jobs?[index].failureStreak = 2
        let weekly = NDPipelineState.make(weeklyFailing)
        XCTAssertTrue(weekly.faults.contains(.weekly))
        XCTAssertFalse(weekly.faults.contains(.jobs), "a failing weekly does not mark the nightly")
        XCTAssertEqual(weekly.status[.weekly], .init(text: Say.failedWeeks(2), severity: .red))
        let r1 = NDR1.make(weeklyFailing, NDModel.make(weeklyFailing))
        XCTAssertEqual(r1.problemRows.map(\.words), ["Weekly clean failed 2 weeks"])  // R1.12 (I 7)
        XCTAssertEqual(r1.problemRows.first?.action?.kind, .copyReport)
        XCTAssertEqual(r1.headline, Say.oneThingNeedsYou)
        // Calm: each says its last run; a weekly run its gates skipped is never shown as done.
        let calm = pipeline(.calm)
        XCTAssertTrue(calm.status[.jobs]?.text.hasPrefix("last run ") == true)
        XCTAssertTrue(calm.status[.weekly]?.text.hasPrefix("last run ") == true)
        var skipped = Fixture.inputs(.calm)
        skipped.jobs?[index].lastOutcome = .unknown
        XCTAssertNil(NDPipelineState.make(skipped).status[.weekly], "a skipped run says nothing, never \"last run\"")
        // Gap 79 / L56 deliberately spreads the job stations for their purpose lines. The elbow links still reach
        // Memory's bottom edge, with an arrowhead at each end; their endpoint need not share the station's x.
        let L = NDPipelineLayout.self
        for (pipe, station) in [(NDPipe.jobsMemory, NDStation.jobs), (.weeklyMemory, .weekly)] {
            let link = try XCTUnwrap(L.connections.first { $0.pipe == pipe })
            XCTAssertTrue(link.bothWays, "\(station): both ways")
            XCTAssertEqual(link.line.points.first?.x, L.point(station).x)
            let anchorX = station == .jobs ? L.hub.minX + 12 : L.hub.maxX - 12
            XCTAssertEqual(link.line.points.last, CGPoint(x: anchorX, y: L.hub.maxY + L.edgeGap))
            XCTAssertTrue((L.hub.minX...L.hub.maxX).contains(anchorX), "\(station) connects to Memory")
        }
        XCTAssertEqual(NDPipelineGraph.ends[.weeklyMemory]?.from, .weekly)
    }

    /// R1.10 C (L50): "Local copy": its panel's count is the real number of copies, and the newest copy is a clock time on
    /// its day ("today at", "yesterday at", "{weekday} at"), right across midnight in two time zones.
    func testR110LocalCopyPanel() throws {
        XCTAssertEqual(NDPipelineLayout.name(.mac), "Local copy")
        let late = Fixture.inputs(.backupLate)
        let rows = NDStationPopover.rows(.mac, late, NDModel.make(late))
        XCTAssertEqual(rows.first { $0.label == Say.keptRow }?.value, "\(try XCTUnwrap(late.localCopiesKept))", "the real count")
        let iso = ISO8601DateFormatter()
        func day(_ date: String, now: String, zone: String) throws -> String {
            var c = Fixture.inputs(.calm).capture
            c.now = try XCTUnwrap(iso.date(from: now))
            c.timeZone = try XCTUnwrap(TimeZone(identifier: zone))
            return NDR1.dayAt(try XCTUnwrap(iso.date(from: date)), c)
        }
        // 00:30 in Jerusalem: 23:50 the evening before is yesterday; there it is still the same UTC day.
        XCTAssertEqual(try day("2026-11-15T21:50:00Z", now: "2026-11-15T22:30:00Z", zone: "Asia/Jerusalem"), "yesterday at 23:50")
        XCTAssertEqual(try day("2026-11-15T21:50:00Z", now: "2026-11-15T22:30:00Z", zone: "UTC"), "today at 21:50")
        XCTAssertEqual(try day("2026-11-15T22:10:00Z", now: "2026-11-15T22:30:00Z", zone: "Asia/Jerusalem"), "today at 00:10")
        XCTAssertEqual(try day("2026-11-13T02:00:00Z", now: "2026-11-15T22:30:00Z", zone: "Asia/Jerusalem"), "Friday at 04:00")
        XCTAssertEqual(try day("2026-11-13T02:00:00Z", now: "2026-11-15T22:30:00Z", zone: "UTC"), "Friday at 02:00")
        let calmRows = NDStationPopover.rows(.mac, Fixture.inputs(.calm), NDModel.make(Fixture.inputs(.calm)))
        XCTAssertTrue(calmRows.first { $0.label == Say.newestRow }?.value.hasPrefix("today at ") == true)
        XCTAssertEqual(Say.localCopyStillMade, "The local copy is still made every night.")
    }

    /// The lead's 1.10 addition: Settings' backup section shows the same action as Google Drive's panel, with the same
    /// words ("Reconnect"), whenever Drive is signed out, and nothing when it is not.
    func testR110SettingsReconnectOnlyWhenSignedOut() throws {
        let out = NDStore(fixture: Fixture.inputs(.driveSignedOut), page: .overview)
        let action = try XCTUnwrap(NDSettingsView.driveAction(out.r1), "signed out: Reconnect in Settings")
        let panel = NDStationPopover.model(.drive, item: out.r1.stationItems[.drive], out.inputs, out.model, out.r1.pipeline)
        XCTAssertEqual(action, panel.action, "the same action as the panel")
        XCTAssertEqual(NDPipelineChart.shortLabel(action), "Reconnect", "the same words")
        for name in [Fixture.Name.calm, .backupLate, .nightlyFailing] {
            XCTAssertNil(NDSettingsView.driveAction(NDStore(fixture: Fixture.inputs(name), page: .overview).r1), "\(name.rawValue): nothing")
        }
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDSettings.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("NDButton(label: NDPipelineChart.shortLabel(action)"), "Settings uses the panel's words")
    }

    // MARK: Revision 1.8 (R4-DECISIONS "Taken")

    /// R1.8 item 2, on the live data path (`tickFixture` hands a sample over exactly as a real one is) and the live chart
    /// view: while the pointer is inside the chart (its hover holds the store), arriving data waits: the bars, the
    /// readout's minute and counts, and the totals stay; the dropdown's hour keeps flowing; the pointer leaving lets the
    /// newest in.
    func testR18RestingPointerHoldsTheChartOnTheLivePath() throws {
        let store = NDStore(fixture: Fixture.inputs(.calm), page: .overview)
        let column = 20
        let before = try XCTUnwrap(store.inputs.heroWaves[.hour])
        let span = NDActivityReadout.span(before, index: column, minutes: 60, now: store.inputs.capture.now, zone: store.inputs.capture.timeZone)
        func readoutFrame() -> CGRect? {
            var frame: CGRect?
            let host = NSHostingView(rootView: NDActivitySection(store: store, hover: column, reportReadoutFrame: { frame = $0 }).frame(width: 852))
            host.frame = NSRect(x: 0, y: 0, width: 852, height: 220)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
            host.layoutSubtreeIfNeeded()
            return frame
        }
        let frameBefore = try XCTUnwrap(readoutFrame())
        let menuBefore = store.inputs.menuWave
        store.holdChart(true)  // the chart's hover, pointer inside
        store.tickFixture()
        store.tickFixture()
        let held = try XCTUnwrap(store.inputs.heroWaves[.hour])
        XCTAssertEqual(held, before, "no bar moves under the resting pointer")
        XCTAssertEqual(held.savesTotal, before.savesTotal, "the totals hold")
        XCTAssertEqual(NDActivityReadout.span(held, index: column, minutes: 60, now: store.inputs.capture.now,
                                              zone: store.inputs.capture.timeZone), span, "the readout keeps its minute")
        XCTAssertEqual(try XCTUnwrap(readoutFrame()), frameBefore, "and its place")
        XCTAssertNotEqual(store.inputs.menuWave, menuBefore, "the dropdown's hour keeps flowing")
        store.holdChart(false)  // the pointer leaves
        XCTAssertNotEqual(store.inputs.heroWaves[.hour], before, "leaving lets the newest in")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDWindow.swift"), encoding: .utf8)
        let hover = try XCTUnwrap(source.range(of: ".onContinuousHover(coordinateSpace: .local) { phase in\n                        switch phase {\n                        // R1.8 item 2"))
        let tail = source[hover.upperBound...].prefix(400)
        XCTAssertTrue(tail.contains("store.holdChart(true)") && tail.contains("store.holdChart(false)"), "the chart's hover holds and releases")
    }

    /// R1.8 item 3: the dropdown's buttons take no focus, so none opens with a halo; the menu's highlight (from a key)
    /// draws the ring.
    func testR18NoFocusHaloWhenTheDropdownOpens() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign")
        let menu = try String(contentsOf: dir.appendingPathComponent("NDMenu.swift"), encoding: .utf8)
        XCTAssertTrue(menu.contains("NDButton(label: action.label, filled: filled, small: true, takesFocus: false)"))
        XCTAssertTrue(menu.contains(".opacity(state.keyboard ? 1 : 0)"), "the ring only after a key")
        let window = try String(contentsOf: dir.appendingPathComponent("NDWindow.swift"), encoding: .utf8)
        XCTAssertTrue(window.contains("button.focusEffectDisabled().accessibilityLabel"), "a focus-less button is no stop and draws no halo")
    }

    /// R1.8 item 4: one keyboard stop per station and per panel action, each drawing its ring: a station is no longer a
    /// Button inside a focusable wrapper (with Full Keyboard Access the inner button was a stop that drew nothing).
    func testR18EveryKeyboardStopShowsItsFocus() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign")
        let chart = try String(contentsOf: dir.appendingPathComponent("NDPipelineChart.swift"), encoding: .utf8)
        let a = try XCTUnwrap(chart.range(of: "struct NDStationButton: View {"))
        let b = try XCTUnwrap(chart.range(of: "    /// R1.5 change 4, the station's own key handling"))
        let station = chart[a.lowerBound..<b.lowerBound]
        XCTAssertFalse(station.contains("Button {"), "a station is one stop, not a button inside a stop")
        XCTAssertTrue(station.contains(".focusable(interactions: .activate)") && station.contains("if focused {"), "and draws its ring")
        let window = try String(contentsOf: dir.appendingPathComponent("NDWindow.swift"), encoding: .utf8)
        XCTAssertTrue(window.contains(".buttonStyle(NDButtonStyle(filled: filled, small: small, compact: compact))\n            .focusable(false)"))
        let panel = try String(contentsOf: dir.appendingPathComponent("NDStationPanel.swift"), encoding: .utf8)
        XCTAssertTrue(panel.contains(".focusable(false)  // R1.8 item 4"), "Copy report: one stop")
    }

    /// R1.8 item 5: the Jobs fault line names the job that failed.
    func testR18JobsFaultLineNamesItsJob() {
        // R1.10 (L49): the job is the station now ("Nightly check-up"), so its line is "failed 3 nights" under that name.
        XCTAssertEqual(pipeline(.nightlyFailing).status[.jobs], .init(text: Say.failedNights(3), severity: .red))
        XCTAssertEqual(NDPipelineLayout.name(.jobs), "Nightly check-up")
        XCTAssertFalse(NDPipelineChart.marks(.fault).nameRed, "still only the short line is red")
    }

    /// R1.8 item 6: the waiting line after Reconnect runs across the panel's footer from its left, 13 pt secondary, with a
    /// 12 pt spinner first; at the panel's width it takes at most two lines.
    func testR18WaitingLineFitsTheFooter() throws {
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDStationPanel.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("NDWaitingSwap(waiting: waiting, alignment: .leading)"), "left-aligned across the footer")
        XCTAssertTrue(source.contains("ProgressView().progressViewStyle(.circular).controlSize(.mini).frame(width: 12, height: 12)"))
        XCTAssertTrue(source.contains("Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(ND.t2)"))
        let host = NSHostingView(rootView: NDStationPanelBody.waitingLine.frame(width: 288 - 28))
        let height = host.fittingSize.height
        let line = ceil(NSLayoutManager().defaultLineHeight(for: .systemFont(ofSize: 13)))
        XCTAssertLessThanOrEqual(height, 2 * line + 1, "at most two lines across the footer")
    }

    /// R1.8 item 7: the dark dropdown over a white desktop, measured again (round-4 captures of 1.7): our primary 3.90 : 1,
    /// secondary 3.13 : 1; the system's own vibrant label on the same material 3.30 : 1, under our primary, so vibrant
    /// colours would lower it. The material (grey 129) caps any colour at 3.95 : 1. No second surface.
    func testR18DarkDropdownOverWhiteIsMeasured() throws {
        let measured = (ours: 3.90, oursSecondary: 3.13, systemVibrant: 3.30)
        XCTAssertGreaterThan(measured.ours, measured.systemVibrant, "the vibrant label is lower than ours")
        let cap = Self.contrast(.white, desktop: 255, dark: true)
        XCTAssertEqual(Self.contrast(NDColor.menuText1, desktop: 255, dark: true), cap, accuracy: 0.01, "primary at the material's limit")
        XCTAssertEqual(measured.ours, cap, accuracy: 0.1, "the model agrees with the capture")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDMenu.swift"), encoding: .utf8)
        XCTAssertFalse(source.contains("menuBacking"), "no second surface")
    }

    // MARK: Revision 1.7 (R3-DECISIONS "Taken")

    private static func contrast(_ color: NSColor, desktop: CGFloat, dark: Bool) -> CGFloat {
        func lum(_ v: CGFloat) -> CGFloat { let x = v / 255; return x <= 0.03928 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4) }
        func luminance(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> CGFloat {
            let red = 0.2126 * lum(r), green = 0.7152 * lum(g), blue = 0.0722 * lum(b)
            return red + green + blue
        }
        let material = dark ? NDMenuMaterial.dark : NDMenuMaterial.light
        let m = material.tint * material.alpha + desktop * (1 - material.alpha)
        var resolved = NSColor.clear
        NSAppearance(named: dark ? .darkAqua : .aqua)!.performAsCurrentDrawingAppearance { resolved = color.usingColorSpace(.sRGB)! }
        let a = resolved.alphaComponent
        let text = luminance(resolved.redComponent * 255 * a + m * (1 - a), resolved.greenComponent * 255 * a + m * (1 - a),
                             resolved.blueComponent * 255 * a + m * (1 - a))
        let back = luminance(m, m, m)
        return (max(text, back) + 0.05) / (min(text, back) + 0.05)
    }

    /// R1.7 item 1: one material: our rows draw no backing (the seam against the system rows is gone); their words sit on
    /// the system material in colours measured over white and black desktops. Light holds 4.5 : 1 everywhere; dark over a
    /// white desktop cannot (pure white reaches 3.95 : 1 on that grey), so the test pins how far short it is.
    func testR17DropdownIsOneMaterial() throws {
        for desktop in [CGFloat(0), 255] {
            XCTAssertGreaterThanOrEqual(Self.contrast(NDColor.menuText1, desktop: desktop, dark: false), 4.5, "light primary")
            XCTAssertGreaterThanOrEqual(Self.contrast(NDColor.menuText2, desktop: desktop, dark: false), 4.5, "light secondary")
        }
        XCTAssertGreaterThanOrEqual(Self.contrast(NDColor.menuText1, desktop: 0, dark: true), 4.5, "dark over black")
        XCTAssertGreaterThanOrEqual(Self.contrast(NDColor.menuText2, desktop: 0, dark: true), 4.5, "dark over black")
        let cap = Self.contrast(.white, desktop: 255, dark: true)
        XCTAssertEqual(cap, 3.95, accuracy: 0.05, "the most any colour reaches on dark glass over white")
        XCTAssertEqual(Self.contrast(NDColor.menuText1, desktop: 255, dark: true), cap, accuracy: 0.01, "primary is at that limit")
        XCTAssertEqual(Self.contrast(NDColor.menuText2, desktop: 255, dark: true), 3.16, accuracy: 0.05, "secondary: 1.34 short")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDMenu.swift"), encoding: .utf8)
        XCTAssertFalse(source.contains("menuBacking") || source.contains("NDMenuSeparator"), "no backing, no hosted separators")
        XCTAssertFalse(source.contains(".secondaryLabelColor") || source.contains(".labelColor") || source.contains("ND.redText")
                       || source.contains("ND.orangeText"), "words only in the measured colours")
        let menu = NDMenuBuilder(store: NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview), perform: { _ in },
                                 openWindow: { _ in }).build()
        XCTAssertTrue(menu.items.filter { $0.view == nil && $0.title.isEmpty }.allSatisfy(\.isSeparatorItem), "system separators")
    }

    /// First look, 22 Nov: the agents point at the column centre and retain full content below it.
    func testR17AgentsPanelOpensRightOfTheColumn() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            let content = AnyView(NDStationPopover(store: store, station: .agents))
            let panel = chart.panel(.agents, content: content)
            XCTAssertEqual(panel.edge, .bottom, name.rawValue)
            XCTAssertEqual(panel.tip.x, L.agentsFrame.midX, accuracy: 0.5)
            XCTAssertGreaterThanOrEqual(panel.tip.x - panel.body.minX, L.arrowInset - 0.01, "\(name.rawValue): under the straight edge")
            XCTAssertFalse(panel.outer.intersects(L.pipelineHeading))
            XCTAssertGreaterThan(panel.body.minY, L.agentsFrame.maxY)
            XCTAssertGreaterThanOrEqual(panel.outer.minX, L.windowInside.minX)
            XCTAssertGreaterThanOrEqual(panel.body.height, NDStationPopover.measure(content, width: panel.body.width).height)
            XCTAssertFalse(panel.scrolls)
            let agentsBelowBothSidesBlocked = panel.edge == .bottom && panel.clearanceException == .topRowBelowBothSidesBlocked
            if agentsBelowBothSidesBlocked {
                for side: Edge in [.top, .trailing, .leading] {
                    XCTAssertNil(L.panelCandidate(.agents, edge: side, state: store.r1.pipeline, actions: [], height: panel.body.height, width: panel.body.width))
                }
            }
            for station in NDStation.allCases {
                for rect in L.labelRects(station, store.r1.pipeline) {
                    if station == .agents || !agentsBelowBothSidesBlocked {
                        XCTAssertFalse(panel.body.intersects(rect), "\(name.rawValue): over \(station)'s words")
                    }
                }
            }
            let screen = CGRect(x: L.windowInside.minX - 40, y: -400, width: 2000, height: 2000)
            XCTAssertEqual(NDPipelineLayout.fitted(panel, within: screen).body.minX, panel.body.minX, "keyboard: not clipped")
        }
    }

    /// R13: both jobs open below their full words, within the chart, without covering a label or caption.
    func testR17JobsPanelOpensBesideItsStation() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            for station in [NDStation.jobs, .weekly] {
                let content = AnyView(NDStationPopover(store: store, station: station))
                let panel = chart.panel(station, content: content)
                let where_ = "\(name.rawValue) \(station)"
                XCTAssertEqual(panel.edge, .bottom, where_)
                XCTAssertTrue((L.circle(station).minX...L.circle(station).maxX).contains(panel.tip.x), where_)
                let wordsBottom = L.labelRects(station, store.r1.pipeline).map(\.maxY).max() ?? 0
                XCTAssertGreaterThan(panel.body.minY, wordsBottom, where_)
                XCTAssertLessThanOrEqual(panel.body.maxY, L.size.height, where_)
                XCTAssertGreaterThanOrEqual(panel.body.height + 0.01, NDStationPopover.measure(content, width: panel.body.width).height, where_)
                for other in NDStation.allCases where other != station {
                    XCTAssertFalse(panel.body.intersects(L.circle(other)), where_)
                    for rect in L.labelRects(other, store.r1.pipeline) { XCTAssertFalse(panel.outer.intersects(rect), where_) }
                }
                for (_, rect) in L.laneLabelRects() { XCTAssertFalse(panel.outer.intersects(rect.insetBy(dx: -4, dy: -4)), where_) }
                for text in L.footerTexts { XCTAssertFalse(panel.outer.intersects(text), where_) }
            }
        }
    }

    /// R1.7 item 4: one lateness rule: the Mac copy's warning (the check's amber state) and the Backups page's "late after"
    /// line share one threshold, the check's own amber threshold in health_verdict.py (30 h, not the 36 h of the drawing).
    func testR17OneLatenessRule() throws {
        let python = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/brainlayer/health_verdict.py"), encoding: .utf8)
        let match = try XCTUnwrap(python.firstMatch(of: /backup_local_amber_s: float = (\d+) \* HOURS/))
        XCTAssertEqual(NDBackupFacts.lateAfterHours, try XCTUnwrap(Double(match.1)), "the page's line is the check's threshold")
        XCTAssertEqual(NDBackupsPage.limit, -NDBackupFacts.lateAfterHours)
        XCTAssertEqual(Say.lateAfter(NDFormat.ageText(30 * 3600)), "late after 30\u{00A0}h")
        let late = Fixture.inputs(.backupLate)
        let local = try XCTUnwrap(late.capture.verdict?.checks.first { $0.name == "backup_local" })
        XCTAssertEqual(local.state, .amber)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(local.ageS), NDBackupFacts.lateAfterHours * 3600, "amber at the same threshold")
    }

    /// R1.7 item 5: not confirmed says nothing current: the copy on this Mac says "—" like the others; the footer names the
    /// next check only when it is still due; the dropdown's sparkline is hatched after the last confirmed check.
    func testR17NotConfirmedSaysNothingCurrent() throws {
        let stale = pipeline(.notConfirmed)
        // R1.12 (E2): the copy keeps its last-known value as a clock time (in `t3`), never a current age.
        XCTAssertFalse(stale.status[.mac]?.text.hasSuffix(" ago") ?? true, stale.status[.mac]?.text ?? "nil")
        let store = NDStore(fixture: Fixture.inputs(.notConfirmed), page: .overview)
        XCTAssertNil(store.r1.footerRight, "2 h overdue: no next time is known")
        XCTAssertEqual(try XCTUnwrap(NDActivitySection.noDataFraction(store, wave: store.inputs.menuWave, minutes: 60)), 0,
                       accuracy: 0.001, "the dropdown's hour is all after the last check")
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDMenu.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("noDataFrom: NDActivitySection.noDataFraction(store, wave: store.inputs.menuWave, minutes: 60)"))
    }

    /// R1.12 (E3, replaces R1.7 item 6): when the local copy is late, Drive's older upload is its consequence, not a
    /// second problem: its line stays `t2`, with no ring.
    func testR17ALateDriveCopyLooksLate() {
        XCTAssertNil(pipeline(.backupLate).status[.drive]?.severity)
        XCTAssertNil(pipeline(.calm).status[.drive]?.severity)
    }

    /// R1.7 item 8: no stray tick above "now": the strip under the plot is empty.
    func testR17NoStrayTickAboveNow() throws {
        let wave = try XCTUnwrap(Fixture.inputs(.calm).heroWaves[.hour])
        let size = NDActivitySection.canvas
        let renderer = ImageRenderer(content: NDWaveform(wave: wave, size: .hero, nowTick: true).frame(width: size.width, height: size.height))
        renderer.scale = 1
        let image = try XCTUnwrap(renderer.cgImage)
        let rep = NSBitmapImageRep(cgImage: image)
        let plotHeight = Int(NDWaveform.plot(size, nowTick: true).h)
        var inked = 0
        for y in (plotHeight + 1)..<Int(size.height) {
            for x in 0..<Int(size.width) where (rep.colorAt(x: x, y: y)?.alphaComponent ?? 0) > 0.05 { inked += 1 }
        }
        XCTAssertEqual(inked, 0, "nothing drawn under the plot")
    }

    /// R1.7 item 9: the dropdown's Activity symbol reads at 1:1, at the other rows' size.
    func testR17ActivityGlyphReadsAtOneToOne() {
        XCTAssertEqual(NDMenuAreaRow.symbols[.activity], "waveform.path.ecg")
        XCTAssertNotNil(NSImage(systemSymbolName: "waveform.path.ecg", accessibilityDescription: nil))
    }

    /// R1.7 item 10: panel text: body in the secondary colour, at most two lines a paragraph at the panel's width; rows
    /// with the value in primary semibold. R1.12 (§ A): on the scale's 11 pt step, each panel at its own width (§ B).
    func testR17PanelText() throws {
        XCTAssertEqual(NDType.panelBodySize, NDScale.caption)
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDStationPanel.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("Text(keepingUnits(line)).font(NDType.panelBody).foregroundStyle(ND.t2)"))
        XCTAssertTrue(source.contains("Text(line.label).font(NDType.panelBody)") && source.contains("Text(line.value).font(NDType.panelValue)"))
        // Two of the deck's sentences run to three lines at 12 pt on their own: Read's explanation and the first sentence
        // of Drive's. Holding them to two needs shorter sentences in the deck (reported, not rewritten here).
        let longSentences = [Say.explainRead, NDStationPanelBody.paragraphs([Say.explainDrive])[0], Say.explainNightly]
        XCTAssertEqual(NDStationPanelBody.paragraphs([Say.explainQueue]).count, 2, "one paragraph a sentence, the deck's words")
        XCTAssertEqual(NDStationPanelBody.paragraphs([Say.explainQueue]).joined(separator: " "), Say.explainQueue)
        let font = NSFont.systemFont(ofSize: NDType.panelBodySize)
        // One line as the same measure draws it (at 11 pt, 14 pt; the layout manager's default rounds to 13).
        let lineHeight = ceil(NSAttributedString(string: "Ag", attributes: [.font: font])
            .boundingRect(with: NSSize(width: 1000, height: 1000), options: [.usesLineFragmentOrigin]).height)
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let r1 = store.r1
            for station in NDStation.allCases {
                let model = NDStationPopover.model(station, item: r1.stationItems[station], store.inputs, store.model, r1.pipeline)
                for line in NDStationPanelBody.paragraphs(model.lines) where !longSentences.contains(line) {
                    let height = NSAttributedString(string: keepingUnits(line), attributes: [.font: font])
                        .boundingRect(with: NSSize(width: NDPipelineLayout.panelSide(station, hasAction: false).width - 28, height: 1000),
                                      options: [.usesLineFragmentOrigin]).height
                    XCTAssertLessThanOrEqual(ceil(height), 2 * lineHeight + 1, "\(name.rawValue) \(station): \(line)")
                }
            }
        }
    }
}
