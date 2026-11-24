import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Phase 14 tab swipe: a two-finger sideways swipe moves to the neighbouring page with VoiceBar's History rules.
/// Pure models and offscreen hosting views only; nothing is attached to a window, menu or status item.
@MainActor
final class BrainBarTabSwipeTests: XCTestCase {
    // MARK: The pager (VoiceBar's NotchHistoryCategoryPager rules)

    func testASidewaysGestureMovesOnePageAndAVerticalOneNever() {
        var pager = NDPagePager()
        pager.begin()
        XCTAssertFalse(pager.move(dx: -3, dy: 0), "inside the 6 pt slop no axis is chosen")
        XCTAssertTrue(pager.move(dx: -60, dy: 4))
        pager.move(dx: -40, dy: 0)
        XCTAssertEqual(pager.end(), 1, "fingers moving left go to the next page")

        pager.begin()
        XCTAssertTrue(pager.move(dx: 100, dy: 0))
        XCTAssertEqual(pager.end(), -1, "fingers moving right go to the previous page")

        pager.begin()
        XCTAssertFalse(pager.move(dx: 5, dy: -30), "a vertical gesture is the page's own scroll")
        XCTAssertFalse(pager.move(dx: -300, dy: -5), "and stays vertical however far it then wanders sideways")
        XCTAssertEqual(pager.end(), 0)

        pager.begin()
        XCTAssertFalse(pager.move(dx: -20, dy: 18), "a diagonal is not clearly sideways")
        pager.move(dx: -200, dy: 0)
        XCTAssertEqual(pager.end(), 0)
    }

    func testTheThresholdIsVoiceBarsAndAFlickMovesOnHoweverShort() {
        XCTAssertEqual(NDPagePager.travelWidth * NDPagePager.commitFraction, 88, accuracy: 0.001,
                       "0.22 of VoiceBar's 400 pt History list")
        var pager = NDPagePager()
        pager.begin()
        for _ in 0..<8 { pager.move(dx: -11, dy: 0) } // 88 pt, every step slow
        XCTAssertEqual(pager.end(), 0, "88 pt is not past the threshold and was not a flick")

        pager.begin()
        for _ in 0..<8 { pager.move(dx: -11, dy: 0) }
        pager.move(dx: -1, dy: 0)
        XCTAssertEqual(pager.end(), 1, "89 pt is")

        pager.begin()
        pager.move(dx: -20, dy: 0)
        pager.move(dx: -16, dy: 0)
        XCTAssertEqual(pager.end(), 1, "a flick moves on however short")

        pager.begin()
        pager.move(dx: -60, dy: 0)
        pager.move(dx: 15, dy: 0)
        XCTAssertEqual(pager.end(), 0, "a fast step back against the swipe is not a flick toward it")
    }

    func testOneGestureMovesAtMostOnePageHoweverLong() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        var swipe = NDPageSwipe()
        let steps = gesture(&swipe, dx: -3000, store: store)
        XCTAssertEqual(steps, [1], "one step, taken on release")
        XCTAssertEqual(store.page, .backups, "never two pages")
    }

    func testTheEndsStopAndNeverWrap() {
        XCTAssertEqual(NDPage.jobs.neighbour(1), .jobs)
        XCTAssertEqual(NDPage.overview.neighbour(-1), .overview)
        XCTAssertEqual(NDPage.overview.neighbour(1), .backups)
        XCTAssertEqual(NDPage.jobs.neighbour(-1), .backups)
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .jobs)
        var swipe = NDPageSwipe()
        gesture(&swipe, dx: -200, store: store)
        XCTAssertEqual(store.page, .jobs, "past Jobs stays on Jobs")
        store.page = .overview
        gesture(&swipe, dx: 200, store: store)
        XCTAssertEqual(store.page, .overview, "before Overview stays on Overview")
    }

    // MARK: Event routing (phases, momentum, precedence)

    func testASidewaysSwipeKeepsItsEventsAndItsMomentumFromThePage() {
        var swipe = NDPageSwipe()
        XCTAssertEqual(swipe.route(.init(phase: .began)), .init(), "the axis is not known yet: the page sees it")
        XCTAssertEqual(swipe.route(.init(phase: .changed, dx: -4)), .init(), "still inside the slop")
        XCTAssertEqual(swipe.route(.init(phase: .changed, dx: -12)), .init(swallow: true), "now sideways: kept from the page")
        for _ in 0..<8 { XCTAssertTrue(swipe.route(.init(phase: .changed, dx: -12)).swallow) }
        XCTAssertEqual(swipe.route(.init(phase: .ended)), .init(swallow: false, step: 1),
                       "the step is taken on release, and the page's scroll view sees its gesture end")
        // The trackpad's momentum after a swipe: never another page, never a page scroll.
        XCTAssertEqual(swipe.route(.init(momentumPhase: .began, dx: -40)), .init(swallow: true))
        for _ in 0..<10 { XCTAssertEqual(swipe.route(.init(momentumPhase: .changed, dx: -40)), .init(swallow: true)) }
        XCTAssertEqual(swipe.route(.init(momentumPhase: .ended)), .init(swallow: true))
        XCTAssertEqual(swipe.route(.init(momentumPhase: .changed, dy: -40)), .init(),
                       "a later momentum (after the swipe's ended) belongs to the page again")
    }

    func testAVerticalScrollReachesThePageAndNeverChangesPage() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .backups)
        var swipe = NDPageSwipe()
        var decisions = [swipe.route(.init(phase: .began))]
        decisions += (0..<12).map { i in swipe.route(.init(phase: .changed, dx: i % 2 == 0 ? -9 : 9, dy: -20)) }
        decisions += (0..<6).map { _ in swipe.route(.init(phase: .changed, dx: -40, dy: 0)) } // wanders sideways later
        decisions.append(swipe.route(.init(phase: .ended)))
        decisions += (0..<5).map { _ in swipe.route(.init(momentumPhase: .changed, dy: -30)) }
        for d in decisions { store.swipePage(d.step) }
        XCTAssertTrue(decisions.allSatisfy { !$0.swallow }, "every event of a vertical scroll reaches the page")
        XCTAssertTrue(decisions.allSatisfy { $0.step == 0 })
        XCTAssertEqual(store.page, .backups)
    }

    func testAPlainMouseWheelIsLeftAlone() {
        var swipe = NDPageSwipe()
        for _ in 0..<20 {
            XCTAssertEqual(swipe.route(.init(dx: -40)), .init(), "a wheel without gesture phases is not a swipe")
        }
    }

    func testAGestureOutsideThePagesOrOverASidewaysScrollerIsNotASwipe() {
        var swipe = NDPageSwipe()
        _ = swipe.route(.init(phase: .began, inside: false))
        for _ in 0..<10 { XCTAssertEqual(swipe.route(.init(phase: .changed, dx: -30)), .init()) }
        XCTAssertEqual(swipe.route(.init(phase: .ended)), .init())

        // Precedence: a gesture that begins over a view that scrolls sideways is that view's for its whole length.
        _ = swipe.route(.init(phase: .began, overSidewaysScroller: true))
        for _ in 0..<10 { XCTAssertEqual(swipe.route(.init(phase: .changed, dx: -30)), .init(), "the scroller gets every event") }
        XCTAssertEqual(swipe.route(.init(phase: .ended)), .init(), "and no page changes")
        XCTAssertEqual(swipe.route(.init(momentumPhase: .changed, dx: -30)), .init(), "nor is its momentum taken")

        // The next gesture over the pages swipes again.
        XCTAssertEqual(swipeSteps(&swipe, dx: -120), [1])
    }

    func testSidewaysScrollersAreRecognisedAndVerticalOnesAreNot() {
        /// `document: nil` sizes the document to the clip, as SwiftUI's vertical ScrollView does (a legacy vertical scroller
        /// takes 15 pt of the 300).
        func scrollView(document: CGFloat?, horizontalScroller: Bool) -> (NSScrollView, NSView) {
            let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 300, height: 200))
            scroll.hasHorizontalScroller = horizontalScroller
            scroll.hasVerticalScroller = true
            let documentView = NSView(frame: NSRect(x: 0, y: 0, width: document ?? scroll.contentSize.width, height: 600))
            let leaf = NSView(frame: NSRect(x: 10, y: 10, width: 20, height: 20))
            documentView.addSubview(leaf)
            scroll.documentView = documentView
            return (scroll, leaf)
        }
        let vertical = scrollView(document: nil, horizontalScroller: false)
        XCTAssertFalse(NDPageSwipe.scrollsSideways(vertical.1), "a vertical scroller leaves the swipe to the pages")
        let wide = scrollView(document: 900, horizontalScroller: false)
        XCTAssertTrue(NDPageSwipe.scrollsSideways(wide.1), "a document wider than its view scrolls sideways")
        let scroller = scrollView(document: nil, horizontalScroller: true)
        XCTAssertTrue(NDPageSwipe.scrollsSideways(scroller.1))
        XCTAssertFalse(NDPageSwipe.scrollsSideways(nil))
    }

    /// The window root's own vertical page scroll must not count as a sideways scroller, or no swipe would ever start.
    func testThePagesOwnScrollViewNeverClaimsTheSwipe() throws {
        for page in NDPage.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.twoFaults), page: page)
            let size = NSSize(width: NDGrid.windowWidth, height: 520) // shorter than the pages: the page scrolls
            let host = NSHostingView(rootView: NDWindowRoot(store: store).frame(width: size.width, height: size.height))
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.2))
            host.layoutSubtreeIfNeeded()
            let scrolls = scrollViews(in: host)
            XCTAssertFalse(scrolls.isEmpty, "\(page): the page scrolls inside an NSScrollView")
            for scroll in scrolls { XCTAssertFalse(NDPageSwipe.scrollsSideways(scroll.documentView ?? scroll), "\(page)") }
            for y in stride(from: NDGrid.titleBar + 10, to: size.height - NDGrid.footer, by: 60) {
                let hit = host.hitTest(NSPoint(x: size.width / 2, y: y))
                XCTAssertFalse(NDPageSwipe.scrollsSideways(hit), "\(page) at y \(y)")
            }
        }
    }

    // MARK: The drawn window: the selected tab follows the page

    func testASwipeMovesTheSelectedTabWithThePage() throws {
        let output = ProcessInfo.processInfo.environment["BRAINBAR_SWIPE_RENDER_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        if let output { try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
        var laidOut: [NDPage] = []
        NDLaidOutPagesProbe.swap = { page, _ in laidOut.append(page) }
        defer { NDLaidOutPagesProbe.swap = nil }

        // Left three times (the third stops at Jobs), then right three times (the third stops at Overview).
        let script: [(name: String, dx: CGFloat, page: NDPage)] = [
            ("1-left-from-overview", -120, .backups), ("2-left-from-backups", -120, .jobs), ("3-left-at-jobs-stops", -120, .jobs),
            ("4-right-from-jobs", 120, .backups), ("5-right-from-backups", 120, .overview), ("6-right-at-overview-stops", 120, .overview),
        ]
        for (suffix, appearance) in NDRenderHarness.appearances {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
            let host = NSHostingView(rootView: NDWindowRoot(store: store, drawsTrafficLights: true))
            host.appearance = NSAppearance(named: appearance)
            func draw(_ name: String) throws {
                let size = NSSize(width: NDGrid.windowWidth, height: NDPageHeight.window(store, page: store.page, screen: nil))
                host.frame = NSRect(origin: .zero, size: size)
                host.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.3))
                host.layoutSubtreeIfNeeded()
                guard let output else { return }
                let rep = try NDRenderHarness.bitmap(size)
                host.cacheDisplay(in: host.bounds, to: rep)
                try NDRenderHarness.write(rep, name: "\(name)-\(suffix)", in: output)
            }
            try draw("swipe-0-start-overview")
            let control = try XCTUnwrap(segmentedControl(in: host), "the drawn title bar holds the tabs")
            XCTAssertEqual(control.selectedSegment, 0)
            var swipe = NDPageSwipe()
            for step in script {
                laidOut.removeAll()
                gesture(&swipe, dx: step.dx, store: store)
                try draw("swipe-\(step.name)")
                let index = try XCTUnwrap(NDPage.allCases.firstIndex(of: step.page))
                XCTAssertEqual(store.page, step.page, step.name)
                XCTAssertEqual(control.selectedSegment, index, "\(step.name): the drawn selected tab names the page")
                XCTAssertEqual(segmentedControl(in: host), control, "\(step.name): the same control, updated in place")
                XCTAssertEqual(store.shownPage, step.page, "\(step.name): the page the root draws")
                if laidOut.isEmpty {
                    XCTAssertTrue(step.name.hasSuffix("stops"), "\(step.name): the root laid out a new page")
                } else {
                    XCTAssertEqual(laidOut.last, step.page, "\(step.name): the root laid out the page the tab names")
                }
            }
        }
    }

    /// The live window's wiring: the root carries the monitor, and a step it reports changes the root's own store.
    /// (The monitor's AppKit event path needs a window, which this Mac's test rules forbid; its decisions are the router's.)
    func testTheWindowRootCarriesTheSwipeMonitorWiredToItsStore() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        let host = NSHostingView(rootView: NDWindowRoot(store: store))
        host.frame = NSRect(x: 0, y: 0, width: NDGrid.windowWidth, height: 600)
        host.layoutSubtreeIfNeeded()
        let probe = try XCTUnwrap(monitor(in: host), "the root installs the swipe monitor")
        XCTAssertEqual(probe.frame.width, NDGrid.windowWidth, accuracy: 0.5, "it covers the window's width")
        XCTAssertNil(probe.hitTest(NSPoint(x: 10, y: 10)), "it never takes a click")
        probe.onStep?(1)
        XCTAssertEqual(store.page, .backups)
        probe.onStep?(-1)
        XCTAssertEqual(store.page, .overview)
    }

    // MARK: Helpers

    /// A trackpad gesture of 10 equal steps (began, changed ×10, ended) through the router, applied to `store`.
    @discardableResult
    private func gesture(_ swipe: inout NDPageSwipe, dx: CGFloat, dy: CGFloat = 0, store: NDStore) -> [Int] {
        let steps = swipeSteps(&swipe, dx: dx, dy: dy)
        for step in steps { store.swipePage(step) }
        return steps
    }

    private func swipeSteps(_ swipe: inout NDPageSwipe, dx: CGFloat, dy: CGFloat = 0) -> [Int] {
        var decisions = [swipe.route(.init(phase: .began))]
        decisions += (0..<10).map { _ in swipe.route(.init(phase: .changed, dx: dx / 10, dy: dy / 10)) }
        decisions.append(swipe.route(.init(phase: .ended)))
        return decisions.map(\.step).filter { $0 != 0 }
    }

    private func segmentedControl(in view: NSView) -> NSSegmentedControl? {
        if let control = view as? NSSegmentedControl { return control }
        for sub in view.subviews { if let found = segmentedControl(in: sub) { return found } }
        return nil
    }

    private func monitor(in view: NSView) -> NDPageSwipeMonitor.Probe? {
        if let probe = view as? NDPageSwipeMonitor.Probe { return probe }
        for sub in view.subviews { if let found = monitor(in: sub) { return found } }
        return nil
    }

    private func scrollViews(in view: NSView) -> [NSScrollView] {
        var found: [NSScrollView] = []
        if let scroll = view as? NSScrollView { found.append(scroll) }
        return found + view.subviews.flatMap { scrollViews(in: $0) }
    }
}
