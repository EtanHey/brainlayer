import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// The shared tab-swipe spec on BrainBar's page tabs: the pill and the pages follow the fingers 1:1, every settle is one
/// critically damped spring from the presented position, slots never change width, and Reduce Motion never overshoots.
/// Pure models and offscreen hosting views only; nothing is shown.
@MainActor
final class BrainBarTabsFollowFingersTests: XCTestCase {
    private let frame = 1.0 / 120
    private var tabs: [GlassTab] { NDPageTabs.tabs(NDStore(fixture: BrainBarDesignFixture.inputs(.twoFaults)).r1) }
    private let metrics = NDPageTabs.metrics

    // MARK: (a) Fixed slots

    /// Spec 4: every selection lays out every tab in the same slot. On main the selected label was drawn Medium and
    /// sized to itself, so the selected tab widened and its neighbours moved ([86, 80, 58] vs [85, 82, 58]).
    func testSlotWidthsAreIdenticalForEverySelection() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.twoFaults), page: .overview)
        var sizes: [CGSize] = []
        var frames: [[CGRect]] = []
        for page in NDPage.allCases {
            store.page = page
            frames.append(metrics.slotFrames(NDPageTabs.tabs(store.r1)))
            let host = NSHostingView(rootView: NDPageTabs(store: store).fixedSize())
            host.layoutSubtreeIfNeeded()
            sizes.append(host.fittingSize)
        }
        for (i, row) in frames.enumerated() { XCTAssertEqual(row, frames[0], "selection \(i): every slot in place") }
        for size in sizes { XCTAssertEqual(size, sizes[0], "the bar never changes size with the selection") }
        // Each slot fits its title at the selected weight, so the selected style never needs more room.
        for (tab, frame) in zip(tabs, metrics.slotFrames(tabs)) {
            let medium = NSAttributedString(string: tab.title, attributes: [.font: NSFont.systemFont(ofSize: 13, weight: .medium)])
            XCTAssertGreaterThanOrEqual(frame.width, medium.size().width + 2 * metrics.padding, tab.title)
        }
        // The range switch (still the system control) gets the same rule: one width per label, whatever is selected.
        for title in NDRange.allCases.map(\.label) {
            XCTAssertEqual(NDSegments.labelImage(title, count: 0, selected: true, size: 11).size.width,
                           NDSegments.labelImage(title, count: 0, selected: false, size: 11).size.width, title)
        }
        for title in NDPage.allCases.map(\.title) {
            XCTAssertEqual(NDSegments.labelImage(title, count: 2, selected: true, size: 13).size.width,
                           NDSegments.labelImage(title, count: 2, selected: false, size: 13).size.width, title)
        }
    }

    /// Spec 1 and 2: between slots the pill's frame interpolates; at rest it is the slot's frame exactly.
    func testThePillInterpolatesBetweenSlotsAndRestsOnTheSlot() {
        let slots = metrics.slotFrames(tabs).map { $0.insetBy(dx: metrics.inset, dy: metrics.inset) }
        for (i, slot) in slots.enumerated() { XCTAssertEqual(metrics.pillFrame(tabs, progress: Double(i)), slot) }
        let half = metrics.pillFrame(tabs, progress: 0.5)
        XCTAssertEqual(half.minX, (slots[0].minX + slots[1].minX) / 2, accuracy: 1e-9)
        XCTAssertEqual(half.width, (slots[0].width + slots[1].width) / 2, accuracy: 1e-9)
        XCTAssertEqual(metrics.pillFrame(tabs, progress: -0.2), slots[0], "past an end the pill stays on the end slot")
        for p in stride(from: 0.0, through: 2.0, by: 0.05) {
            let centre = metrics.pillFrame(tabs, progress: p).midX
            XCTAssertEqual(metrics.progress(forPillCentre: centre, tabs), p, accuracy: 1e-9, "a dragged pill maps back")
        }
    }

    // MARK: (b) One spring, sampled at 120 Hz

    /// The pill and the pages, sampled every 1/120 s through a slow swipe across every tab, a fast flick, a click on the
    /// far tab and a click during a settle: no frame moves either more than 1/3 of a tab, and each ends exactly on the
    /// destination slot within 0.5 s of the last input.
    func testEveryScriptMovesSmoothlyAndRestsOnTheSlotWithinHalfASecond() {
        let width = Double(NDGrid.windowWidth)
        let third = Double(metrics.slotWidths(tabs).min()!) / 3
        var report: [String] = []

        /// `script` drives the pager and calls `frame()` once per 1/120 s it lets pass (each a drawn frame).
        func run(_ name: String, from start: Int, script: (GlassTabPager, inout Double, () -> Void) -> Void, expect: Int) {
            let pager = GlassTabPager(count: 3, selection: start)
            pager.isLive = { true }
            var frames: [(pill: CGRect, content: Double)] = []
            func sample() { frames.append((metrics.pillFrame(tabs, progress: pager.progress), pager.progress * width)) }
            sample()
            var time = 50.0
            script(pager, &time, sample)
            let lastInput = time
            var restedAt: Double?
            for _ in 0..<120 {
                time += frame
                pager.tick(at: time)
                sample()
                if pager.phase == .resting, restedAt == nil { restedAt = time }
            }
            var maxPill = 0.0, maxContent = 0.0
            for (a, b) in zip(frames, frames.dropFirst()) {
                maxPill = max(maxPill, Double(abs(b.pill.minX - a.pill.minX)), Double(abs(b.pill.maxX - a.pill.maxX)))
                maxContent = max(maxContent, abs(b.content - a.content))
            }
            let settle = (restedAt ?? .infinity) - lastInput
            report.append(String(format: "%@: max pill step %.2f pt (limit %.2f), max content step %.1f pt (limit %.0f), rest %.3f s",
                                 name, maxPill, third, maxContent, width / 3, settle))
            XCTAssertLessThanOrEqual(maxPill, third, "\(name): the pill never jumps")
            XCTAssertLessThanOrEqual(maxContent, width / 3, "\(name): the pages never jump")
            XCTAssertEqual(pager.selection, expect, name)
            XCTAssertLessThanOrEqual(settle, 0.5, "\(name): at rest within 0.5 s")
            XCTAssertEqual(metrics.pillFrame(tabs, progress: pager.progress),
                           metrics.slotFrames(tabs)[expect].insetBy(dx: metrics.inset, dy: metrics.inset),
                           "\(name): the pill rests on the destination slot's exact frame")
            XCTAssertEqual(pager.progress * width, Double(expect) * width, "\(name): the page rests exactly in place")
        }

        // Tracking: one finger step per frame (trackpads report at the display's rate), points as scroll deltas.
        func track(_ pager: GlassTabPager, _ time: inout Double, _ sample: () -> Void, points: [Double]) {
            pager.grab(at: time)
            for dx in points {
                time += frame
                pager.drag(by: -dx / width, at: time)
                sample()
            }
            time += frame
            pager.release(at: time)
        }
        func settleFor(_ pager: GlassTabPager, _ time: inout Double, _ sample: () -> Void, frames count: Int) {
            for _ in 0..<count {
                time += frame
                pager.tick(at: time)
                sample()
            }
        }

        run("slow swipe 0→1", from: 0, script: { pager, time, sample in
            track(pager, &time, sample, points: Array(repeating: -4, count: 150)) // 600 pt over 1.25 s
        }, expect: 1)
        run("slow swipe 1→2", from: 1, script: { pager, time, sample in
            track(pager, &time, sample, points: Array(repeating: -4, count: 150))
        }, expect: 2)
        run("slow swipe back 2→1", from: 2, script: { pager, time, sample in
            track(pager, &time, sample, points: Array(repeating: 4, count: 150))
        }, expect: 1)
        run("fast flick 0→1", from: 0, script: { pager, time, sample in
            track(pager, &time, sample, points: [-10, -18, -24, -28, -30])
        }, expect: 1)
        run("click far tab 0→2", from: 0, script: { pager, time, _ in
            pager.select(2, at: time)
        }, expect: 2)
        run("click during a settle 0→2 then 0", from: 0, script: { pager, time, sample in
            pager.select(2, at: time)
            settleFor(pager, &time, sample, frames: 12) // 100 ms into the settle
            XCTAssertGreaterThan(pager.progress, 0.5, "the first settle is under way")
            pager.select(0, at: time)
        }, expect: 0)
        run("swipe during a settle 0→1, on to 2", from: 0, script: { pager, time, sample in
            pager.select(1, at: time)
            settleFor(pager, &time, sample, frames: 6)
            let presented = pager.progress
            pager.grab(at: time)
            XCTAssertEqual(pager.progress, presented, "the swipe takes hold where the pill is drawn")
            for _ in 0..<20 {
                time += self.frame
                pager.drag(by: 24 / width, at: time)
                sample()
            }
            time += self.frame
            pager.release(at: time)
        }, expect: 2)
        print("[w294] 120 Hz table:\n" + report.joined(separator: "\n"))
    }

    /// Spec 5: a click during a settle retargets from the presented position and keeps its velocity (no restart from the
    /// old resting tab, no stop).
    func testAClickDuringASettleCarriesThePresentedPositionAndVelocity() {
        let pager = GlassTabPager(count: 3, selection: 0)
        pager.isLive = { true }
        var time = 0.0
        pager.select(2, at: time)
        for _ in 0..<10 {
            time += frame
            pager.tick(at: time)
        }
        let (position, velocity) = (pager.progress, pager.velocity)
        XCTAssertGreaterThan(velocity, 1)
        pager.select(1, at: time)
        XCTAssertEqual(pager.progress, position, "it starts where it is drawn")
        XCTAssertEqual(pager.velocity, velocity, "with the velocity it had")
        time += frame
        pager.tick(at: time)
        XCTAssertGreaterThan(pager.progress, position, "still moving the way it was going, for a frame")
    }

    // MARK: (c) Live tracking from the real scroll events

    /// Spec 1: during `.changed` the pages and the pill follow the scroll deltas 1:1, before any release. On main the
    /// page changed only at `.ended`.
    func testProgressFollowsScrollDeltasDuringChanged() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        let host = NSHostingView(rootView: NDWindowRoot(store: store))
        host.frame = NSRect(x: 0, y: 0, width: NDGrid.windowWidth, height: 600)
        host.layoutSubtreeIfNeeded()
        let probe = try XCTUnwrap(monitor(in: host), "the window root's own swipe monitor")
        let window = NSObject()
        probe.attach(to: window, isVisible: { false }) // routes events, registers nothing
        let at = NSPoint(x: 450, y: 300)

        XCTAssertFalse(probe.route(try scroll(phase: .began), in: window, at: at))
        var travel: CGFloat = 0
        var seen: [Double] = []
        for _ in 0..<30 {
            _ = probe.route(try scroll(phase: .changed, dx: -10), in: window, at: at)
            travel += 10
            seen.append(store.tabs.progress)
            XCTAssertEqual(store.tabs.progress, Double(travel / NDGrid.windowWidth), accuracy: 1e-9,
                           "after \(travel) pt the pages are \(travel) pt along")
            XCTAssertEqual(store.page, .overview, "nothing is selected until the release")
        }
        let pill = metrics.pillFrame(NDPageTabs.tabs(store.r1), progress: store.tabs.progress)
        let slots = metrics.slotFrames(NDPageTabs.tabs(store.r1))
        XCTAssertGreaterThan(pill.minX, slots[0].minX + metrics.inset, "the pill has left Overview's slot with the fingers")
        XCTAssertEqual(store.tabs.phase, .tracking)
        _ = probe.route(try scroll(phase: .ended), in: window, at: at)
        XCTAssertNotEqual(store.tabs.phase, .tracking, "released")
        XCTAssertEqual(seen, seen.sorted(), "monotonic with the fingers")
    }

    // MARK: The window's height holds while two pages share it

    /// QA r2 (1.6.13): the window resized to the new page right after each swap. Now, on the controller's real path, the
    /// window's height never changes on a frame where the pill and pages are between tabs (a live swipe, its settle, a
    /// click and its settle); it moves once, after they rest, on the existing 200 ms resize.
    func testTheWindowHeightHoldsWhileTwoPagesShareIt() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        controller.visibleHeightOverride = 2000
        controller.reduceMotionForTesting = false
        controller.animatesPageSwitchForTesting = true
        let store = controller.newDesignStore
        let window = controller.windowForTesting
        var resizes: [(progress: Double, from: CGFloat, to: CGFloat)] = []
        controller.pageFrameAnimatorForTesting = { target, _, _, finish in
            resizes.append((store.tabs.progress, window.frame.height, target.height))
            window.setFrame(target, display: false)
            finish()
        }
        store.tabs.isLive = { true }
        var time = 100.0
        var fractionalFrames = 0
        func frameCheck(_ name: String) {
            let p = store.tabs.progress
            guard p != p.rounded() else { return }
            fractionalFrames += 1
            XCTAssertTrue(resizes.isEmpty, "\(name): no resize while between tabs (progress \(p))")
        }
        func settle(_ name: String) {
            for _ in 0..<120 where store.tabs.phase != .resting {
                time += frame
                let height = window.frame.height
                store.tabs.tick(at: time)
                if store.tabs.phase != .resting {
                    XCTAssertEqual(window.frame.height, height, "\(name): the window holds its height mid-settle")
                    frameCheck(name)
                }
            }
            XCTAssertEqual(store.tabs.phase, .resting, name)
        }

        // A live swipe Overview → Backups: tracking, then its settle.
        let start = window.frame.height
        store.drivePages(.grab(-30), at: time, width: NDGrid.windowWidth)
        for _ in 0..<40 {
            time += frame
            store.drivePages(.move(-12), at: time, width: NDGrid.windowWidth)
            XCTAssertEqual(window.frame.height, start, "swipe: the window holds its height while the fingers move")
            frameCheck("swipe")
        }
        time += frame
        store.drivePages(.release, at: time, width: NDGrid.windowWidth)
        XCTAssertEqual(store.page, .backups, "the release picked Backups")
        XCTAssertTrue(resizes.isEmpty, "...but the window waits for the rest")
        settle("swipe settle")
        XCTAssertEqual(resizes.count, 1, "then the height moves once")
        XCTAssertEqual(resizes.last?.progress, 1, "from the rested page")
        XCTAssertEqual(store.shownPage, .backups)

        // A click Backups → Jobs, and a click back to Overview mid-settle.
        resizes.removeAll()
        store.tabs.select(2, at: time)
        for _ in 0..<10 {
            time += frame
            store.tabs.tick(at: time)
            frameCheck("click")
        }
        store.tabs.select(0, at: time)
        settle("click back")
        XCTAssertEqual(store.page, .overview)
        XCTAssertEqual(resizes.count, 1, "one resize, after the last settle, to the page it rested on")
        XCTAssertEqual(resizes.last?.progress, 0)
        XCTAssertEqual(window.contentRect(forFrameRect: window.frame).height,
                       NDPageHeight.window(store, page: .overview, screen: controller.fitScreenHeight,
                                           titleBarHeight: controller.fitTitleBarHeight), accuracy: 0.5)
        XCTAssertGreaterThan(fractionalFrames, 60, "the checks ran on frames between tabs")
    }

    // MARK: (d) Reduce Motion

    /// Spec 6: with Reduce Motion the pill (and the progress the pages cross-fade by) never passes its target, however
    /// fast the flick or the interrupted settle.
    func testReduceMotionNeverOvershoots() {
        for script in 0..<3 {
            let pager = GlassTabPager(count: 3, selection: 0)
            pager.isLive = { true }
            pager.reduceMotion = true
            var time = 0.0
            switch script {
            case 0: // a hard flick
                pager.grab(at: time)
                for dx in [-40.0, -80, -120, -160] {
                    time += frame
                    pager.drag(by: -dx / 900, at: time)
                }
                time += frame
                pager.release(at: time)
            case 1: // a click on the far tab
                pager.select(2, at: time)
            default: // a click back during a fast settle
                pager.select(2, at: time)
                for _ in 0..<8 {
                    time += frame
                    pager.tick(at: time)
                }
                pager.select(1, at: time)
            }
            let target = Double(pager.selection)
            let side = (pager.progress - target).sign
            for _ in 0..<120 {
                time += frame
                pager.tick(at: time)
                XCTAssertTrue(pager.progress == target || (pager.progress - target).sign == side,
                              "script \(script): never past \(target) (at \(pager.progress))")
            }
            XCTAssertEqual(pager.progress, target, "script \(script): rests on the target")
        }
        // Without Reduce Motion the same hard flick carries its velocity (the spec's release), which is the contrast.
        let pager = GlassTabPager(count: 3, selection: 0)
        pager.isLive = { true }
        pager.grab(at: 0)
        pager.drag(by: 0.2, at: frame)
        pager.release(at: 2 * frame)
        XCTAssertGreaterThan(pager.velocity, 0, "velocity is carried into the settle")
        // And the rubber band is off: the fingers past an end move nothing.
        let reduced = GlassTabPager(count: 3, selection: 0)
        reduced.reduceMotion = true
        reduced.grab(at: 0)
        reduced.drag(by: -0.3, at: frame)
        XCTAssertEqual(reduced.progress, 0)
    }

    /// Reduce Motion's pages: in place, cross-faded by the same progress (no slide); the slide otherwise.
    func testThePagesOnScreenFollowTheProgress() {
        XCTAssertEqual(NDPageStrip.pages(progress: 0, shown: .overview), [.overview], "at rest only the selected page exists")
        XCTAssertEqual(NDPageStrip.pages(progress: 0.4, shown: .overview), [.overview, .backups])
        XCTAssertEqual(NDPageStrip.pages(progress: 1.5, shown: .jobs), [.backups, .jobs])
        XCTAssertEqual(NDPageStrip.pages(progress: 0.2, shown: .jobs), [.overview, .backups, .jobs],
                       "the page the window is sized for is laid out from the first frame")
        XCTAssertEqual(NDPageStrip.pages(progress: -0.1, shown: .overview), [.overview], "the rubber band shows no page past the end")
    }

    // MARK: Renders (offscreen)

    /// The header at progress 0, 0.5, 1, 1.5, 2 and at rest after a click, dark and light; and one window mid-swipe.
    /// `BRAINBAR_W294_RENDER_DIR` writes the PNGs and the state table.
    func testHeaderRendersAcrossTheSwipe() throws {
        guard let dir = ProcessInfo.processInfo.environment["BRAINBAR_W294_RENDER_DIR"] else { return }
        let url = URL(fileURLWithPath: dir, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var table = ["| state | progress | pill x | pill width | slot frames (x, width) |", "|---|---|---|---|---|"]
        for (suffix, appearance) in NDRenderHarness.appearances {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.twoFaults), page: .overview)
            let states: [(String, Double?)] = [("p0", 0), ("p0.5", 0.5), ("p1", 1), ("p1.5", 1.5), ("p2", 2), ("rest-after-click-jobs", nil)]
            for (name, progress) in states {
                if let progress { store.tabs.present(progress) } else { store.tabs.select(2, at: 0) }
                let size = NSSize(width: 520, height: NDGrid.titleBar)
                for (style, solid) in [("glass", false), ("solid", true)] {
                    let view = NDPageTabs(store: store).fixedSize().environment(\.glassTabSolidPill, solid)
                        .frame(width: size.width, height: size.height).background(ND.bar)
                    let rep = try NDRenderHarness.host(view, size: size, appearance: appearance)
                    try NDRenderHarness.write(rep, name: "after-header-\(name)-\(style)-\(suffix)", in: url)
                }
                if suffix == "dark" {
                    let t = NDPageTabs.tabs(store.r1)
                    let pill = metrics.pillFrame(t, progress: store.tabs.progress)
                    let slots = metrics.slotFrames(t).map { "(\(Int($0.minX)), \(Int($0.width)))" }.joined(separator: " ")
                    table.append("| \(name) | \(store.tabs.progress) | \(pill.minX) | \(pill.width) | \(slots) |")
                }
            }
            let window = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
            window.tabs.present(0.5)
            let size = NSSize(width: NDGrid.windowWidth, height: 640)
            let root = NDWindowRoot(store: window, drawsTrafficLights: true).environment(\.glassTabSolidPill, true)
                .frame(width: size.width, height: size.height)
            try NDRenderHarness.write(try NDRenderHarness.host(root, size: size, appearance: appearance),
                                      name: "after-window-mid-swipe-p0.5-\(suffix)", in: url)
        }
        try table.joined(separator: "\n").write(to: url.appendingPathComponent("state-table.md"), atomically: true, encoding: .utf8)
    }

    // MARK: Helpers

    private func monitor(in view: NSView) -> NDPageSwipeMonitor.Probe? {
        if let probe = view as? NDPageSwipeMonitor.Probe { return probe }
        for sub in view.subviews { if let found = monitor(in: sub) { return found } }
        return nil
    }

    private enum ScrollPhase: Int64 { case began = 1, changed = 2, ended = 4 }
    private func scroll(phase: ScrollPhase, dx: Int32 = 0, dy: Int32 = 0) throws -> NSEvent {
        let cg = try XCTUnwrap(CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 2, wheel1: dy, wheel2: dx, wheel3: 0))
        cg.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
        cg.setIntegerValueField(.scrollWheelEventScrollPhase, value: phase.rawValue)
        return try XCTUnwrap(NSEvent(cgEvent: cg))
    }
}
