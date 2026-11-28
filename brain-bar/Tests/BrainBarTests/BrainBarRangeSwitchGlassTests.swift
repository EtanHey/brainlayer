import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// w316: the Activity range switch (1 h, 3 h, 24 h) on the same glass pill as the page tabs. Offscreen hosting views only.
@MainActor
final class BrainBarRangeSwitchGlassTests: XCTestCase {
    private func store(_ range: NDRange = .hour) -> NDStore {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        store.range = range
        return store
    }

    private func host<V: View>(_ view: V) -> NSHostingView<V> {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 400, height: 40)
        host.layoutSubtreeIfNeeded()
        return host
    }

    private func segmentedControls(in view: NSView) -> [NSSegmentedControl] {
        (view as? NSSegmentedControl).map { [$0] } ?? view.subviews.flatMap(segmentedControls(in:))
    }

    // MARK: The switch is the glass tab bar

    /// One selection style in the window: the range switch draws the page tabs' glass pill, never the system's blue
    /// segmented control. On main the hosted switch held an NSSegmentedControl.
    func testTheRangeSwitchIsTheGlassTabBar() {
        let s = store(.threeHours)
        let hosted = host(NDRangeControl(store: s).environment(\.glassTabSolidPill, true))
        XCTAssertTrue(segmentedControls(in: hosted).isEmpty, "no system segmented control in the range switch")
        let bar = NDRangeControl(store: s).bar
        XCTAssertTrue(bar.pager === s.rangeTabs, "the bar draws the store's range pager")
        XCTAssertEqual(bar.tabs.map(\.title), ["1 h", "3 h", "24 h"])
        XCTAssertTrue(bar.tabs.allSatisfy { $0.count == 0 }, "no counts on a range")
        XCTAssertEqual(bar.metrics, NDRangeControl.metrics)
        // The page tabs' tokens: one track, one pill, one rim.
        XCTAssertEqual(bar.colors.pill, NDPageTabs.colors.pill)
        XCTAssertEqual(bar.colors.track, NDPageTabs.colors.track)
        XCTAssertEqual(bar.colors.pillRim, NDPageTabs.colors.pillRim)
        XCTAssertEqual(bar.colors.selectedText, NDPageTabs.colors.selectedText)
        XCTAssertEqual(bar.colors.text, NDPageTabs.colors.text)
        // The heading's row: the bar is exactly as tall as the 22 pt heading, 11 pt labels as before.
        XCTAssertEqual(NDRangeControl.metrics.height, 22)
        XCTAssertEqual(NDRangeControl.metrics.fontSize, 11)
        XCTAssertEqual(hosted.fittingSize.height, 22)
    }

    /// Fixed slots: every selection lays out every range in the same slot, and the switch never changes size, so the
    /// pill is the only thing that moves.
    func testSlotWidthsAreConstantAcrossSelections() {
        let metrics = NDRangeControl.metrics
        let tabs = NDRangeControl.tabs
        var sizes: [CGSize] = []
        for range in NDRange.allCases {
            let s = store(range)
            XCTAssertEqual(s.rangeTabs.selection, NDRange.allCases.firstIndex(of: range))
            sizes.append(host(NDRangeControl(store: s).environment(\.glassTabSolidPill, true)).fittingSize)
            XCTAssertEqual(metrics.pillFrame(tabs, progress: s.rangeTabs.progress),
                           metrics.slotFrames(tabs)[NDRange.allCases.firstIndex(of: range)!].insetBy(dx: 2, dy: 2),
                           "\(range.label): the pill rests on its slot")
        }
        for size in sizes { XCTAssertEqual(size, sizes[0], "the switch never changes size with the selection") }
        XCTAssertEqual(sizes[0].width, metrics.slotFrames(tabs).last!.maxX)
        for (tab, frame) in zip(tabs, metrics.slotFrames(tabs)) {
            let medium = NSAttributedString(string: tab.title, attributes: [.font: NSFont.systemFont(ofSize: 11, weight: .medium)])
            XCTAssertGreaterThanOrEqual(frame.width, medium.size().width + 2 * metrics.padding, tab.title)
        }
    }

    // MARK: Selecting a range

    /// A plain click changes the chart's range as the pill arrives (within `rangeCommitDistance` of its slot, < 0.30 s
    /// at 120 Hz, for the next range and two over), never on the click's frame or while the pill travels; the pill then
    /// still settles to true rest. Review r1 B2: on 41b899bc the range waited for true rest, ~0.44 s, about 0.2 s after
    /// the pill looked done.
    func testAPlainClickChangesTheChartRangeAsThePillArrives() {
        for (from, to) in [(NDRange.hour, NDRange.threeHours), (.hour, .day), (.day, .hour)] {
            let s = store(from)
            let target = Double(NDRange.allCases.firstIndex(of: to)!)
            s.rangeTabs.isLive = { true }
            let start = 10.0
            var time = start
            s.rangeTabs.select(Int(target), at: time)
            XCTAssertEqual(s.range, from, "\(to.label): not committed on the click's frame")
            var committedAt: Double?
            var previous = abs(s.rangeTabs.progress - target)
            while s.rangeTabs.phase != .resting, time - start < 1 {
                time += 1.0 / 120
                s.rangeTabs.tick(at: time)
                let distance = abs(s.rangeTabs.progress - target)
                XCTAssertLessThanOrEqual(distance, previous + 1e-12, "\(to.label): one settle toward the pick")
                previous = distance
                guard committedAt == nil else { continue }
                if s.range == to {
                    committedAt = time - start
                    XCTAssertLessThan(distance, NDStore.rangeCommitDistance, "\(to.label): committed only once arrived")
                    XCTAssertEqual(s.rangeTabs.phase, .settling, "\(to.label): before the spring's tail reaches rest")
                } else {
                    XCTAssertGreaterThanOrEqual(distance, NDStore.rangeCommitDistance, "\(to.label): travelling, not committed")
                }
            }
            XCTAssertNotNil(committedAt, "\(to.label): the chart's range changed")
            XCTAssertLessThan(committedAt ?? 1, 0.30, "\(from.label) -> \(to.label): the chart answers within 0.30 s")
            XCTAssertLessThan(time - start, 0.5, "the pill still rests within half a second")
            XCTAssertEqual(s.rangeTabs.progress, target)
            XCTAssertEqual(s.range, to)
            print(String(format: "[w316] range click %@ -> %@: chart at %.3f s, pill at rest %.3f s", from.label, to.label,
                         committedAt ?? -1, time - start))
        }

        // Two picks in a row: only the one the pill arrives at reaches the chart.
        let s = store(.hour)
        s.rangeTabs.isLive = { true }
        var time = 30.0
        s.rangeTabs.select(2, at: time)
        time += 1.0 / 120
        s.rangeTabs.tick(at: time)
        s.rangeTabs.select(1, at: time)
        while s.rangeTabs.phase != .resting { time += 1.0 / 120; s.rangeTabs.tick(at: time) }
        XCTAssertEqual(s.range, .threeHours)

        // Not drawn (a closed window): the pick lands at once.
        let closed = store(.hour)
        closed.rangeTabs.select(1, at: 0)
        XCTAssertEqual(closed.range, .threeHours)
        XCTAssertEqual(closed.rangeTabs.progress, 1)
    }

    /// The guard #1896 learned on the M1: the store publishes nothing before the range pill arrives (a publication on
    /// the click's frame re-evaluates every observer of the store as the motion starts), and does publish once it has.
    func testTheStorePublishesNothingBeforeTheRangePillArrives() {
        let s = store(.hour)
        s.rangeTabs.isLive = { true }
        var publications = 0
        let sink = s.objectWillChange.sink { publications += 1 }
        defer { sink.cancel() }
        var time = 3.0
        s.rangeTabs.select(2, at: time)
        XCTAssertEqual(publications, 0, "the click's frame publishes nothing")
        while s.rangeTabs.phase != .resting {
            time += 1.0 / 120
            s.rangeTabs.tick(at: time)
            if abs(s.rangeTabs.progress - 2) >= NDStore.rangeCommitDistance {
                XCTAssertEqual(publications, 0, "nothing published before the pill arrives")
            }
        }
        XCTAssertGreaterThan(publications, 0, "the range publishes once the pill has arrived")
        XCTAssertEqual(s.range, .day)
    }

    /// The page tabs keep their contract: no commit distance, the page reaches the store only at true rest.
    func testThePageTabsStillCommitAtTrueRest() {
        let s = store(.hour)
        XCTAssertNil(s.tabs.commitDistance)
        s.tabs.isLive = { true }
        var time = 7.0
        s.tabs.select(2, at: time)
        while s.tabs.phase != .resting {
            XCTAssertEqual(s.page, .overview, "no page commit before true rest (progress \(s.tabs.progress))")
            time += 1.0 / 120
            s.tabs.tick(at: time)
        }
        XCTAssertEqual(s.page, .jobs)
    }

    /// Review r1 B1: an outside assignment equal to the chart's current range, made while a different pick is still
    /// settling, wins. On 41b899bc the didSet's equality guard skipped `follow`, so the pending 24 h landed over it.
    func testAnOutsideRangeEqualToTheCurrentOneWinsOverAPendingPick() {
        let s = store(.hour)
        s.rangeTabs.isLive = { true }
        // The media clock: an outside `range` reaches the pager through `follow`, which stamps its settle with it.
        var time = CACurrentMediaTime()
        s.rangeTabs.select(2, at: time)
        for _ in 0..<6 { time += 1.0 / 120; s.rangeTabs.tick(at: time) }
        XCTAssertEqual(s.range, .hour, "the 24 h pick is still travelling")
        s.range = .hour
        XCTAssertEqual(s.rangeTabs.selection, 0, "the pill turns back to 1 h")
        while s.rangeTabs.phase != .resting { time += 1.0 / 120; s.rangeTabs.tick(at: time) }
        XCTAssertEqual(s.range, .hour, "store: the outside request stands")
        XCTAssertEqual(s.rangeTabs.selection, 0, "pill selection: 1 h")
        XCTAssertEqual(s.rangeTabs.progress, 0, "pill position: 1 h")
        XCTAssertEqual(s.range.axis.0, "60 min ago", "chart: the Activity chart draws store.range, the 1 h axis")
        XCTAssertEqual(NDRangeControl(store: s).bar.accessibilitySelection.wrappedValue, 0, "VoiceOver: 1 h")
    }

    /// The pill dragged and let go over a slot picks that range, like the page tabs.
    func testADraggedPillPicksTheRangeItIsLetGoOn() {
        let s = store(.hour)
        let metrics = NDRangeControl.metrics, tabs = NDRangeControl.tabs
        let pill = metrics.pillFrame(tabs, progress: 0)
        var press = GlassTabPress.begin(at: CGPoint(x: pill.midX, y: 11), pager: s.rangeTabs, tabs: tabs, metrics: metrics, time: 1)
        XCTAssertTrue(press.onPill)
        let target = metrics.slotFrames(tabs)[2].midX - pill.midX
        press.move(translation: target, pager: s.rangeTabs, tabs: tabs, metrics: metrics, time: 1.2)
        press.end(at: CGPoint(x: pill.midX + target, y: 11), pager: s.rangeTabs, tabs: tabs, metrics: metrics, time: 1.4)
        XCTAssertEqual(s.range, .day)
    }

    /// A range set anywhere else (the capture driver) moves the pill without echoing back.
    func testARangeSetElsewhereMovesThePill() {
        let s = store(.hour)
        s.range = .threeHours
        XCTAssertEqual(s.rangeTabs.selection, 1)
        XCTAssertEqual(s.rangeTabs.progress, 1, "not drawn: lands at once")
        XCTAssertEqual(s.range, .threeHours)
    }

    // MARK: VoiceOver and Reduce Motion

    /// VoiceOver reads a picker named "Range" with the three ranges, its selected value the store's range; a pick there
    /// changes the range like a click.
    func testVoiceOverReadsAPickerWithTheSelectedRange() {
        let s = store(.threeHours)
        let bar = NDRangeControl(store: s).bar
        XCTAssertEqual(bar.accessibilityLabel, "Range")
        XCTAssertEqual(bar.tabs.map(\.spokenLabel), ["1 h", "3 h", "24 h"], "the labels VoiceOver read before")
        XCTAssertEqual(bar.accessibilitySelection.wrappedValue, 1, "the selected value is the store's range")
        bar.accessibilitySelection.wrappedValue = 2
        XCTAssertEqual(s.range, .day, "a VoiceOver pick changes the range")
        XCTAssertEqual(bar.accessibilitySelection.wrappedValue, 2)
        s.range = .hour
        XCTAssertEqual(bar.accessibilitySelection.wrappedValue, 0, "a range set elsewhere is what VoiceOver reads")
    }

    /// Reduce Motion: the pill moves without overshoot, even when a click retargets it mid-flight.
    func testReduceMotionNeverOvershoots() {
        let s = store(.hour)
        s.rangeTabs.isLive = { true }
        s.rangeTabs.reduceMotion = true
        var time = 5.0
        s.rangeTabs.select(2, at: time)
        for _ in 0..<12 { time += 1.0 / 120; s.rangeTabs.tick(at: time) }
        s.rangeTabs.select(1, at: time)
        XCTAssertEqual(s.range, .hour, "the interrupted pick never reached the chart")
        let start = s.rangeTabs.progress
        XCTAssertGreaterThan(start, 1, "mid-flight past the new target")
        while s.rangeTabs.phase != .resting {
            time += 1.0 / 120
            s.rangeTabs.tick(at: time)
            XCTAssertGreaterThanOrEqual(s.rangeTabs.progress, 1 - 1e-12, "never past 1 h's neighbour, 3 h")
            XCTAssertLessThanOrEqual(s.rangeTabs.progress, start + 1e-12, "never back toward 24 h")
        }
        XCTAssertEqual(s.rangeTabs.progress, 1)
        XCTAssertEqual(s.range, .threeHours)
    }

    // MARK: Renders (offscreen)

    /// The Activity block with each range selected, dark and light. `BRAINBAR_W316_RENDER_DIR` writes the PNGs and the
    /// state table; `BRAINBAR_W316_RENDER_PREFIX` names the side (before / after).
    func testRangeSwitchRenders() throws {
        guard let dir = ProcessInfo.processInfo.environment["BRAINBAR_W316_RENDER_DIR"] else { return }
        let prefix = ProcessInfo.processInfo.environment["BRAINBAR_W316_RENDER_PREFIX"] ?? "after"
        let url = URL(fileURLWithPath: dir, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var table = ["| side | appearance | range | control size (pt) |", "|---|---|---|---|"]
        for (suffix, appearance) in NDRenderHarness.appearances {
            for range in NDRange.allCases {
                let s = store(range)
                _ = try NDRenderHarness.activity(s, hover: nil, name: "\(prefix)-activity-\(range.rawValue)-\(suffix)",
                                                 appearance: appearance, in: url)
                let size = NSSize(width: 200, height: 40)
                let view = NDRangeControl(store: s).environment(\.glassTabSolidPill, true)
                    .padding(9).frame(width: size.width, height: size.height).background(ND.bg)
                try NDRenderHarness.write(try NDRenderHarness.host(view, size: size, appearance: appearance),
                                          name: "\(prefix)-switch-\(range.rawValue)-\(suffix)", in: url)
                let fitted = host(NDRangeControl(store: s).environment(\.glassTabSolidPill, true)).fittingSize
                table.append("| \(prefix) | \(suffix) | \(range.label) | \(fitted.width) × \(fitted.height) |")
            }
        }
        try table.joined(separator: "\n").write(to: url.appendingPathComponent("\(prefix)-state-table.md"), atomically: true,
                                                 encoding: .utf8)
    }
}
