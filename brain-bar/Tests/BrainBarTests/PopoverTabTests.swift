import AppKit
import XCTest
@testable import BrainBar

final class PopoverTabTests: XCTestCase {
    private var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-popover-tab-\(UUID().uuidString).db"
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    // MARK: - PopoverTab Enum

    func testPopoverTabHasThreeCases() {
        XCTAssertEqual(PopoverTab.allCases.count, 3)
    }

    func testPopoverTabRawValues() {
        XCTAssertEqual(PopoverTab.dashboard.rawValue, 0)
        XCTAssertEqual(PopoverTab.injections.rawValue, 1)
        XCTAssertEqual(PopoverTab.graph.rawValue, 2)
    }

    func testPopoverTabLabels() {
        XCTAssertEqual(PopoverTab.dashboard.label, "Dashboard")
        XCTAssertEqual(PopoverTab.injections.label, "Injections")
        XCTAssertEqual(PopoverTab.graph.label, "Graph")
    }

    func testPopoverTabDashboardSizeIsCompact() {
        XCTAssertEqual(PopoverTab.dashboard.contentSize.width, 360)
    }

    func testPopoverTabGraphSizeIsWiderThanDashboard() {
        XCTAssertGreaterThan(
            PopoverTab.graph.contentSize.width,
            PopoverTab.dashboard.contentSize.width
        )
    }

    func testPopoverTabGraphSizeIsTallerThanDashboard() {
        XCTAssertGreaterThan(
            PopoverTab.graph.contentSize.height,
            PopoverTab.dashboard.contentSize.height
        )
    }

    // MARK: - StatusPopoverView Tab Integration

    @MainActor
    func testStatusPopoverViewHasSegmentedControl() {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(collector: collector)
        _ = vc.view

        let segmented = findSegmentedControl(in: vc.view)
        XCTAssertNotNil(segmented, "Popover should contain a segmented control")
        XCTAssertEqual(segmented?.segmentCount, 3)
    }

    @MainActor
    func testSegmentedControlDisablesTabsWithoutDependencies() {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(collector: collector)
        _ = vc.view

        let segmented = findSegmentedControl(in: vc.view)!
        XCTAssertTrue(segmented.isEnabled(forSegment: 0), "Dashboard always enabled")
        XCTAssertFalse(segmented.isEnabled(forSegment: 1), "Injections disabled without store")
        XCTAssertFalse(segmented.isEnabled(forSegment: 2), "Graph disabled without database")
    }

    @MainActor
    func testSegmentedControlEnablesTabsWithDependencies() throws {
        let db = BrainDatabase(path: tempDBPath)
        defer { db.close() }
        let injStore = try InjectionStore(databasePath: tempDBPath)
        defer { injStore.stop() }

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(
            collector: collector,
            injectionStore: injStore,
            database: db
        )
        _ = vc.view

        let segmented = findSegmentedControl(in: vc.view)!
        XCTAssertTrue(segmented.isEnabled(forSegment: 0))
        XCTAssertTrue(segmented.isEnabled(forSegment: 1))
        XCTAssertTrue(segmented.isEnabled(forSegment: 2))
    }

    @MainActor
    func testPreferredContentSizeMatchesDashboardOnLoad() {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(collector: collector)
        _ = vc.view

        XCTAssertEqual(vc.preferredContentSize, PopoverTab.dashboard.contentSize)
    }

    @MainActor
    func testShowTabUpdatesPreferredContentSize() throws {
        let db = BrainDatabase(path: tempDBPath)
        defer { db.close() }
        let injStore = try InjectionStore(databasePath: tempDBPath)
        defer { injStore.stop() }

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(
            collector: collector,
            injectionStore: injStore,
            database: db
        )
        _ = vc.view

        vc.showTab(.injections)
        XCTAssertEqual(vc.preferredContentSize, PopoverTab.injections.contentSize)

        vc.showTab(.graph)
        XCTAssertEqual(vc.preferredContentSize, PopoverTab.graph.contentSize)

        vc.showTab(.dashboard)
        XCTAssertEqual(vc.preferredContentSize, PopoverTab.dashboard.contentSize)
    }

    @MainActor
    func testCurrentTabTracksSelection() throws {
        let db = BrainDatabase(path: tempDBPath)
        defer { db.close() }

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(collector: collector, database: db)
        _ = vc.view

        XCTAssertEqual(vc.currentTab, .dashboard)

        vc.showTab(.graph)
        XCTAssertEqual(vc.currentTab, .graph)
    }

    @MainActor
    func testExistingInitStillWorksWithoutNewParams() {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let vc = StatusPopoverView(collector: collector)
        XCTAssertFalse(vc.isViewLoaded)
        _ = vc.view
        XCTAssertTrue(vc.isViewLoaded)
    }

    // MARK: - Helpers

    private func findSegmentedControl(in view: NSView) -> NSSegmentedControl? {
        for subview in view.subviews {
            if let segmented = subview as? NSSegmentedControl {
                return segmented
            }
            if let found = findSegmentedControl(in: subview) {
                return found
            }
        }
        return nil
    }
}
