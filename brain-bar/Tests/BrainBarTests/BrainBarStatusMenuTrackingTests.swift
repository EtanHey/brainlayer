import AppKit
import XCTest
@testable import BrainBar

@MainActor final class BrainBarStatusMenuTrackingTests: XCTestCase {
    private final class Item: BrainBarStatusItem {
        var length: CGFloat = NSStatusItem.variableLength
        var menu: NSMenu?
        let button: NSButton? = NSButton(frame: .zero)
    }
    private final class Host: BrainBarStatusItemHosting {
        let item = Item()
        var removals = 0
        func statusItem(withLength: CGFloat) -> any BrainBarStatusItem { item }
        func removeStatusItem(_ item: any BrainBarStatusItem) { removals += 1 }
    }

    private func fixture() throws -> (BrainBarStatusPopoverController, NDStore, Host) {
        NSApplication.shared.setActivationPolicy(.prohibited)
        setenv(BrainBarDesignFixture.environmentKey, "calm", 1)
        defer { unsetenv(BrainBarDesignFixture.environmentKey) }
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let host = Host()
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: host)
        return (controller, try XCTUnwrap(NDStore.existing(for: runtime)), host)
    }

    private func pump() { RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.05)) }

    func testInitialMenuHasSettingsAndQuitBeforeAnyOpen() throws {
        let (controller, _, host) = try fixture()
        defer { controller.stop() }
        let menu = controller.contextMenuForTesting
        XCTAssertTrue(host.item.menu === menu)
        for action in [BrainBarMenuAction.settings, .quit] {
            let item = try XCTUnwrap(menu.items.first { ($0.representedObject as? BrainBarMenuAction) == action },
                                     "The initial menu must offer \(action) before AppKit begins tracking")
            XCTAssertTrue(item.isEnabled)
            XCTAssertNotNil(item.action)
            XCTAssertNotNil(item.target)
        }
    }

    func testNewIconAndTooltipStayFrozenThroughTrackingThenShowLatestState() throws {
        let (controller, store, host) = try fixture()
        defer { controller.stop() }
        pump()
        let menu = controller.contextMenuForTesting
        controller.menuWillOpen(menu)
        let image = host.item.button?.image
        let tooltip = host.item.button?.toolTip
        store.setInputsForTesting(BrainBarDesignFixture.inputs(.twoFaults))
        controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.stats, state: .idle)
        pump()
        XCTAssertTrue(host.item.button?.image === image)
        XCTAssertEqual(host.item.button?.toolTip, tooltip)
        controller.menuDidClose(menu)
        XCTAssertTrue(host.item.button?.image === image, "tracking teardown finishes before the redraw")
        pump()
        XCTAssertFalse(host.item.button?.image === image)
        XCTAssertEqual(host.item.button?.toolTip, store.r1.headline)
    }

    func testImmediateReopenAndStopDiscardThePendingRedraw() throws {
        let (controller, store, host) = try fixture()
        pump()
        let menu = controller.contextMenuForTesting
        controller.menuWillOpen(menu)
        let image = host.item.button?.image
        store.setInputsForTesting(BrainBarDesignFixture.inputs(.twoFaults))
        controller.menuDidClose(menu)
        controller.menuWillOpen(menu)
        pump()
        XCTAssertTrue(host.item.button?.image === image)
        controller.menuDidClose(menu)
        controller.stop()
        pump()
        XCTAssertTrue(host.item.button?.image === image)
        XCTAssertNil(menu.delegate)
        XCTAssertNil(host.item.menu)
        XCTAssertEqual(host.removals, 1)
    }
}
