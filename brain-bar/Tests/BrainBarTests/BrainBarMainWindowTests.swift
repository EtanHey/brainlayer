import AppKit
import XCTest
@testable import BrainBar

/// #963 PR 1 / vNext D5: BrainBar is a small menu-bar status item, and Dashboard + Settings live
/// in one REAL window. The old anchored panel closed on any click away, which broke
/// click-through testing; this window stays until it is closed, like VoiceBar's Settings window.
@MainActor
final class BrainBarMainWindowTests: XCTestCase {
    private final class MemoryDefaults: BrainBarKeyValueStoring {
        var values: [String: String] = [:]
        func string(forKey defaultName: String) -> String? { values[defaultName] }
        func setString(_ value: String?, forKey defaultName: String) { values[defaultName] = value }
    }

    private func controller(defaults: MemoryDefaults = MemoryDefaults()) -> BrainBarDashboardPanelController {
        BrainBarDashboardPanelController(
            runtime: BrainBarRuntime(),
            frameStore: BrainBarWindowFrameStore(defaults: defaults, key: BrainBarDashboardPanelController.frameDefaultsKey)
        )
    }

    private func keyDown(_ characters: String, modifiers: NSEvent.ModifierFlags = .command, in window: NSWindow) -> NSEvent {
        NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: modifiers, timestamp: 0,
            windowNumber: window.windowNumber, context: nil, characters: characters,
            charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0
        )!
    }

    func test_the_window_is_a_real_resizable_window_not_a_floating_panel() {
        let controller = controller()
        let window = controller.windowForTesting

        XCTAssertFalse(window is NSPanel, "a panel floats above everything and is dismissed like a popover")
        XCTAssertEqual(window.level, .normal)
        XCTAssertTrue(window.canBecomeMain)
        XCTAssertTrue(window.canBecomeKey)
        XCTAssertFalse(window.hidesOnDeactivate)
        XCTAssertFalse(window.isReleasedWhenClosed, "closing hides the window; the status item reopens the same one")
        for style: NSWindow.StyleMask in [.titled, .closable, .miniaturizable, .resizable] {
            XCTAssertTrue(window.styleMask.contains(style), "missing \(style)")
        }
        XCTAssertEqual(window.minSize, NSSize(width: 760, height: 560))
        XCTAssertEqual(window.title, "BrainBar")
    }

    func test_the_window_opens_without_a_status_item_anchor() {
        let controller = controller()
        defer { controller.dismiss() }

        controller.toggle()
        XCTAssertTrue(controller.isShownForTesting, "the hotkey and brainbar://toggle open the window with no anchor")
        controller.toggle()
        XCTAssertFalse(controller.isShownForTesting, "toggle closes the open window")
        controller.showDashboard()
        XCTAssertTrue(controller.isShownForTesting)
    }

    func test_clicking_away_does_not_close_the_window() {
        let controller = controller()
        defer { controller.dismiss() }
        controller.showDashboard()
        // The old panel ignored focus loss for its first 0.2 s, then dismissed itself.
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))

        controller.windowForTesting.resignKey()
        NotificationCenter.default.post(name: NSWindow.didResignKeyNotification, object: controller.windowForTesting)
        NotificationCenter.default.post(name: NSApplication.didResignActiveNotification, object: NSApp)
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        XCTAssertTrue(controller.isShownForTesting, "losing focus must not dismiss the window")
    }

    func test_command_w_closes_and_command_comma_opens_settings() {
        let controller = controller()
        defer { controller.dismiss() }
        controller.showDashboard()
        let window = controller.windowForTesting

        XCTAssertTrue(window.performKeyEquivalent(with: keyDown(",", in: window)))
        XCTAssertEqual(controller.selectedTabForTesting, .settings)
        XCTAssertTrue(controller.isShownForTesting)

        XCTAssertTrue(window.performKeyEquivalent(with: keyDown("w", in: window)))
        XCTAssertFalse(controller.isShownForTesting)
        XCTAssertFalse(window.performKeyEquivalent(with: keyDown("w", modifiers: [], in: window)), "a bare w is typing")
    }

    func test_the_window_resizes_on_both_axes_down_to_its_minimum() {
        let controller = controller()
        let window = controller.windowForTesting

        let taller = controller.windowWillResize(window, to: NSSize(width: 1_100, height: 900))
        XCTAssertEqual(taller, NSSize(width: 1_100, height: 900))
        let tiny = controller.windowWillResize(window, to: NSSize(width: 300, height: 200))
        XCTAssertEqual(tiny, NSSize(width: 760, height: 560))
    }

    func test_position_and_size_are_restored_on_the_next_launch() throws {
        let defaults = MemoryDefaults()
        let first = controller(defaults: defaults)
        first.showDashboard()
        let visible = try XCTUnwrap(NSScreen.screens.first?.visibleFrame)
        let placed = NSRect(x: visible.minX + 40, y: visible.minY + 40, width: 1_000, height: 700)
        first.windowForTesting.setFrame(placed, display: false)
        first.windowDidEndLiveResize(Notification(name: NSWindow.didEndLiveResizeNotification, object: first.windowForTesting))
        first.dismiss()
        XCTAssertNotNil(defaults.values[BrainBarDashboardPanelController.frameDefaultsKey])

        let relaunched = controller(defaults: defaults)
        defer { relaunched.dismiss() }
        relaunched.showDashboard()
        XCTAssertEqual(relaunched.windowForTesting.frame, placed)
    }

    // MARK: status item

    func test_the_status_item_opens_a_small_menu_like_voicebar() throws {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(runtime: runtime)
        let status = BrainBarStatusPopoverController(runtime: runtime, dashboardPanelController: windowController)
        defer { status.stop(); windowController.dismiss() }

        XCTAssertTrue(status.statusItemForTesting.menu === status.contextMenuForTesting, "every click opens the menu")
        let titles = status.contextMenuForTesting.items.filter { !$0.isSeparatorItem }.map(\.title)
        XCTAssertEqual(Array(titles.dropFirst()), ["Open Dashboard", "Settings…", "Restart BrainBar", "Quit BrainBar"])
        XCTAssertFalse(status.contextMenuForTesting.items[0].isEnabled, "the first row is a status line")

        let open = try XCTUnwrap(status.contextMenuForTesting.items.first { $0.title == "Open Dashboard" })
        _ = (open.target as AnyObject).perform(open.action, with: open)
        XCTAssertTrue(windowController.isShownForTesting)
        XCTAssertEqual(windowController.selectedTabForTesting, .dashboard)

        let settings = try XCTUnwrap(status.contextMenuForTesting.items.first { $0.title == "Settings…" })
        _ = (settings.target as AnyObject).perform(settings.action, with: settings)
        XCTAssertEqual(windowController.selectedTabForTesting, .settings)
    }

    func test_the_status_line_says_whether_anything_needs_attention() {
        XCTAssertEqual(
            BrainBarStatusPopoverController.statusLineTitle(for: .failVisible("Watcher stopped")),
            "Needs attention: Watcher stopped"
        )
        XCTAssertEqual(
            BrainBarStatusPopoverController.statusLineTitle(for: .init(badgeOn: false, reason: "", activeCodes: [])),
            "Nothing needs attention"
        )
    }
}
