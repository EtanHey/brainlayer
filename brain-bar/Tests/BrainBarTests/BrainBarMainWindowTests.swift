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

    /// A large display, so nothing here depends on the machine running the tests.
    private static let bigScreen = [NSRect(x: 0, y: 0, width: 2_560, height: 1_400)]

    private func controller(
        defaults: MemoryDefaults = MemoryDefaults(),
        screens: [NSRect] = BrainBarMainWindowTests.bigScreen
    ) -> BrainBarDashboardPanelController {
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime(),
            frameStore: BrainBarWindowFrameStore(defaults: defaults, key: BrainBarDashboardPanelController.frameDefaultsKey),
            screenFrames: { screens }
        )
        controller.visibleFrameOverride = screens.last
        return controller
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
        XCTAssertEqual(controller.windowWillResize(window, to: .zero), controller.fittedFrameSize)
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
        defer { controller.dismiss(); NDSettingsWindowController.closeIfOpen() }
        controller.showDashboard()
        let window = controller.windowForTesting

        XCTAssertTrue(window.performKeyEquivalent(with: keyDown(",", in: window)))
        XCTAssertTrue(NDSettingsWindowController.isOpenForTesting)
        XCTAssertTrue(controller.isShownForTesting)

        XCTAssertTrue(window.performKeyEquivalent(with: keyDown("w", in: window)))
        XCTAssertFalse(controller.isShownForTesting)
        XCTAssertFalse(window.performKeyEquivalent(with: keyDown("w", modifiers: [], in: window)), "a bare w is typing")
    }

    /// Position and size are saved and restored against injected screens. The window is placed
    /// without being ordered front, so the real display (a small one on the CI runner) plays no
    /// part (#1020 CI).

    func test_late_geometry_events_after_close_do_not_replace_the_saved_external_display_position() {
        let defaults = MemoryDefaults()
        let screens = [NSRect(x: 0, y: 0, width: 1728, height: 1084),
            NSRect(x: 1728, y: 210, width: 1920, height: 1050)]
        let saved = NSRect(x: 2338, y: 399, width: 900, height: 640)
        let store = BrainBarWindowFrameStore(defaults: defaults, key: BrainBarDashboardPanelController.frameDefaultsKey)
        store.persist(frame: saved)
        let first = controller(defaults: defaults, screens: screens)
        first.placeWindowForTesting()
        let placed = first.windowForTesting.frame
        XCTAssertEqual(placed.minX, saved.minX)
        XCTAssertEqual(placed.maxY, saved.maxY)
        first.dismiss()

        // AppKit can deliver geometry changes during teardown/policy changes. Once closed,
        // these are no longer a user placement and must not replace the saved frame.
        first.windowForTesting.setFrame(NSRect(x: 100, y: 100, width: 900, height: 640), display: false)
        first.windowDidMove(Notification(name: NSWindow.didMoveNotification, object: first.windowForTesting))
        first.windowDidEndLiveResize(Notification(name: NSWindow.didEndLiveResizeNotification,
            object: first.windowForTesting))
        first.dismiss() // Termination can close through both status.stop() and the delegate.
        XCTAssertEqual(store.persistedFrame(), placed)

        let relaunched = controller(defaults: defaults, screens: screens)
        relaunched.placeWindowForTesting()
        XCTAssertEqual(relaunched.windowForTesting.frame, placed)
        let detached = controller(defaults: defaults, screens: [screens[0]])
        detached.placeWindowForTesting()
        XCTAssertNotEqual(detached.windowForTesting.frame.origin, saved.origin)
    }

    func test_a_saved_frame_is_kept_clamped_or_dropped_for_the_current_screens() {
        let minimum = BrainBarDashboardPanelController.minSize
        let screen = NSRect(x: 0, y: 0, width: 1_440, height: 875)
        func restore(_ frame: NSRect, _ screens: [NSRect] = [screen]) -> NSRect? {
            BrainBarDashboardPanelController.restoredFrame(frame, minSize: minimum, visibleFrames: screens)
        }
        // Fits: unchanged.
        XCTAssertEqual(restore(NSRect(x: 100, y: 50, width: 900, height: 640)), NSRect(x: 100, y: 50, width: 900, height: 640))
        // Hangs off the right and top: moved inside, size kept.
        XCTAssertEqual(restore(NSRect(x: 1_000, y: 600, width: 900, height: 640)), NSRect(x: 540, y: 235, width: 900, height: 640))
        // Taller and wider than the screen: shrunk to it, never below the minimum.
        XCTAssertEqual(restore(NSRect(x: 0, y: 0, width: 2_000, height: 1_200)), NSRect(x: 0, y: 0, width: 1_440, height: 875))
        let tiny = NSRect(x: 0, y: 0, width: 700, height: 500)
        XCTAssertEqual(restore(NSRect(x: 0, y: 0, width: 900, height: 640), [tiny])?.size, minimum)
        // On the second display it overlaps most.
        let second = NSRect(x: 1_440, y: 0, width: 1_920, height: 1_055)
        XCTAssertEqual(restore(NSRect(x: 1_500, y: 100, width: 900, height: 640), [screen, second]), NSRect(x: 1_500, y: 100, width: 900, height: 640))
        // No longer on any screen, or smaller than the minimum: not restored.
        XCTAssertNil(restore(NSRect(x: 5_000, y: 5_000, width: 900, height: 640)))
        XCTAssertNil(restore(NSRect(x: 100, y: 100, width: 300, height: 200)))
    }

    // MARK: status item

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
