import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarWindowActivationPolicyTests: XCTestCase {
    func testOpenCloseAndReopenChangeTheApplicationPolicyAfterTheWindowCloses() {
        let app = NSApplication.shared
        let previousPolicy = app.activationPolicy()
        app.setActivationPolicy(.accessory)
        var transitions: [NSApplication.ActivationPolicy] = []
        weak var observedWindow: NSWindow?
        let policy = BrainBarWindowActivationPolicy { value in
            if value == .accessory {
                XCTAssertFalse(observedWindow?.isVisible ?? true, "close must finish before demoting the app")
            }
            transitions.append(value)
            app.setActivationPolicy(value)
        }
        let controller = makeController(policy: policy)
        observedWindow = controller.windowForTesting
        defer {
            controller.dismiss()
            app.setActivationPolicy(previousPolicy)
            BrainBarSettingsActions.installOpenHandler {}
        }
        let window = controller.windowForTesting

        controller.showDashboard()
        XCTAssertEqual(app.activationPolicy(), .regular)
        XCTAssertTrue(window.isVisible)
        let frame = window.frame
        controller.showSettings()
        XCTAssertEqual(transitions, [.regular], "showing the same window twice is one open")

        window.performClose(nil) // Title-bar close and Cmd-W both use this path.
        XCTAssertEqual(app.activationPolicy(), .accessory)
        XCTAssertFalse(window.isVisible)
        window.setFrameOrigin(NSPoint(x: frame.minX + 30, y: frame.minY + 30))

        controller.showSettings()
        XCTAssertEqual(app.activationPolicy(), .regular)
        XCTAssertTrue(window.isVisible)
        XCTAssertEqual(window.frame, frame)
        controller.dismiss() // Hotkey/menu toggle uses the same close path.
        XCTAssertEqual(app.activationPolicy(), .accessory)
        XCTAssertEqual(transitions, [.regular, .accessory, .regular, .accessory])
    }

    func testFocusLossAndMinimizationKeepTheWindowRegistered() {
        var transitions: [NSApplication.ActivationPolicy] = []
        let policy = BrainBarWindowActivationPolicy { transitions.append($0) }
        let controller = makeController(policy: policy)
        defer {
            controller.dismiss()
            BrainBarSettingsActions.installOpenHandler {}
        }
        controller.showDashboard()
        let window = controller.windowForTesting

        window.resignKey()
        NotificationCenter.default.post(name: NSWindow.didResignKeyNotification, object: window)
        NotificationCenter.default.post(name: NSApplication.didResignActiveNotification, object: NSApp)
        window.miniaturize(nil)
        XCTAssertEqual(transitions, [.regular])
        controller.show()
        XCTAssertFalse(window.isMiniaturized)
        XCTAssertTrue(window.isVisible)
        XCTAssertEqual(transitions, [.regular])
    }

    func testOnlyClosingTheLastRegisteredWindowReturnsToAccessory() {
        var transitions: [NSApplication.ActivationPolicy] = []
        let policy = BrainBarWindowActivationPolicy { transitions.append($0) }
        let first = NSWindow()
        let second = NSWindow()
        policy.windowOpened(first)
        policy.windowOpened(second)
        policy.windowOpened(first)
        policy.windowClosed(first)
        policy.windowClosed(first)
        XCTAssertEqual(transitions, [.regular])
        policy.windowClosed(second)
        XCTAssertEqual(transitions, [.regular, .accessory])
    }

    func testDockReopenBringsTheSameWindowBackWithoutChangingItsSelectedPage() {
        var transitions: [NSApplication.ActivationPolicy] = []
        let policy = BrainBarWindowActivationPolicy { transitions.append($0) }
        let controller = makeController(policy: policy)
        let delegate = AppDelegate(dashboardPanel: controller)
        defer {
            controller.dismiss()
            BrainBarSettingsActions.installOpenHandler {}
        }
        let window = controller.windowForTesting
        controller.showSettings()
        window.orderBack(nil)

        XCTAssertFalse(delegate.applicationShouldHandleReopen(NSApplication.shared, hasVisibleWindows: true))
        XCTAssertTrue(window.isVisible)
        XCTAssertEqual(controller.selectedTabForTesting, .settings)
        XCTAssertEqual(transitions, [.regular])

        controller.dismiss()
        XCTAssertFalse(delegate.applicationShouldHandleReopen(NSApplication.shared, hasVisibleWindows: false))
        XCTAssertTrue(controller.windowForTesting === window)
        XCTAssertTrue(window.isVisible)
        XCTAssertEqual(controller.selectedTabForTesting, .settings)
        XCTAssertEqual(transitions, [.regular, .accessory, .regular])
        XCTAssertFalse(delegate.applicationShouldTerminateAfterLastWindowClosed(NSApplication.shared))
    }

    func testRenderControllersDoNotOverrideTheirHarnessActivationPolicy() {
        let app = NSApplication.shared
        let previousPolicy = app.activationPolicy()
        defer {
            app.setActivationPolicy(previousPolicy)
            BrainBarSettingsActions.installOpenHandler {}
        }
        for harnessPolicy: NSApplication.ActivationPolicy in [.prohibited, .regular] {
            app.setActivationPolicy(harnessPolicy)
            let controller = makeController()
            controller.showDashboard()
            XCTAssertEqual(app.activationPolicy(), harnessPolicy)
            controller.dismiss()
            XCTAssertEqual(app.activationPolicy(), harnessPolicy)
        }
    }

    private func makeController(policy: BrainBarWindowActivationPolicy? = nil) -> BrainBarDashboardPanelController {
        BrainBarDashboardPanelController(
            workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime(), activationPolicy: policy
        )
    }
}
