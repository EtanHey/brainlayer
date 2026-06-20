import AppKit
#if BRAINBAR_UI
import SwiftUI
#endif

@MainActor
enum BrainBarSettingsActions {
    private(set) static var suppressDashboardResignDismiss = false

#if BRAINBAR_UI
    private static var windowController: NSWindowController?
    private static var closeObserver: NSObjectProtocol?

    static func openSettingsWindow() {
        NSApp.activate(ignoringOtherApps: true)

        if let controller = windowController {
            promoteForSettings()
            controller.window?.makeKeyAndOrderFront(nil)
            return
        }

        let hosting = NSHostingController(rootView: BrainBarSettingsView())
        let window = NSWindow(contentViewController: hosting)
        window.title = "BrainLayer Settings"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.isReleasedWhenClosed = false
        window.titlebarAppearsTransparent = false
        window.level = .normal
        window.center()

        let controller = NSWindowController(window: window)
        windowController = controller

        closeObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification,
            object: window,
            queue: .main
        ) { _ in
            Task { @MainActor in demoteAfterSettings() }
        }

        promoteForSettings()
        controller.showWindow(nil)
        window.makeKeyAndOrderFront(nil)
    }

    private static func promoteForSettings() {
        suppressDashboardResignDismiss = true
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    private static func demoteAfterSettings() {
        suppressDashboardResignDismiss = false
        if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
        closeObserver = nil
        windowController = nil
        NSApp.setActivationPolicy(.accessory)
    }
#else
    // BrainBarDaemon target is headless and never opens Settings (the file is shared
    // via symlink, but BrainBarSettingsView + its deps live only in the BrainBar target).
    // Keep the original no-op so the daemon target compiles.
    static func openSettingsWindow() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
    }
#endif
}
