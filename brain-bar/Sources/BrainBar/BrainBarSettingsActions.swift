import AppKit
#if BRAINBAR_UI
import SwiftUI
#endif

@MainActor
enum BrainBarSettingsActions {
    private static var openHandler: (() -> Void)?

    static func installOpenHandler(_ handler: @escaping () -> Void) {
        openHandler = handler
    }

#if BRAINBAR_UI
    static func openSettingsWindow(databasePath _: String?) {
        NSApp.activate(ignoringOtherApps: true)
        openHandler?()
    }
#else
    // BrainBarDaemon target is headless and never opens Settings (the file is shared
    // via symlink, but BrainBarSettingsView + its deps live only in the BrainBar target).
    // Keep the original no-op so the daemon target compiles.
    static func openSettingsWindow(databasePath _: String?) {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
    }
#endif
}
