// BrainBarApp.swift — Entry point for BrainBar menu bar daemon.
//
// Menu bar app (no Dock icon) that owns the BrainLayer SQLite database
// and serves MCP tools over /tmp/brainbar.sock.

import AppKit
import SwiftUI

// MARK: - App Delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var server: BrainBarServer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        let srv = BrainBarServer()
        server = srv
        srv.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        server?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}

// MARK: - SwiftUI App entry point

@main
struct BrainBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        MenuBarExtra("BrainBar", systemImage: "brain.head.profile") {
            VStack(alignment: .leading, spacing: 6) {
                Text("BrainBar")
                    .font(.system(.caption, weight: .bold))
                Text("Memory daemon active")
                    .font(.system(.caption2))
                    .foregroundStyle(.secondary)
                Divider()
                Button("Quit BrainBar") {
                    NSApplication.shared.terminate(nil)
                }
                .keyboardShortcut("q")
            }
            .padding(8)
        }

        Settings {
            EmptyView()
        }
    }
}
