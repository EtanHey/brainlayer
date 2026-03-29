// BrainBarApp.swift — Entry point for BrainBar menu bar daemon.
//
// Menu bar app (no Dock icon) that owns the BrainLayer SQLite database
// and serves MCP tools over /tmp/brainbar.sock.

import AppKit
import Combine
import SwiftUI

// MARK: - App Delegate

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var server: BrainBarServer?
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private var collector: StatsCollector?
    private var cancellables: Set<AnyCancellable> = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        let srv = BrainBarServer()
        server = srv
        srv.start()

        let collector = StatsCollector(
            dbPath: BrainBarServer.defaultDBPath(),
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        self.collector = collector
        configureStatusItem(with: collector)
        collector.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        collector?.stop()
        server?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    @objc
    private func togglePopover(_ sender: Any?) {
        guard let button = statusItem?.button else { return }
        guard let popover else { return }

        if popover.isShown {
            popover.performClose(sender)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
        }
    }

    private func configureStatusItem(with collector: StatsCollector) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        guard let button = item.button else { return }

        button.target = self
        button.action = #selector(togglePopover(_:))
        button.imagePosition = .imageOnly
        button.appearsDisabled = false
        button.toolTip = "BrainBar dashboard"

        let popover = NSPopover()
        popover.behavior = .transient
        popover.contentSize = NSSize(width: 360, height: 270)
        popover.contentViewController = NSHostingController(rootView: StatusPopoverView(collector: collector))

        Publishers.CombineLatest(collector.$stats, collector.$state)
            .receive(on: RunLoop.main)
            .sink { [weak self] stats, state in
                Task { @MainActor [weak self] in
                    self?.statusItem?.button?.image = SparklineRenderer.render(
                        state: state,
                        values: stats.recentActivityBuckets
                    )
                    self?.statusItem?.button?.contentTintColor = state.color
                }
            }
            .store(in: &cancellables)

        self.statusItem = item
        self.popover = popover
    }
}

// MARK: - SwiftUI App entry point

@main
struct BrainBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}
