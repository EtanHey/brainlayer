// BrainBarApp.swift — Entry point for BrainBar menu bar daemon.
//
// Menu bar app (no Dock icon) that owns the BrainLayer SQLite database
// and serves MCP tools over /tmp/brainbar.sock.

import AppKit
import Combine
import SwiftUI

enum BrainBarAppSupport {
    static func hotkeyPermissionFailureMessage(permissions: HotkeyPermissionStatus) -> String {
        "BrainBar could not start the fallback hotkey listener. Enable \(permissions.missingPermissionsMessage) in System Settings. The CGEventTap fallback requires both Input Monitoring and Accessibility."
    }
}

// MARK: - App Delegate

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var server: BrainBarServer?
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private var collector: StatsCollector?
    private var quickCapturePanel: QuickCapturePanelController?
    private var searchPanel: SearchPanelController?
    private var quickCaptureHotkey: HotkeyManager?
    private var cancellables: Set<AnyCancellable> = []
    private var sharedDatabase: BrainDatabase?
    private let hotkeyRouteStatus = HotkeyRouteStatus()
    private var pendingBrainBarURLs: [URL] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Single-instance enforcement: exit if another BrainBar is already running
        let runningInstances = NSRunningApplication.runningApplications(
            withBundleIdentifier: Bundle.main.bundleIdentifier ?? "com.brainlayer.BrainBar"
        )
        let otherInstances = runningInstances.filter { $0.processIdentifier != ProcessInfo.processInfo.processIdentifier }
        if let existingInstance = otherInstances.first {
            NSLog("[BrainBar] Another instance is already running (PID %d). Exiting.", existingInstance.processIdentifier)
            NSApp.terminate(nil)
            return
        }

        NSApp.setActivationPolicy(.accessory)

        let sharedDatabase = BrainDatabase(path: BrainBarServer.defaultDBPath())
        self.sharedDatabase = sharedDatabase

        let srv = BrainBarServer(database: sharedDatabase)
        srv.onDatabaseReady = { [weak self] database in
            Task { @MainActor in
                self?.configureQuickCapture(database: database)
            }
        }
        server = srv
        srv.start()

        let collector = StatsCollector(
            dbPath: BrainBarServer.defaultDBPath(),
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        self.collector = collector
        hotkeyRouteStatus.onFallbackChange = { [weak self] in
            self?.configureQuickCaptureHotkey()
        }
        configureStatusItem(with: collector)
        collector.start()
        configureQuickCaptureHotkey()
    }

    func applicationWillTerminate(_ notification: Notification) {
        quickCaptureHotkey?.stop()
        collector?.stop()
        server?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        ingestBrainBarURLs(urls)
    }

    func showSearchPanel() {
        searchPanel?.show()
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
        popover.contentSize = NSSize(width: 360, height: 320)
        popover.contentViewController = StatusPopoverView(collector: collector, hotkeyStatus: hotkeyRouteStatus)

        Publishers.CombineLatest(collector.$stats, collector.$state)
            .receive(on: RunLoop.main)
            .sink { [weak self] stats, state in
                self?.statusItem?.button?.image = SparklineRenderer.render(
                    state: state,
                    values: stats.recentActivityBuckets
                )
                self?.statusItem?.button?.contentTintColor = state.color
            }
            .store(in: &cancellables)

        self.statusItem = item
        self.popover = popover
    }

    private func configureQuickCaptureHotkey() {
        quickCaptureHotkey?.stop()
        quickCaptureHotkey = nil

        guard hotkeyRouteStatus.useCGEventTapFallback else {
            hotkeyRouteStatus.refreshStatusLine(eventTapActive: false)
            return
        }

        let gesture = GestureStateMachine()
        gesture.onSingleTap = { [weak self] in
            self?.quickCapturePanel?.toggle()
        }
        gesture.onDoubleTap = { [weak self] in
            self?.searchPanel?.show()
        }

        let hotkey = HotkeyManager(gesture: gesture)
        hotkey.configure(keycodes: [118, 129], useModifierMode: false)
        let started = hotkey.start()
        quickCaptureHotkey = started ? hotkey : nil
        hotkeyRouteStatus.refreshStatusLine(eventTapActive: started)
        if !started {
            let permissions = HotkeyManager.permissionStatus()
            let message = BrainBarAppSupport.hotkeyPermissionFailureMessage(permissions: permissions)
            NSLog("[BrainBar.Hotkey] %@", message)

            let alert = NSAlert()
            alert.alertStyle = .warning
            alert.messageText = "BrainBar hotkey permission missing"
            alert.informativeText = message
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }

    private func configureQuickCapture(database: BrainDatabase) {
        guard quickCapturePanel == nil else { return }
        quickCapturePanel = QuickCapturePanelController(db: database)
        if searchPanel == nil {
            searchPanel = SearchPanelController(db: database)
        }
        flushPendingBrainBarURLs()
    }

    private func ingestBrainBarURLs(_ urls: [URL]) {
        for url in urls {
            guard BrainBarURLAction.parse(url: url) != nil else { continue }
            if quickCapturePanel != nil {
                handleBrainBarURL(url)
            } else {
                pendingBrainBarURLs.append(url)
            }
        }
    }

    private func flushPendingBrainBarURLs() {
        guard quickCapturePanel != nil, !pendingBrainBarURLs.isEmpty else { return }
        let batch = pendingBrainBarURLs
        pendingBrainBarURLs.removeAll()
        for url in batch {
            handleBrainBarURL(url)
        }
    }

    private func handleBrainBarURL(_ url: URL) {
        guard let action = BrainBarURLAction.parse(url: url) else {
            NSLog("[BrainBar] Unhandled URL %@", url.absoluteString)
            return
        }
        switch action {
        case .toggle:
            quickCapturePanel?.toggle()
        case .search:
            searchPanel?.show()
        }
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
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Search BrainLayer") {
                    appDelegate.showSearchPanel()
                }
                .keyboardShortcut("k", modifiers: [.command])
            }
        }
    }
}
