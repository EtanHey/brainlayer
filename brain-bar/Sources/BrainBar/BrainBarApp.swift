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
    private var injectionStore: InjectionStore?
    private var quickCapturePanel: QuickCapturePanelController?
    private var searchPanel: SearchPanelController?
    private var quickCaptureHotkey: HotkeyManager?
    private var cancellables: Set<AnyCancellable> = []
    private var sharedDatabase: BrainDatabase?
    private let hotkeyRouteStatus = HotkeyRouteStatus()
    private var pendingBrainBarURLs: [URL] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Register Apple Events handler for brainbar:// URLs.
        // This is more reliable than application(_:open:) under the SwiftUI App lifecycle.
        NSAppleEventManager.shared().setEventHandler(
            self,
            andSelector: #selector(handleGetURLEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kInternetEventClass),
            andEventID: AEEventID(kAEGetURL)
        )

        // Single-instance enforcement
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

        // STEP 1: Show status item IMMEDIATELY — no DB access, no blocking
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "BrainBar")
            button.target = self
            button.action = #selector(togglePopover(_:))
            button.toolTip = "BrainBar — loading..."
        }
        self.statusItem = item
        NSLog("[BrainBar] Status item visible — loading backend async")

        // STEP 2: Load everything else on a background queue
        hotkeyRouteStatus.onFallbackChange = { [weak self] in
            self?.configureQuickCaptureHotkey()
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let dbPath = BrainBarServer.defaultDBPath()
            NSLog("[BrainBar] Opening database at %@", dbPath)
            let sharedDatabase = BrainDatabase(path: dbPath)

            let srv = BrainBarServer(database: sharedDatabase)
            srv.onDatabaseReady = { [weak self] database in
                Task { @MainActor in
                    self?.configureQuickCapture(database: database)
                }
            }

            let collector = StatsCollector(
                dbPath: dbPath,
                daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
            )

            let injStore = try? InjectionStore(databasePath: dbPath)
            NSLog("[BrainBar] Backend loaded — injectionStore=%@", injStore != nil ? "OK" : "nil")

            // STEP 3: Wire up UI on main thread
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.sharedDatabase = sharedDatabase
                self.server = srv
                srv.start()
                self.collector = collector
                self.injectionStore = injStore

                // Upgrade status item with live dashboard
                self.upgradeStatusItem(with: collector)
                collector.start()
                self.configureQuickCaptureHotkey()
                NSLog("[BrainBar] Fully initialized — dashboard live")
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        quickCaptureHotkey?.stop()
        collector?.stop()
        injectionStore?.stop()
        server?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    @objc private func handleGetURLEvent(_ event: NSAppleEventDescriptor, withReplyEvent reply: NSAppleEventDescriptor) {
        guard let urlString = event.paramDescriptor(forKeyword: AEKeyword(keyDirectObject))?.stringValue else {
            NSLog("[BrainBar] URL event missing direct object: %@", event.description)
            return
        }
        guard let url = URL(string: urlString) else {
            NSLog("[BrainBar] Malformed URL in event: %@", urlString)
            return
        }
        ingestBrainBarURLs([url])
    }

    func showSearchPanel() {
        searchPanel?.show()
    }

    @objc
    private func togglePopover(_ sender: Any?) {
        guard let button = statusItem?.button else { return }

        // If DB hasn't loaded yet, show a loading popover
        if popover == nil {
            let loadingPopover = NSPopover()
            loadingPopover.behavior = .transient
            loadingPopover.contentSize = NSSize(width: 300, height: 80)
            let vc = NSViewController()
            let label = NSTextField(labelWithString: "BrainBar loading database...")
            label.font = .systemFont(ofSize: 14)
            label.alignment = .center
            label.frame = NSRect(x: 20, y: 20, width: 260, height: 40)
            let view = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 80))
            view.addSubview(label)
            vc.view = view
            loadingPopover.contentViewController = vc
            loadingPopover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            return
        }

        if popover!.isShown {
            popover!.performClose(sender)
        } else {
            popover!.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover!.contentViewController?.view.window?.makeKey()
        }
    }

    private func upgradeStatusItem(with collector: StatsCollector) {
        // AIDEV-NOTE: statusItem already created in applicationDidFinishLaunching.
        // This upgrades it with live sparkline + popover once DB is ready.
        guard let item = statusItem, let button = item.button else { return }

        button.target = self
        button.action = #selector(togglePopover(_:))
        button.imagePosition = .imageOnly
        button.appearsDisabled = false
        button.toolTip = "BrainBar dashboard"

        let popover = NSPopover()
        popover.behavior = .transient
        popover.contentSize = PopoverTab.dashboard.contentSize
        let statusVC = StatusPopoverView(
            collector: collector,
            hotkeyStatus: hotkeyRouteStatus,
            injectionStore: injectionStore,
            database: sharedDatabase
        )
        statusVC.onPreferredSizeChange = { [weak popover] size in
            popover?.contentSize = size
        }
        popover.contentViewController = statusVC

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

        self.popover = popover
        button.toolTip = "BrainBar — connected"
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
