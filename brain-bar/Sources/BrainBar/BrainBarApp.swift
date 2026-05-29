import AppKit
import BrainBarLifecycle
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let runtime = BrainBarRuntime()

    private var statusPopoverController: BrainBarStatusPopoverController?
    private var collector: StatsCollector?
    private var dashboardPanel: BrainBarDashboardPanelController?
    private var quickCaptureHotkey: HotkeyManager?
    private var pendingBrainBarURLs: [URL] = []
    private var hotkeyFileWatcher: DispatchSourceFileSystemObject?
    private var uiHeartbeatTimer: DispatchSourceTimer?
    private var daemonWatchdog: BrainBarLifecycleWatchdog?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSAppleEventManager.shared().setEventHandler(
            self,
            andSelector: #selector(handleGetURLEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kInternetEventClass),
            andEventID: AEEventID(kAEGetURL)
        )

        startHotkeyFileWatcher()

        let runningInstances = NSRunningApplication.runningApplications(
            withBundleIdentifier: Bundle.main.bundleIdentifier ?? "com.brainlayer.BrainBar"
        )
        let otherInstances = runningInstances.filter { $0.processIdentifier != ProcessInfo.processInfo.processIdentifier }
        if let existingInstance = otherInstances.first {
            if BrainBarRestartHandoff.consumeIfMatches(existingPID: existingInstance.processIdentifier) {
                NSLog("[BrainBar] Continuing launch for requested restart while PID %d exits.", existingInstance.processIdentifier)
            } else {
                NSLog("[BrainBar] Another instance is already running (PID %d). Exiting.", existingInstance.processIdentifier)
                NSApp.terminate(nil)
                return
            }
        }

        NSApp.setActivationPolicy(.accessory)
        startUIHeartbeat()
        startDaemonWatchdog()
        configureRuntimeCallbacks()

        runtime.hotkeyStatus.onFallbackChange = { [weak self] in
            self?.configureQuickCaptureHotkey()
        }

        let dashboardPanel = BrainBarDashboardPanelController(runtime: runtime)
        self.dashboardPanel = dashboardPanel
        statusPopoverController = BrainBarStatusPopoverController(
            runtime: runtime,
            dashboardPanelController: dashboardPanel
        )

        let dbPath = BrainBarServer.defaultDBPath()
        NSLog("[BrainBar] Starting UI shell; database at %@", dbPath)
        let collector = BrainBarAppSupport.makeUIStatsCollector(
            dbPath: dbPath,
            brainBusEvents: BrainBusClient()
        )
        self.collector = collector
        BrainBarAppSupport.wireRuntime(runtime, dbPath: dbPath, collector: collector)

        flushPendingBrainBarURLs()

        collector.start()
        configureQuickCaptureHotkey()
        NSLog("[BrainBar] Runtime wired — launchMode=%@", String(describing: runtime.launchMode))
    }

    func applicationWillTerminate(_ notification: Notification) {
        statusPopoverController?.stop()
        statusPopoverController = nil
        dashboardPanel?.dismiss()
        dashboardPanel = nil
        uiHeartbeatTimer?.cancel()
        uiHeartbeatTimer = nil
        daemonWatchdog?.stop()
        daemonWatchdog = nil
        hotkeyFileWatcher?.cancel()
        quickCaptureHotkey?.stop()
        collector?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        ingestBrainBarURLs(urls)
    }

    func showSearchPanel() {
        runtime.presentQuickAction(.search)
        showDashboardPanel()
    }

    func showQuickCapturePanel() {
        runtime.presentQuickAction(.capture)
        showDashboardPanel()
    }

    private func configureRuntimeCallbacks() {
        runtime.onToggleRequested = { [weak self] in
            self?.toggleWindowSurface(nil)
        }
        runtime.onSearchRequested = { [weak self] in
            self?.showSearchPanel()
        }
        runtime.onQuickCaptureRequested = { [weak self] in
            self?.showQuickCapturePanel()
        }
    }

    private func startUIHeartbeat() {
        let timer = BrainBarLifecycleWatchdog.makeHeartbeatTimer(
            path: BrainBarLifecycleWatchdog.uiHeartbeatPath,
            interval: 5,
            queue: .main
        )
        uiHeartbeatTimer = timer
    }

    private func startDaemonWatchdog() {
        let watchdog = BrainBarLifecycleWatchdog.makeDaemonWatchdog()
        daemonWatchdog = watchdog
        watchdog.start()
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

    // MARK: - Hotkey File Watcher

    private static let toggleFlagPath = "/tmp/.brainbar-toggle"
    private static let searchFlagPath = "/tmp/.brainbar-search"

    private func startHotkeyFileWatcher() {
        let fd = Darwin.open("/tmp", O_EVTONLY)
        guard fd >= 0 else { return }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: .write,
            queue: .main
        )
        source.setEventHandler { [weak self] in
            self?.checkHotkeyFlags()
        }
        source.setCancelHandler { Darwin.close(fd) }
        source.resume()
        hotkeyFileWatcher = source
    }

    private func checkHotkeyFlags() {
        if FileManager.default.fileExists(atPath: Self.toggleFlagPath) {
            try? FileManager.default.removeItem(atPath: Self.toggleFlagPath)
            runtime.handleToggleRequest()
        }
        if FileManager.default.fileExists(atPath: Self.searchFlagPath) {
            try? FileManager.default.removeItem(atPath: Self.searchFlagPath)
            showSearchPanel()
        }
    }

    // MARK: - Menu Bar Popover

    @objc
    private func toggleWindowSurface(_ sender: Any?) {
        if let statusPopoverController {
            statusPopoverController.toggle(sender)
        }
    }

    func showDashboardPanel() {
        if let statusPopoverController {
            statusPopoverController.show(nil)
        }
    }

    // MARK: - Quick Capture / Search

    private func configureQuickCaptureHotkey() {
        quickCaptureHotkey?.stop()
        quickCaptureHotkey = nil

        guard runtime.hotkeyStatus.useCGEventTapFallback else {
            runtime.hotkeyStatus.refreshStatusLine(eventTapActive: false)
            return
        }

        let gesture = GestureStateMachine()
        gesture.onSingleTap = { [weak self] in
            self?.runtime.handleToggleRequest()
        }
        gesture.onDoubleTap = { [weak self] in
            self?.runtime.handleToggleRequest()
        }

        let hotkey = HotkeyManager(gesture: gesture)
        hotkey.configure(keycodes: [118, 129], useModifierMode: false)
        let started = hotkey.start()
        quickCaptureHotkey = started ? hotkey : nil
        runtime.hotkeyStatus.refreshStatusLine(eventTapActive: started)

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

    private func ingestBrainBarURLs(_ urls: [URL]) {
        for url in urls {
            guard BrainBarURLAction.parse(url: url) != nil else { continue }
            if isReadyToHandleBrainBarURL() {
                handleBrainBarURL(url)
            } else {
                pendingBrainBarURLs.append(url)
            }
        }
    }

    private func flushPendingBrainBarURLs() {
        guard isReadyToHandleBrainBarURL(), !pendingBrainBarURLs.isEmpty else { return }
        let batch = pendingBrainBarURLs
        pendingBrainBarURLs.removeAll()
        for url in batch {
            handleBrainBarURL(url)
        }
    }

    private func isReadyToHandleBrainBarURL() -> Bool {
        true
    }

    private func handleBrainBarURL(_ url: URL) {
        guard let action = BrainBarURLAction.parse(url: url) else {
            NSLog("[BrainBar] Unhandled URL %@", url.absoluteString)
            return
        }

        switch action {
        case .toggle:
            runtime.handleToggleRequest()
        case .search:
            showSearchPanel()
        }
    }
}

@main
struct BrainBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Toggle BrainBar") {
                    appDelegate.runtime.handleToggleRequest()
                }

                Button("Search BrainLayer") {
                    appDelegate.showSearchPanel()
                }
                .keyboardShortcut("k", modifiers: [.command])

                Button("Capture Note") {
                    appDelegate.showQuickCapturePanel()
                }
            }
        }
    }
}
