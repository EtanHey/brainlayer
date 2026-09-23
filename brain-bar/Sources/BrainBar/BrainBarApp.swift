import AppKit
import BrainBarLifecycle
import SwiftUI

enum BrainBarAppMenuCommands {
    static let settingsSceneTitle = "Settings…"
    static let settingsSceneEntryCount = 0
    static let manualCommandTitles = [settingsSceneTitle]

    static func isSettingsTitle(_ title: String) -> Bool {
        title.replacingOccurrences(of: "...", with: "…") == settingsSceneTitle
    }

    static var settingsEntryCountForTesting: Int {
        settingsSceneEntryCount + manualCommandTitles.filter(isSettingsTitle).count
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let runtime = BrainBarRuntime()

    private var statusPopoverController: BrainBarStatusPopoverController?
    private var collector: StatsCollector?
    private var dashboardPanel: BrainBarDashboardPanelController?
    private var toggleHotkey: HotkeyManager?
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
            self?.configureToggleHotkey()
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
        configureToggleHotkey()
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
        toggleHotkey?.stop()
        collector?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        ingestBrainBarURLs(urls)
    }

    private func configureRuntimeCallbacks() {
        runtime.onToggleRequested = { [weak self] in
            self?.toggleWindowSurface(nil)
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

    // MARK: - Hotkey

    private func configureToggleHotkey() {
        toggleHotkey?.stop()
        toggleHotkey = nil

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
        toggleHotkey = started ? hotkey : nil
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
        dashboardPanel != nil && statusPopoverController != nil
    }

    private func handleBrainBarURL(_ url: URL) {
        guard let action = BrainBarURLAction.parse(url: url) else {
            NSLog("[BrainBar] Unhandled URL %@", url.absoluteString)
            return
        }

        switch action {
        case .toggle:
            runtime.handleToggleRequest()
        case .dashboard, .settings:
            dashboardPanel?.showURLDestination(action)
        }
    }
}

struct BrainBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button(BrainBarAppMenuCommands.settingsSceneTitle) {
                    BrainBarSettingsActions.openSettingsWindow(databasePath: appDelegate.runtime.databasePath)
                }
                .keyboardShortcut(",", modifiers: [.command])
            }
        }
    }
}
