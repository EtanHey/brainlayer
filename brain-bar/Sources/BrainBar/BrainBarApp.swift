import AppKit
import ApplicationServices
import Combine
import SwiftUI

enum BrainBarAppSupport {
    static func hotkeyPermissionFailureMessage(permissions: HotkeyPermissionStatus) -> String {
        "BrainBar could not start the fallback hotkey listener. Enable \(permissions.missingPermissionsMessage) in System Settings. The CGEventTap fallback requires both Input Monitoring and Accessibility."
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let runtime = BrainBarRuntime()
    private static let menuBarWindowAutosaveKey = "NSWindow Frame BrainBarMenuBarExtraWindow"

    private var server: BrainBarServer?
    private var legacyStatusItem: NSStatusItem?
    private var legacyPopover: NSPopover?
    private var collector: StatsCollector?
    private var injectionStore: InjectionStore?
    private var quickCapturePanel: QuickCapturePanelController?
    private var searchPanel: SearchPanelController?
    private var dashboardPanel: BrainBarDashboardPanelController?
    private var quickCaptureHotkey: HotkeyManager?
    private weak var menuBarExtraWindow: NSWindow?
    private weak var discoveredMenuBarWindow: NSWindow?
    private var cancellables: Set<AnyCancellable> = []
    private var sharedDatabase: BrainDatabase?
    private var pendingBrainBarURLs: [URL] = []
    private var hotkeyFileWatcher: DispatchSourceFileSystemObject?
    private var menuBarWindowObservers: [NSObjectProtocol] = []
    private var menuBarWindowSyncTask: Task<Void, Never>?
    private var wasVisibleAccessibilityWindow = false

    private var launchMode: BrainBarLaunchMode {
        runtime.launchMode
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSAppleEventManager.shared().setEventHandler(
            self,
            andSelector: #selector(handleGetURLEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kInternetEventClass),
            andEventID: AEEventID(kAEGetURL)
        )

        startHotkeyFileWatcher()

        if launchMode == .menuBarWindow {
            UserDefaults.standard.removeObject(forKey: Self.menuBarWindowAutosaveKey)
            dashboardPanel = BrainBarDashboardPanelController(runtime: runtime)
        }

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
        configureRuntimeCallbacks()

        runtime.hotkeyStatus.onFallbackChange = { [weak self] in
            self?.configureQuickCaptureHotkey()
        }

        if launchMode == .legacyStatusItem {
            createLegacyStatusItem()
        }

        let dbPath = BrainBarServer.defaultDBPath()
        NSLog("[BrainBar] Starting server before database readiness at %@", dbPath)
        let server = BrainBarServer(dbPath: dbPath)
        server.onStartRejected = { reason in
            NSLog("[BrainBar] Startup rejected: %@", reason)
            Task { @MainActor in
                NSApp.terminate(nil)
            }
        }
        server.onDatabaseReady = { [weak self] database in
            Task { @MainActor in
                guard let self else { return }
                self.sharedDatabase = database
                self.configureQuickCapture(database: database)
                self.runtime.install(
                    collector: self.collector ?? StatsCollector(
                        dbPath: dbPath,
                        daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
                    ),
                    injectionStore: self.injectionStore,
                    database: database
                )
                self.flushPendingBrainBarURLs()
            }
        }

        let collector = StatsCollector(
            dbPath: dbPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        let injectionStore = try? InjectionStore(databasePath: dbPath)

        self.server = server
        self.collector = collector
        self.injectionStore = injectionStore

        server.start()

        if launchMode == .legacyStatusItem {
            installLegacyMenuBarSurface(with: collector)
        }

        collector.start()
        configureQuickCaptureHotkey()
        NSLog("[BrainBar] Socket ready; database will self-heal — launchMode=%@", String(describing: launchMode))
    }

    func applicationWillTerminate(_ notification: Notification) {
        menuBarWindowObservers.forEach(NotificationCenter.default.removeObserver)
        menuBarWindowObservers.removeAll()
        menuBarWindowSyncTask?.cancel()
        menuBarWindowSyncTask = nil
        hotkeyFileWatcher?.cancel()
        quickCaptureHotkey?.stop()
        collector?.stop()
        injectionStore?.stop()
        server?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        ingestBrainBarURLs(urls)
    }

    func showSearchPanel() {
        guard launchMode == .menuBarWindow else {
            searchPanel?.show()
            return
        }

        runtime.presentQuickAction(.search)
        showMenuBarWindow(nil)
    }

    func showQuickCapturePanel() {
        guard launchMode == .menuBarWindow else {
            quickCapturePanel?.toggle()
            return
        }

        runtime.presentQuickAction(.capture)
        showMenuBarWindow(nil)
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

    // MARK: - Legacy Shell

    private func createLegacyStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "BrainBar")
            button.target = self
            button.action = #selector(toggleLegacySurface(_:))
            button.toolTip = "BrainBar legacy shell — loading..."
        }
        legacyStatusItem = item
    }

    @objc
    private func toggleWindowSurface(_ sender: Any?) {
        if launchMode == .legacyStatusItem {
            toggleLegacySurface(sender)
            return
        }

        if let dashboardPanel {
            dashboardPanel.toggle()
            return
        }

        if let window = menuBarExtraWindow ?? discoverMenuBarWindow() {
            runtime.windowCoordinator.attach(window: window)
            ensureDefaultMenuBarWindowFrame(window)
            if window.isVisible {
                BrainBarWindowFrameStore().persist(frame: window.frame)
                window.orderOut(sender)
            } else {
                showMenuBarWindow(sender)
            }
            return
        }

        if visibleMenuBarExtraAccessibilityWindow() != nil {
            persistVisibleMenuBarWindowFrame()
            _ = pressMenuBarExtraItem()
            return
        }

        if pressMenuBarExtraItem() {
            scheduleVisibleMenuBarWindowRestore()
            return
        }

        guard let window = discoverMenuBarWindow() else {
            NSLog("[BrainBar] Could not discover MenuBarExtra window for toggle request")
            return
        }

        runtime.windowCoordinator.attach(window: window)
        ensureDefaultMenuBarWindowFrame(window)
        if window.isVisible {
            window.orderOut(sender)
        } else {
            showMenuBarWindow(sender)
        }
    }

    func showDashboardPanel() {
        guard launchMode == .menuBarWindow else {
            toggleLegacySurface(nil)
            return
        }

        dashboardPanel?.show()
    }

    private func showMenuBarWindow(_ sender: Any?) {
        if launchMode == .menuBarWindow, let dashboardPanel {
            dashboardPanel.show()
            return
        }

        if let window = menuBarExtraWindow ?? discoverMenuBarWindow() {
            runtime.windowCoordinator.attach(window: window)
            ensureDefaultMenuBarWindowFrame(window)
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(sender)
            window.orderFrontRegardless()
            return
        }

        if visibleMenuBarExtraAccessibilityWindow() != nil {
            _ = restoreVisibleMenuBarWindowFrame()
            return
        }

        if pressMenuBarExtraItem() {
            scheduleVisibleMenuBarWindowRestore()
            return
        }

        guard let window = discoverMenuBarWindow() else {
            NSLog("[BrainBar] Could not discover MenuBarExtra window for show request")
            return
        }

        runtime.windowCoordinator.attach(window: window)
        ensureDefaultMenuBarWindowFrame(window)
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(sender)
        window.orderFrontRegardless()
    }

    private func discoverMenuBarWindow() -> NSWindow? {
        if let menuBarExtraWindow {
            return menuBarExtraWindow
        }

        if let discoveredMenuBarWindow {
            return discoveredMenuBarWindow
        }

        let window = NSApp.windows.first { candidate in
            let isExcluded = searchPanel?.panelForTesting === candidate
            return !isExcluded &&
                candidate.title == "BrainBar" &&
                candidate.frame.width >= 400 &&
                candidate.frame.height >= 300
        }
        discoveredMenuBarWindow = window
        return window
    }

    private func orderedScreensByMouseLocation() -> [NSScreen] {
        let screens = NSScreen.screens
        guard let preferredIndex = screens.firstIndex(where: { $0.frame.contains(NSEvent.mouseLocation) }) else {
            return screens
        }
        var ordered = screens
        let preferredScreen = ordered.remove(at: preferredIndex)
        ordered.insert(preferredScreen, at: 0)
        return ordered
    }

    private func startMenuBarWindowSync() {
        menuBarWindowSyncTask?.cancel()
        menuBarWindowSyncTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard !Task.isCancelled else { break }
                self.syncVisibleMenuBarWindow()
            }
        }
    }

    private func syncVisibleMenuBarWindow() {
        guard let window = visibleMenuBarExtraAccessibilityWindow() else {
            wasVisibleAccessibilityWindow = false
            return
        }

        if !wasVisibleAccessibilityWindow {
            _ = restoreVisibleMenuBarWindowFrame(window: window)
        }

        persistVisibleMenuBarWindowFrame(window: window)
        wasVisibleAccessibilityWindow = true
    }

    private func pressMenuBarExtraItem() -> Bool {
        guard let item = menuBarExtraItemAccessibilityElement() else { return false }
        return AXUIElementPerformAction(item, kAXPressAction as CFString) == .success
    }

    private func menuBarExtraItemAccessibilityElement() -> AXUIElement? {
        let appElement = AXUIElementCreateApplication(ProcessInfo.processInfo.processIdentifier)
        var extrasBarRef: CFTypeRef?
        let extrasBarError = AXUIElementCopyAttributeValue(
            appElement,
            kAXExtrasMenuBarAttribute as CFString,
            &extrasBarRef
        )
        guard extrasBarError == .success, let extrasBar = extrasBarRef else {
            return nil
        }

        var childrenRef: CFTypeRef?
        let childrenError = AXUIElementCopyAttributeValue(
            extrasBar as! AXUIElement,
            kAXChildrenAttribute as CFString,
            &childrenRef
        )
        guard childrenError == .success, let children = childrenRef as? [AXUIElement] else {
            return nil
        }

        let mouseLocation = NSEvent.mouseLocation
        var titledMatches: [(element: AXUIElement, frame: CGRect)] = []
        var fallbackMatches: [(element: AXUIElement, frame: CGRect)] = []

        for item in children {
            var titleRef: CFTypeRef?
            _ = AXUIElementCopyAttributeValue(item, kAXTitleAttribute as CFString, &titleRef)
            let title = (titleRef as? String)?.lowercased() ?? ""
            guard let frame = accessibilityElementFrame(item) else { continue }
            if title == "brain" || title == "brainbar" {
                titledMatches.append((item, frame))
            } else {
                fallbackMatches.append((item, frame))
            }
        }

        if let preferredFrame = BrainBarWindowPlacement.preferredMenuBarItemFrame(
            candidates: titledMatches.map(\.frame),
            mouseLocation: mouseLocation
        ) {
            return titledMatches.first(where: { $0.frame == preferredFrame })?.element
        }

        if let preferredFrame = BrainBarWindowPlacement.preferredMenuBarItemFrame(
            candidates: fallbackMatches.map(\.frame),
            mouseLocation: mouseLocation
        ) {
            return fallbackMatches.first(where: { $0.frame == preferredFrame })?.element
        }

        return children.first
    }

    private func visibleMenuBarExtraAccessibilityWindow() -> AXUIElement? {
        let appElement = AXUIElementCreateApplication(ProcessInfo.processInfo.processIdentifier)
        var windowsRef: CFTypeRef?
        let windowsError = AXUIElementCopyAttributeValue(
            appElement,
            kAXWindowsAttribute as CFString,
            &windowsRef
        )
        guard windowsError == .success, let windows = windowsRef as? [AXUIElement] else {
            return nil
        }

        var dialogCandidate: AXUIElement?
        for window in windows {
            var titleRef: CFTypeRef?
            var subroleRef: CFTypeRef?
            _ = AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &titleRef)
            _ = AXUIElementCopyAttributeValue(window, kAXSubroleAttribute as CFString, &subroleRef)
            let title = titleRef as? String ?? ""
            let subrole = subroleRef as? String ?? ""
            let frame = accessibilityElementFrame(window)
            let isMainWindowSize = (frame?.width ?? 0) >= 760 && (frame?.height ?? 0) >= 560

            if title == "BrainBar" && isMainWindowSize {
                return window
            }

            if subrole == (kAXSystemDialogSubrole as String), isMainWindowSize {
                dialogCandidate = dialogCandidate ?? window
            }
        }

        return dialogCandidate
    }

    private func accessibilityElementFrame(_ element: AXUIElement) -> CGRect? {
        var positionRef: CFTypeRef?
        var sizeRef: CFTypeRef?
        let positionError = AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &positionRef)
        let sizeError = AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeRef)
        guard positionError == .success,
              sizeError == .success,
              let positionValue = positionRef,
              let sizeValue = sizeRef
        else {
            return nil
        }

        var position = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(positionValue as! AXValue, .cgPoint, &position),
              AXValueGetValue(sizeValue as! AXValue, .cgSize, &size)
        else {
            return nil
        }

        return CGRect(origin: position, size: size)
    }

    private func applyAccessibilityFrame(_ frame: CGRect, to window: AXUIElement) -> Bool {
        var size = frame.size
        guard let sizeValue = AXValueCreate(.cgSize, &size) else { return false }
        let sizeError = AXUIElementSetAttributeValue(window, kAXSizeAttribute as CFString, sizeValue)

        var position = frame.origin
        guard let positionValue = AXValueCreate(.cgPoint, &position) else { return false }
        let positionError = AXUIElementSetAttributeValue(window, kAXPositionAttribute as CFString, positionValue)

        return sizeError == .success && positionError == .success
    }

    private func persistVisibleMenuBarWindowFrame(window: AXUIElement? = nil) {
        let screenFrames = NSScreen.screens.map(\.frame)
        guard let window = window ?? visibleMenuBarExtraAccessibilityWindow(),
              let accessibilityFrame = accessibilityElementFrame(window),
              let appKitFrame = BrainBarWindowPlacement.appKitFrame(
                  fromAccessibility: accessibilityFrame,
                  screenFrames: screenFrames
              )
        else {
            return
        }

        BrainBarWindowFrameStore().persist(frame: appKitFrame)
        discoveredMenuBarWindow?.setFrame(appKitFrame, display: false)
    }

    private func restoreVisibleMenuBarWindowFrame(window: AXUIElement? = nil) -> Bool {
        let screenFrames = NSScreen.screens.map(\.frame)
        let visibleScreenFrames = NSScreen.screens.map(\.visibleFrame)
        let preferredVisibleScreenFrames = orderedScreensByMouseLocation().map(\.visibleFrame)
        guard let window = window ?? visibleMenuBarExtraAccessibilityWindow()
        else {
            return false
        }

        let persistedFrame = BrainBarWindowFrameStore().persistedFrame()
        let targetFrame: CGRect

        if let persistedFrame,
           BrainBarWindowPlacement.isRestorable(frame: persistedFrame, screenFrames: visibleScreenFrames) {
            targetFrame = BrainBarWindowPlacement.clearingMenuBar(
                frame: persistedFrame,
                screenFrames: visibleScreenFrames
            )
        } else if let currentWindowFrame = accessibilityElementFrame(window),
                  let iconElement = menuBarExtraItemAccessibilityElement(),
                  let iconFrame = accessibilityElementFrame(iconElement),
                  let anchoredFrame = BrainBarWindowPlacement.anchoredFrameBelowMenuBarItem(
                      currentAccessibilityFrame: currentWindowFrame,
                      menuBarItemAccessibilityFrame: iconFrame,
                      screenFrames: screenFrames,
                      visibleScreenFrames: visibleScreenFrames,
                      gap: BrainBarWindowPlacement.menuBarIconGap
                  ) {
            targetFrame = anchoredFrame
        } else if let resolvedFrame = BrainBarWindowPlacement.resolvedFrame(
            persistedFrame: nil,
            screenFrames: preferredVisibleScreenFrames
        ) {
            targetFrame = resolvedFrame
        } else {
            return false
        }

        guard let accessibilityFrame = BrainBarWindowPlacement.accessibilityFrame(
            fromAppKit: targetFrame,
            screenFrames: screenFrames
        ) else {
            return false
        }

        let applied = applyAccessibilityFrame(accessibilityFrame, to: window)
        if applied {
            discoveredMenuBarWindow?.setFrame(targetFrame, display: false)
            BrainBarWindowFrameStore().persist(frame: targetFrame)
        }
        return applied
    }

    private func scheduleVisibleMenuBarWindowRestore(attempt: Int = 0) {
        if restoreVisibleMenuBarWindowFrame() {
            return
        }

        guard attempt < 12 else {
            NSLog("[BrainBar] Timed out waiting for MenuBarExtra accessibility window")
            return
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.scheduleVisibleMenuBarWindowRestore(attempt: attempt + 1)
        }
    }

    private func ensureDefaultMenuBarWindowFrame(_ window: NSWindow) {
        let screenFrames = NSScreen.screens.map(\.visibleFrame)
        let fullScreenFrames = NSScreen.screens.map(\.frame)
        let preferredVisibleScreenFrames = orderedScreensByMouseLocation().map(\.visibleFrame)
        let isUsableSize = window.frame.width >= 760 && window.frame.height >= 560
        let isUsablePosition = BrainBarWindowPlacement.isRestorable(
            frame: window.frame,
            screenFrames: screenFrames
        )

        guard !isUsableSize || !isUsablePosition else { return }

        let persistedFrame = BrainBarWindowFrameStore().persistedFrame()
        if persistedFrame == nil,
           let iconElement = menuBarExtraItemAccessibilityElement(),
           let iconFrame = accessibilityElementFrame(iconElement),
           let anchoredFrame = BrainBarWindowPlacement.anchoredFrameBelowMenuBarItem(
               currentFrame: CGRect(origin: window.frame.origin, size: BrainBarWindowPlacement.defaultSize),
               menuBarItemAccessibilityFrame: iconFrame,
               screenFrames: fullScreenFrames,
               visibleScreenFrames: screenFrames,
               gap: BrainBarWindowPlacement.menuBarIconGap
           ) {
            window.setFrame(anchoredFrame, display: false)
            BrainBarWindowFrameStore().persist(frame: anchoredFrame)
            return
        }

        guard let resolvedFrame = BrainBarWindowPlacement.resolvedFrame(
            persistedFrame: persistedFrame,
            screenFrames: preferredVisibleScreenFrames
        ) else {
            return
        }

        window.setFrame(resolvedFrame, display: false)
    }

    private func installMenuBarWindowObservers() {
        let center = NotificationCenter.default
        menuBarWindowObservers = [
            center.addObserver(
                forName: NSWindow.didBecomeKeyNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                guard let window = notification.object as? NSWindow else { return }
                Task { @MainActor [weak self] in
                    self?.handleMenuBarWindowBecameKey(window)
                }
            },
            center.addObserver(
                forName: NSWindow.didMoveNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                guard let window = notification.object as? NSWindow else { return }
                Task { @MainActor [weak self] in
                    self?.persistObservedMenuBarWindowFrame(window)
                }
            },
            center.addObserver(
                forName: NSWindow.didEndLiveResizeNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                guard let window = notification.object as? NSWindow else { return }
                Task { @MainActor [weak self] in
                    self?.persistObservedMenuBarWindowFrame(window)
                }
            },
        ]
    }

    private func handleMenuBarWindowBecameKey(_ window: NSWindow) {
        guard isMenuBarExtraWindow(window) else { return }

        menuBarExtraWindow = window
        discoveredMenuBarWindow = window
        runtime.windowCoordinator.attach(window: window)
        ensureDefaultMenuBarWindowFrame(window)
    }

    private func persistObservedMenuBarWindowFrame(_ window: NSWindow) {
        guard isMenuBarExtraWindow(window) else { return }

        menuBarExtraWindow = window
        discoveredMenuBarWindow = window
        BrainBarWindowFrameStore().persist(frame: window.frame)
    }

    private func isMenuBarExtraWindow(_ window: NSWindow) -> Bool {
        guard searchPanel?.panelForTesting !== window else { return false }
        return window.title == "BrainBar" &&
            window.frame.width >= 760 &&
            window.frame.height >= 560
    }

    @objc
    private func toggleLegacySurface(_ sender: Any?) {
        guard launchMode == .legacyStatusItem, let button = legacyStatusItem?.button else { return }

        guard let popover = legacyPopover else {
            showLegacyLoadingPopover(relativeTo: button)
            return
        }

        if popover.isShown {
            popover.performClose(sender)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
        }
    }

    private func showLegacyLoadingPopover(relativeTo button: NSStatusBarButton) {
        let loadingPopover = NSPopover()
        loadingPopover.behavior = .transient
        loadingPopover.contentSize = NSSize(width: 320, height: 96)

        let hosting = NSHostingController(
            rootView: BrainBarLoadingView(
                title: "BrainBar",
                subtitle: "Opening database and warming the dashboard..."
            )
            .frame(width: 320, height: 96)
        )

        loadingPopover.contentViewController = hosting
        loadingPopover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
    }

    private func installLegacyMenuBarSurface(with collector: StatsCollector) {
        guard launchMode == .legacyStatusItem, let item = legacyStatusItem, let button = item.button else {
            return
        }

        let popover = NSPopover()
        popover.behavior = .transient
        popover.contentSize = NSSize(width: 760, height: 560)
        popover.contentViewController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime)
                .frame(width: 760, height: 560)
        )
        legacyPopover = popover

        button.target = self
        button.action = #selector(toggleLegacySurface(_:))
        button.toolTip = "BrainBar legacy shell"

        Publishers.CombineLatest(collector.$stats, collector.$state)
            .receive(on: RunLoop.main)
            .sink { [weak self] stats, state in
                let livePresentation = BrainBarLivePresentation.derive(stats: stats)
                self?.legacyStatusItem?.button?.image = SparklineRenderer.render(
                    state: state,
                    values: stats.recentEnrichmentBuckets,
                    accentColor: livePresentation.accentColor
                )
            }
            .store(in: &cancellables)
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

    private func configureQuickCapture(database: BrainDatabase) {
        guard launchMode == .legacyStatusItem else { return }
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

    /// URL actions are dispatched once the backing surface is ready. In
    /// menuBarWindow mode the command bar is driven by `runtime.database` +
    /// the MenuBarExtra window, so readiness means the database has been
    /// installed into the runtime. In legacy mode the floating panel still
    /// drives routing.
    private func isReadyToHandleBrainBarURL() -> Bool {
        switch launchMode {
        case .menuBarWindow:
            return runtime.database != nil
        case .legacyStatusItem:
            return quickCapturePanel != nil
        }
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
    private let launchMode = BrainBarLaunchMode.resolve()

    var body: some Scene {
        MenuBarExtra(isInserted: .constant(launchMode == .menuBarWindow)) {
            Button("Open Dashboard") {
                appDelegate.showDashboardPanel()
            }

            Button("Search BrainLayer") {
                appDelegate.showSearchPanel()
            }

            Button("Capture Note") {
                appDelegate.showQuickCapturePanel()
            }
        } label: {
            BrainBarMenuBarLabel(runtime: appDelegate.runtime)
        }
        .menuBarExtraStyle(.menu)

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

private struct BrainBarMenuBarLabel: View {
    @ObservedObject var runtime: BrainBarRuntime

    var body: some View {
        if let collector = runtime.collector {
            let livePresentation = BrainBarLivePresentation.derive(stats: collector.stats)
            HStack(spacing: 6) {
                Image(systemName: "brain")
                Image(
                    nsImage: SparklineRenderer.render(
                        state: collector.state,
                        values: collector.stats.recentEnrichmentBuckets,
                        size: NSSize(width: 22, height: 12),
                        accentColor: livePresentation.accentColor
                    )
                )
                .interpolation(.high)
            }
        } else {
            Image(systemName: "brain")
        }
    }
}
