import AppKit
import SwiftUI
import Combine

/// Installed by the UI shell only. Render harnesses keep their own activation policy.
@MainActor
final class BrainBarWindowActivationPolicy {
    private var openWindows: Set<ObjectIdentifier> = []
    private let setPolicy: (NSApplication.ActivationPolicy) -> Void

    init(setPolicy: @escaping (NSApplication.ActivationPolicy) -> Void = { policy in
        NSApplication.shared.setActivationPolicy(policy)
    }) {
        self.setPolicy = setPolicy
    }

    func windowOpened(_ window: NSWindow) {
        let wasEmpty = openWindows.isEmpty
        openWindows.insert(ObjectIdentifier(window))
        if wasEmpty { setPolicy(.regular) }
    }

    /// Called after close has ordered the window out, never on focus loss or minimization.
    func windowClosed(_ window: NSWindow) {
        guard openWindows.remove(ObjectIdentifier(window)) != nil else { return }
        if openWindows.isEmpty { setPolicy(.accessory) }
    }
}

enum BrainBarDisclosureAnimation {
    enum Direction {
        case open
        case close
    }

    enum Curve: Equatable {
        case easeInOut
    }

    struct Timing: Equatable {
        let duration: TimeInterval
        let curve: Curve
    }

    struct Layout: Equatable {
        let containerHeight: CGFloat
        let windowHeight: CGFloat
    }

    private static let standardDuration: TimeInterval = 0.25

    static func timing(for direction: Direction, reduceMotion: Bool) -> Timing {
        _ = direction
        return Timing(duration: reduceMotion ? 0 : standardDuration, curve: .easeInOut)
    }

    static func animation(for direction: Direction, reduceMotion: Bool) -> Animation? {
        let timing = timing(for: direction, reduceMotion: reduceMotion)
        guard timing.duration > 0 else { return nil }
        return .easeInOut(duration: timing.duration)
    }

    static func containerHeight(
        progress: CGFloat,
        collapsedContainerHeight: CGFloat,
        expandedContainerHeight: CGFloat
    ) -> CGFloat {
        let clampedProgress = min(max(progress, 0), 1)
        return collapsedContainerHeight
            + ((expandedContainerHeight - collapsedContainerHeight) * clampedProgress)
    }

    static func windowHeight(
        containerHeight: CGFloat,
        chromeAndSurroundingContentHeight: CGFloat
    ) -> CGFloat {
        chromeAndSurroundingContentHeight + containerHeight
    }

    static func layout(
        progress: CGFloat,
        collapsedContainerHeight: CGFloat,
        expandedContainerHeight: CGFloat,
        chromeAndSurroundingContentHeight: CGFloat
    ) -> Layout {
        let containerHeight = containerHeight(
            progress: progress,
            collapsedContainerHeight: collapsedContainerHeight,
            expandedContainerHeight: expandedContainerHeight
        )
        return Layout(
            containerHeight: containerHeight,
            windowHeight: windowHeight(
                containerHeight: containerHeight,
                chromeAndSurroundingContentHeight: chromeAndSurroundingContentHeight
            )
        )
    }
}

@MainActor
final class BrainBarDashboardPanelState: ObservableObject {
    @Published var attentionExpanded = false
    /// "+N more" in the attention list: every row, inside a height-capped scroll.
    @Published var attentionShowsAll = false
    /// Show log's sentence when the job has no log yet (Codex #1062 r1 B1); nil after a log opens.
    @Published var jobAlertLogNote: BrainBarJobAlertLogNote?

    /// The note's sentence, only while `reason` is still the alert it was produced for.
    func jobAlertLogMessage(forReason reason: String) -> String? {
        jobAlertLogNote.flatMap { $0.reason == reason ? $0.message : nil }
    }
    @Published var detailsExpanded = BrainBarOnePageComposition.detailsExpandedByDefault
    @Published var signalCoverageExpanded = false
    @Published var dashboardHeight: CGFloat = 0
    @Published var headerHeight: CGFloat = 0
    @Published var selectedTab: BrainBarTab = .dashboard
    @Published var settingsActivationRevision = 0
#if BRAINBAR_UI
    let settingsNavigation = BrainBarSettingsNavigation()
#endif
#if DEBUG
    var renderedSummaryTileHeights: [String: CGFloat] = [:]
    var renderedCardSizes: [String: CGSize] = [:]
    var renderedReceiptValues: [String: String] = [:]
#endif
    var fittingHeight: CGFloat {
        if selectedTab == .settings { return 720 }
        return max(
            BrainBarDisclosureAnimation.windowHeight(
                containerHeight: dashboardHeight,
                chromeAndSurroundingContentHeight: headerHeight
            ),
            300
        )
    }

    func disclosureAnimationDidComplete() {}

#if BRAINBAR_UI
    /// The sidebar row that is selected: Dashboard, or the settings page on show (#963).
    var sidebarSelection: BrainBarSidebarItem {
        selectedTab == .dashboard ? .dashboard : BrainBarSidebarItem(section: settingsNavigation.selected)
    }

    func select(_ item: BrainBarSidebarItem) {
        if let section = item.settingsSection {
            settingsNavigation.select(section)
            selectedTab = .settings
        } else {
            selectedTab = .dashboard
        }
    }
#endif
}

/// The one BrainBar window (#963, vNext D5): a real, titled window that stays open until it is
/// closed, like VoiceBar's Settings window. The old menu-bar panel floated above everything and
/// dismissed itself on any click away, which broke click-through testing.
final class BrainBarMainWindow: NSWindow {
    /// Cmd-, from inside the window. The window handles its own key equivalents because an
    /// accessory app shows no menu bar to route them.
    var onSettingsShortcut: (() -> Void)?
    var onClose: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func close() {
        super.close()
        onClose?()
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        if event.type == .keyDown,
           event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command {
            switch event.charactersIgnoringModifiers {
            case "w":
                performClose(nil)
                return true
            case ",":
                onSettingsShortcut?()
                return true
            default:
                break
            }
        }
        return super.performKeyEquivalent(with: event)
    }
}

@MainActor
final class BrainBarDashboardPanelController: NSObject, NSWindowDelegate {
    static let defaultSize = NSSize(
        width: BrainBarWindowPlacement.defaultSize.width,
        height: BrainBarWindowPlacement.defaultSize.height
    )
    static let minSize = NSSize(
        width: BrainBarWindowPlacement.minimumSize.width,
        height: BrainBarWindowPlacement.minimumSize.height
    )
    /// Where the window's last position and size are kept between launches.
    static let frameDefaultsKey = "brainbar.window.main-frame"

    let windowForTesting: NSWindow
    let contentViewControllerForTesting: NSViewController
    var isShownForTesting: Bool { window.isVisible }
    var naturalDashboardHeightForTesting: CGFloat { panelState.dashboardHeight }

    private let window: BrainBarMainWindow
    private let panelState = BrainBarDashboardPanelState()
    private let frameStore: BrainBarWindowFrameStore
    private let screenFrames: () -> [NSRect]
    private let activationPolicy: BrainBarWindowActivationPolicy?
    private let driveAuth: BrainBarDriveAuthModel
    private var hasPlacedWindow = false
    private var isClosingWindow = false
#if BRAINBAR_UI
    /// L6: the content shown follows `BrainBarDesign`, read live. Classic keeps main's hosting controller untouched;
    /// the new design swaps in its own and swaps back.
    private let classicContent: NSViewController
    private var newContent: NSViewController?
    private(set) var shownDesign: BrainBarDesign = .classic
    private let runtimeForDesign: BrainBarRuntime
    private let workspaceForDesign: any BrainBarWorkspaceActing
    nonisolated(unsafe) private var designObserver: NSObjectProtocol?
    var designDefaults: UserDefaults = .standard
    private var hasNewDesignStore = false
    private lazy var newDesignToolbar = NDToolbar(store: newDesignStore)
    lazy var newDesignStore: NDStore = {
        hasNewDesignStore = true
        let store = NDStore.shared(for: runtimeForDesign, workspace: workspaceForDesign)
        store.onOpenWindow = { [weak self] page in self?.showPage(page) }
        store.onOpenSettings = { [weak self] in self?.showSettings() }
        return store
    }()
#endif
    /// Only used to place the very first window below the menu-bar icon.
    weak var statusItemButton: NSView?

    init(
        workspace: any BrainBarWorkspaceActing,
        runtime: BrainBarRuntime,
        activationPolicy: BrainBarWindowActivationPolicy? = nil,
        frameStore: BrainBarWindowFrameStore = BrainBarWindowFrameStore(key: frameDefaultsKey),
        screenFrames: @escaping () -> [NSRect] = { NSScreen.screens.map(\.visibleFrame) }
    ) {
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(workspace: workspace, runtime: runtime, managesWindowFrame: false, panelState: panelState)
                .frame(minWidth: Self.minSize.width)
                .frame(maxWidth: .infinity)
        )
        hostingController.sizingOptions = []
        hostingController.view.frame = NSRect(origin: .zero, size: Self.defaultSize)
        hostingController.view.autoresizingMask = [.width, .height]

        contentViewControllerForTesting = hostingController
#if BRAINBAR_UI
        classicContent = hostingController
        runtimeForDesign = runtime
        workspaceForDesign = workspace
#endif
        self.driveAuth = runtime.driveAuth
        self.frameStore = frameStore
        self.screenFrames = screenFrames
        self.activationPolicy = activationPolicy
        window = Self.makeWindow(contentViewController: hostingController)
        windowForTesting = window
        super.init()
        window.delegate = self
        window.onSettingsShortcut = { [weak self] in self?.showSettings() }
        window.onClose = { [weak self] in
            guard let self else { return }
            self.activationPolicy?.windowClosed(self.window)
        }
        BrainBarSettingsActions.installOpenHandler { [weak self] in
            self?.showSettings()
        }
#if BRAINBAR_UI
        applyDesign()
        designObserver = NotificationCenter.default.addObserver(forName: UserDefaults.didChangeNotification, object: nil,
                                                                queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.applyDesign() }
        }
#endif
    }

#if BRAINBAR_UI
    /// Swaps the window's content to the design `BrainBarDesign` names now. No animation: the switch is a setting.
    func applyDesign() {
        let design = BrainBarDesign.current(in: designDefaults)
        guard design != shownDesign else { return }
        shownDesign = design
        let frame = window.frame
        switch design {
        case .classic:
            window.contentViewController = classicContent
            // Main's chrome: no toolbar, the default style.
            window.toolbar = nil
            window.toolbarStyle = .automatic
            window.titlebarSeparatorStyle = .automatic
            window.minSize = Self.minSize
            window.contentMinSize = Self.minSize
            window.contentMaxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
            pageObserver = nil
            headlineObserver = nil
            fittedFrameSize = nil
            // F04 and N02: back on Classic, the new design's window work stops and its Settings window closes.
            if hasNewDesignStore { newDesignStore.setWindowActive(false) }
            NDSettingsWindowController.closeIfOpen()
        case .new:
            let content = newContent ?? {
                let hosting = NSHostingController(rootView: NDWindowRoot(store: newDesignStore))
                hosting.sizingOptions = []
                return hosting
            }()
            newContent = content
            window.contentViewController = content
            // One unified 52 pt row: traffic lights, the centred pages control and the count (review fix 1). The toolbar
            // first, then the minimums. With the unified toolbar AppKit holds the frame minimum up to 20 pt above
            // 560 pt (measured 780 × 580 in a headless test); the page scrolls under the toolbar below its content.
            window.toolbar = newDesignToolbar.toolbar
            window.toolbarStyle = .unified
            window.titlebarSeparatorStyle = .none  // the root draws the one hairline at 52 pt
            // L17: the window takes each page's own height, width 900, and snaps on a tab switch with the top edge fixed.
            fitPage(newDesignStore.page)
            if pageObserver == nil {
                // D1: after the page is stored (didSet), never from `$page` (willSet).
                newDesignStore.pageDidChange = { [weak self] page in self?.fitPage(page) }
                pageObserver = AnyCancellable { [weak store = newDesignStore] in
                    MainActor.assumeIsolated { store?.pageDidChange = nil }
                }
                // A headline that gains or loses a fault line changes the page's height: refit when what it says changes.
                headlineObserver = newDesignStore.$model.map { _ in () }.receive(on: RunLoop.main).sink { [weak self] in
                    MainActor.assumeIsolated {
                        guard let self else { return }
                        let key = NDPageHeight.key(self.newDesignStore)
                        guard key != self.lastHeadlineKey else { return }
                        self.lastHeadlineKey = key
                        self.fitPage(self.newDesignStore.page)
                    }
                }
            }
            if window.isVisible { newDesignStore.setWindowActive(true) }
            return
        }
        // Keep the user's frame across the swap, never below the design's minimum.
        let minimum = window.frameRect(forContentRect: NSRect(origin: .zero, size: currentMinSize)).size
        window.setFrame(NSRect(x: frame.minX, y: frame.maxY - max(frame.height, minimum.height),
                               width: max(frame.width, minimum.width), height: max(frame.height, minimum.height)), display: false)
    }

    private var currentMinSize: NSSize { shownDesign == .new ? NDGrid.windowMinimum : Self.minSize }

    nonisolated(unsafe) private var pageObserver: AnyCancellable?
    nonisolated(unsafe) private var headlineObserver: AnyCancellable?
    /// The frame `fitPage` last chose; `windowWillResize` returns it in the new design.
    private(set) var fittedFrameSize: NSSize?
    private var lastHeadlineKey: String?

    /// Tests set the screen's visible height, so the L25 cap never depends on the machine running them.
    var visibleHeightOverride: CGFloat?
    /// The visible height the L25 cap is computed from: the window's screen, else the main screen.
    var fitScreenHeight: CGFloat? {
        visibleHeightOverride ?? window.screen?.visibleFrame.height ?? NSScreen.main?.visibleFrame.height
    }

    /// The page's size, exactly (min = max), keeping the window's top-left corner. R2 item 3: a shown window's height
    /// change animates over 200 ms ease-out, and the page is never drawn clipped: a taller page appears after the window
    /// has grown, a shorter one before it shrinks; under Reduce Motion (or hidden) both change at once.
    func fitPage(_ page: NDPage) {
        // R1.2 change 22 and L25: the page's own height, from its laid-out content, capped at the screen's visible height
        // minus 40 pt, past which the page scrolls under its fade and the footer stays.
        let size = NSSize(width: NDGrid.windowWidth, height: NDPageHeight.window(newDesignStore, page: page, screen: fitScreenHeight))
        let frameSize = window.frameRect(forContentRect: NSRect(origin: .zero, size: size)).size
        fittedFrameSize = frameSize
        let top = window.frame.maxY
        let target = NSRect(x: window.frame.minX, y: top - frameSize.height, width: frameSize.width, height: frameSize.height)
        let duration = NDMotion.pageResize(reduceMotion: NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)
        let order = NDPageSwitch.order(from: window.frame.height, to: target.height,
                                       animates: window.isVisible && duration != nil)
        guard order != .together, let duration else {
            // Content limits only: AppKit keeps one pair of size limits, and setting `minSize`/`maxSize` clears these.
            window.contentMinSize = size
            window.contentMaxSize = size
            window.setFrame(target, display: window.isVisible, animate: false)
            newDesignStore.shownPage = page
            return
        }
        if order == .pageThenFrame { newDesignStore.shownPage = page }
        // While it moves, the limits span both heights; at the end they are the page's size exactly.
        let current = window.contentRect(forFrameRect: window.frame).size
        window.contentMinSize = NSSize(width: size.width, height: min(current.height, size.height))
        window.contentMaxSize = NSSize(width: size.width, height: max(current.height, size.height))
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = duration
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            window.animator().setFrame(target, display: true)
        }, completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.fittedFrameSize == frameSize else { return }
                self.window.contentMinSize = size
                self.window.contentMaxSize = size
                if order == .frameThenPage { self.newDesignStore.shownPage = page }
            }
        })
    }

    /// The new design's pages: open the window on one (motion 2: no window animation).
    func showPage(_ page: NDPage) {
        applyDesign()
        guard shownDesign == .new else { return showDashboard() }
        newDesignStore.page = page
        show()
    }

    deinit {
        if let designObserver { NotificationCenter.default.removeObserver(designObserver) }
    }
#endif

    /// The hotkey and `brainbar://toggle`: open the window, or close it when it is open.
    func toggle(anchoredTo anchorView: NSView? = nil) {
        if window.isVisible {
            dismiss()
        } else {
            show(anchoredTo: anchorView)
        }
    }

    func show(anchoredTo anchorView: NSView? = nil) {
        Task { [driveAuth] in await driveAuth.refreshIfStale() }
#if BRAINBAR_UI
        if shownDesign == .new { newDesignStore.setWindowActive(true) }
#endif
        if panelState.selectedTab == .settings { panelState.settingsActivationRevision += 1 }
        if !hasPlacedWindow || isClosingWindow {
            placeWindow(near: anchorView ?? statusItemButton)
            hasPlacedWindow = true
        }
        activationPolicy?.windowOpened(window)
        if window.isMiniaturized { window.deminiaturize(nil) }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
#if BRAINBAR_UI
        if shownDesign == .new {
            Self.clearInitialFocus(window)
            // SwiftUI may hand focus on after the window becomes key; clear it again on the next turn.
            Task { @MainActor [weak window] in if let window { Self.clearInitialFocus(window) } }
        }
#endif
        isClosingWindow = false
    }

    /// R1.6 item 7: nothing holds focus when the window opens, so no focus ring shows until Tab. The window itself is
    /// the first responder; Tab from it starts the key-view loop at the first control.
    static func clearInitialFocus(_ window: NSWindow) {
        window.makeFirstResponder(nil)
    }

    func dismiss() {
        window.close()
    }

    func showDashboard() {
#if BRAINBAR_UI
        if BrainBarDesign.current(in: designDefaults) == .new { return showPage(.overview) }
#endif
        panelState.selectedTab = .dashboard
        show()
    }

    func showSettings() {
#if BRAINBAR_UI
        if BrainBarDesign.current(in: designDefaults) == .new {
            NDSettingsWindowController.show(store: newDesignStore, activationPolicy: activationPolicy)
            return
        }
#endif
        panelState.selectedTab = .settings
        show()
    }

#if BRAINBAR_UI
    func showSettings(section: BrainBarSettingsSection) {
        if BrainBarDesign.current(in: designDefaults) == .new {
            switch section {
            case .jobs: return showPage(.jobs)
            case .backups: return showPage(.backups)
            case .advanced: return showSettings()
            }
        }
        panelState.settingsNavigation.select(section)
        showSettings()
    }

    func showURLDestination(_ action: BrainBarURLAction) {
        switch action {
        case .dashboard: showDashboard()
        case .settings(let section): showSettings(section: section)
        case .toggle: toggle()
        }
    }
#endif

    /// The first time: the last saved frame, fitted to the current screens, else below the
    /// menu-bar icon, else centred. Afterwards the window keeps wherever the user left it.
    private func placeWindow(near anchorView: NSView?) {
        if let saved = frameStore.persistedFrame(),
           let restored = Self.restoredFrame(saved, minSize: Self.minSize, visibleFrames: screenFrames()) {
            window.setFrame(restored, display: false)
            return
        }
#if BRAINBAR_UI
        window.setContentSize(shownDesign == .new ? NDGrid.windowSize : Self.defaultSize)
#else
        window.setContentSize(Self.defaultSize)
#endif
        if let anchorView, let anchorWindow = anchorView.window,
           let screen = anchorWindow.screen ?? NSScreen.screens.first {
            let anchorRect = anchorWindow.convertToScreen(anchorView.convert(anchorView.bounds, to: nil))
            window.setFrameOrigin(Self.anchorOrigin(
                anchorRect: anchorRect, panelSize: window.frame.size, visibleFrame: screen.visibleFrame
            ))
        } else {
            window.center()
        }
    }

    private func persistFrame() {
        guard hasPlacedWindow, !isClosingWindow else { return }
        frameStore.persist(frame: window.frame)
    }

    func windowDidEndLiveResize(_ notification: Notification) { persistFrame() }
    func windowDidMove(_ notification: Notification) { persistFrame() }
    func windowWillClose(_ notification: Notification) { prepareForClose() }

    func prepareForClose() {
#if BRAINBAR_UI
        // F04: closing the window stops the hero and backup-history reads and cancels one in flight.
        if hasNewDesignStore { newDesignStore.setWindowActive(false) }
#endif
        persistFrame()
        // Policy changes and teardown can move a closed window. Keep the last user frame.
        isClosingWindow = true
    }

    /// A real window resizes on both axes, down to the size the dashboard needs.
    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
#if BRAINBAR_UI
        // L17: the new design's window is the page's size; the delegate holds it even if AppKit's layout clears the
        // content limits (it keeps one pair of size limits, and the toolbar's own layout can reset it).
        if shownDesign == .new, let fitted = fittedFrameSize { return fitted }
        let minimum = currentMinSize
#else
        let minimum = Self.minSize
#endif
        return NSSize(width: max(frameSize.width, minimum.width), height: max(frameSize.height, minimum.height))
    }

    /// Places the window as the first show would, without ordering it front (where AppKit would
    /// also constrain it to the real display).
    func placeWindowForTesting() {
        guard !hasPlacedWindow else { return }
        placeWindow(near: nil)
        hasPlacedWindow = true
    }

    func setDetailsExpandedForTesting(_ expanded: Bool) { panelState.detailsExpanded = expanded }
    func setSignalCoverageExpandedForTesting(_ expanded: Bool) { panelState.signalCoverageExpanded = expanded }
    func setAttentionExpandedForTesting(_ expanded: Bool) { panelState.attentionExpanded = expanded }

    var selectedTabForTesting: BrainBarTab { panelState.selectedTab }

#if BRAINBAR_UI
    var selectedSettingsSectionForTesting: BrainBarSettingsSection { panelState.settingsNavigation.selected }
    var sidebarSelectionForTesting: BrainBarSidebarItem { panelState.sidebarSelection }
    func selectSidebarItemForTesting(_ item: BrainBarSidebarItem) { panelState.select(item) }
#endif

    private static func makeWindow(contentViewController: NSViewController) -> BrainBarMainWindow {
        let window = BrainBarMainWindow(
            contentRect: NSRect(origin: .zero, size: defaultSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "BrainBar"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        window.hidesOnDeactivate = false
        window.level = .normal
        window.collectionBehavior = [.moveToActiveSpace]
        window.minSize = minSize
        window.contentViewController = contentViewController
        window.contentMinSize = minSize
        window.setContentSize(defaultSize)
        return window
    }

    /// Where a saved frame goes on the current screens: unchanged when it fits the visible frame
    /// it overlaps most; otherwise shrunk to fit (never below `minSize`) and moved inside it. Nil
    /// when it is smaller than `minSize` or no longer on any screen, so the default placement is
    /// used. Doing this ourselves keeps the result the same on every display, instead of leaving
    /// AppKit to nudge a partly off-screen window when it is ordered front.
    static func restoredFrame(_ saved: NSRect, minSize: NSSize, visibleFrames: [NSRect]) -> NSRect? {
        guard saved.width >= minSize.width, saved.height >= minSize.height else { return nil }
        func overlap(_ screen: NSRect) -> CGFloat {
            let shared = screen.intersection(saved)
            return shared.isNull ? 0 : shared.width * shared.height
        }
        guard let screen = visibleFrames.max(by: { overlap($0) < overlap($1) }), overlap(screen) > 0 else { return nil }
        let width = min(saved.width, max(screen.width, minSize.width))
        let height = min(saved.height, max(screen.height, minSize.height))
        return NSRect(
            x: min(max(saved.minX, screen.minX), max(screen.maxX - width, screen.minX)),
            y: min(max(saved.minY, screen.minY), max(screen.maxY - height, screen.minY)),
            width: width,
            height: height
        )
    }

    static func anchorOrigin(anchorRect: NSRect, panelSize: NSSize, visibleFrame: NSRect) -> NSPoint {
        let gap = BrainBarWindowPlacement.menuBarIconGap
        let targetX = min(max(anchorRect.maxX - panelSize.width, visibleFrame.minX), visibleFrame.maxX - panelSize.width)
        let targetY = max(min(anchorRect.minY - gap - panelSize.height, visibleFrame.maxY - panelSize.height), visibleFrame.minY)
        return NSPoint(x: targetX, y: targetY)
    }
}
