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
    var contentViewControllerForTesting: NSViewController { window.contentViewController! }
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
    private var newContent: NSHostingController<NDWindowRoot>?
    private let runtimeForDesign: BrainBarRuntime
    private let workspaceForDesign: any BrainBarWorkspaceActing
    private var hasNewDesignStore = false
    private lazy var newDesignToolbar = NDToolbar(store: newDesignStore)
    lazy var newDesignStore: NDStore = {
        hasNewDesignStore = true
        let store = NDStore.shared(for: runtimeForDesign, workspace: workspaceForDesign)
        store.onOpenWindow = { [weak self] page in self?.showPage(page) }
        store.onOpenSettings = { [weak self] in self?.showSettings() }
        store.windowActivationPolicy = activationPolicy
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
#if BRAINBAR_UI
        let hostingController = NSViewController()
        hostingController.view = NSView(frame: NSRect(origin: .zero, size: Self.defaultSize))
#else
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(workspace: workspace, runtime: runtime, managesWindowFrame: false, panelState: panelState)
                .frame(minWidth: Self.minSize.width)
                .frame(maxWidth: .infinity)
        )
        hostingController.sizingOptions = []
        hostingController.view.frame = NSRect(origin: .zero, size: Self.defaultSize)
        hostingController.view.autoresizingMask = [.width, .height]

#endif

#if BRAINBAR_UI
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
        installWindowContent()
#endif
    }

#if BRAINBAR_UI
    /// Installs the app's only window content once. Old design preferences are ignored.
    private func installWindowContent() {
        guard newContent == nil else { return }
        let content = NSHostingController(rootView: NDWindowRoot(store: newDesignStore))
        content.sizingOptions = []
        newContent = content
        window.contentViewController = content
        // One unified 52 pt row: traffic lights, the centred pages control and the count (review fix 1). The toolbar
        // first, then the minimums. With the unified toolbar AppKit holds the frame minimum up to 20 pt above
        // 560 pt (measured 780 × 580 in a headless test); the page scrolls under the toolbar below its content.
        window.toolbar = newDesignToolbar.toolbar
        window.toolbarStyle = .unified
        window.titlebarSeparatorStyle = .none  // the root draws the one hairline at 52 pt
        // Width 900; the height is the tallest page's (or, with `NDWindowSizing.perPageHeights`, L17's per-page height,
        // resized on a tab switch with the top edge fixed).
        fitPage(newDesignStore.page)
        if pageObserver == nil {
            // D1: after the page is stored (didSet), never from `$page` (willSet).
            newDesignStore.pageDidChange = { [weak self] page in self?.fitPage(page) }
            // Tab-swipe spec 4: no height change while the pill and pages are between tabs. A refit asked for then
            // (a page change, any store update) is held and runs once at rest; a resize still running when they
            // start to move again finishes at once.
            newDesignStore.tabsDidRest = { [weak self] in self?.runHeldPageFit() }
            newDesignStore.tabsWillMove = { [weak self] in self?.finishPageResizeNow() }
            pageObserver = AnyCancellable { [weak store = newDesignStore] in
                MainActor.assumeIsolated { store?.pageDidChange = nil }
            }
            // L57: inputs can change Jobs and Backups without changing the headline or model.
            // Coalesce willSet publications; measure only after the complete store update.
            pageFitObserver = NDPageFitObserver(store: newDesignStore) { [weak self] in
                guard let self else { return }
                self.fitPage(self.newDesignStore.page)
            }
        }
        if window.isVisible { newDesignStore.setWindowActive(true) }
    }

    private var currentMinSize: NSSize { NDGrid.windowMinimum }

    nonisolated(unsafe) private var pageObserver: AnyCancellable?
    private var pageFitObserver: NDPageFitObserver?
    /// The frame `fitPage` last chose; `windowWillResize` returns it in the new design.
    private(set) var fittedFrameSize: NSSize?
    private var pageResizeInFlight = false
    private var pageFitPending = false

    var visibleFrameOverride: NSRect?
    private var fitVisibleFrame: NSRect? {
        if let visibleFrameOverride { return visibleFrameOverride }
        let frame = window.screen?.visibleFrame ?? NSScreen.main?.visibleFrame
        guard let visibleHeightOverride else { return frame }
        return NSRect(x: frame?.minX ?? 0, y: frame?.minY ?? 0,
                      width: frame?.width ?? NDGrid.windowWidth, height: visibleHeightOverride)
    }

    var fitTitleBarHeight: CGFloat {
        window.frameRect(forContentRect: NSRect(x: 0, y: 0, width: NDGrid.windowWidth, height: 100)).height - 100
    }

    /// Tests set the screen's visible height, so the L57 cap never depends on the machine running them.
    var visibleHeightOverride: CGFloat?
    /// The visible height the L57 cap is computed from: the window's screen, else the main screen.
    var fitScreenHeight: CGFloat? {
        fitVisibleFrame?.height
    }

    /// The page's size, exactly (min = max), keeping the window's top-left corner. R2 item 3: a shown window's height
    /// change animates over 200 ms ease-out, and the page is never drawn clipped: a taller page appears after the window
    /// has grown, a shorter one before it shrinks; under Reduce Motion (or hidden) both change at once.
    /// R1.10.2 and R1.10.3: the new page is laid out now, before the height moves, by construction: the page is handed to
    /// the hosting controller as its root view's own value (applied on the layout that follows, with no wait for the store's
    /// change to be observed), then the content is laid out. Before, the swap only marked the store, and SwiftUI applied it
    /// on a later pass that a displayless machine could defer past the whole resize.
    private func layOutShownPage(_ page: NDPage) {
        newContent?.rootView = NDWindowRoot(store: newDesignStore, page: page)
        guard let content = window.contentView else { return }
        content.needsLayout = true
        content.layoutSubtreeIfNeeded()
    }

    /// Tests only: run a page switch as a visible window would (the window itself is never shown).
    var animatesPageSwitchForTesting: Bool?
    /// Tests only: the Reduce Motion setting a page switch reads (a CI runner's session has it on).
    var reduceMotionForTesting: Bool?
    /// Tests only: the window-frame animation of a page switch (target, duration, curve, completion). A headless runner
    /// never advances AppKit's animation, so a test steps the frames itself through this seam.
    var pageFrameAnimatorForTesting: ((NSRect, Double, CAMediaTimingFunction, @escaping () -> Void) -> Void)?

    /// A refit asked for while the tabs moved.
    private var pageFitHeld = false
    /// The running resize: its target, and its completion (run once, by AppKit or by `finishPageResizeNow`).
    private var runningPageResize: (target: NSRect, finish: () -> Void)?
    private var pageResizeGeneration = 0

    private func runHeldPageFit() {
        guard pageFitHeld else { return }
        pageFitHeld = false
        fitPage(newDesignStore.page)
    }

    /// The tabs started to move during a resize: land the window on the resize's target now, so its height never
    /// moves while two pages share it.
    func finishPageResizeNow() {
        guard let running = runningPageResize else { return }
        runningPageResize = nil
        if pageFrameAnimatorForTesting == nil {
            // A zero-length animation of the same frame replaces the running one.
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0
                window.animator().setFrame(running.target, display: true)
            }
        }
        window.setFrame(running.target, display: window.isVisible, animate: false)
        running.finish()
    }

    func fitPage(_ page: NDPage) {
        // Tab-swipe spec 4: the height never moves while the pill and pages are between tabs; it waits for the rest.
        guard newDesignStore.tabs.phase == .resting else {
            pageFitHeld = true
            return
        }
        // Finish one frame animation before starting another; the newest content wins at its completion.
        guard !pageResizeInFlight else {
            pageFitPending = true
            if newDesignStore.shownPage != page {
                newDesignStore.shownPage = page
                layOutShownPage(page)
            }
            return
        }
        let size = NSSize(width: NDGrid.windowWidth,
                          height: NDPageHeight.fitted(newDesignStore, page: page, screen: fitScreenHeight,
                                                      titleBarHeight: fitTitleBarHeight))
        let frameSize = window.frameRect(forContentRect: NSRect(origin: .zero, size: size)).size
        let target = NDPageHeight.frame(size: frameSize, keeping: window.frame, visibleFrame: fitVisibleFrame)
        // A publication that changes only words or wave samples needs no window resize.
        if fittedFrameSize == frameSize, window.frame == target, newDesignStore.shownPage == page { return }
        fittedFrameSize = frameSize
        let duration = NDMotion.pageResize(reduceMotion: reduceMotionForTesting ?? NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)
        let order = NDPageSwitch.order(from: window.frame.height, to: target.height,
                                       animates: (animatesPageSwitchForTesting ?? window.isVisible) && duration != nil)
        guard order != .together, let duration else {
            // Content limits only: AppKit keeps one pair of size limits, and setting `minSize`/`maxSize` clears these.
            window.contentMinSize = size
            window.contentMaxSize = size
            window.setFrame(target, display: window.isVisible, animate: false)
            newDesignStore.shownPage = page
            layOutShownPage(page)
            return
        }
        // R1.8 item 1: the new page first, in both directions; then the height. R1.10.1: at once, never animated: an
        // animated swap is a transition, and the page's scroll view lays out both of a transition's pages stacked, the old
        // on top, so the shrinking window showed the old page under the new tab.
        var swap = Transaction()
        swap.disablesAnimations = true
        withTransaction(swap) { newDesignStore.shownPage = page }
        layOutShownPage(page)
        // While it moves, the limits span both heights; at the end they are the page's size exactly.
        let current = window.contentRect(forFrameRect: window.frame).size
        window.contentMinSize = NSSize(width: size.width, height: min(current.height, size.height))
        window.contentMaxSize = NSSize(width: size.width, height: max(current.height, size.height))
        let c = NDPageSwitch.curve
        let curve = CAMediaTimingFunction(controlPoints: c.0, c.1, c.2, c.3)
        pageResizeInFlight = true
        pageResizeGeneration += 1
        let generation = pageResizeGeneration
        let finish: () -> Void = { [weak self] in
            MainActor.assumeIsolated {
                // Once per resize: AppKit's completion after `finishPageResizeNow` already ran it changes nothing.
                guard let self, self.pageResizeGeneration == generation, self.pageResizeInFlight else { return }
                self.runningPageResize = nil
                self.pageResizeInFlight = false
                self.window.contentMinSize = size
                self.window.contentMaxSize = size
                if self.pageFitPending {
                    self.pageFitPending = false
                    self.fitPage(self.newDesignStore.page)
                }
            }
        }
        runningPageResize = (target, finish)
        if let animator = pageFrameAnimatorForTesting { return animator(target, duration, curve, finish) }
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = duration
            context.timingFunction = curve
            window.animator().setFrame(target, display: true)
        }, completionHandler: finish)
    }

    /// The new design's pages: open the window on one (motion 2: no window animation).
    func showPage(_ page: NDPage) {
        installWindowContent()
        newDesignStore.page = page
        show()
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
        newDesignStore.setWindowActive(true)
#endif
        if panelState.selectedTab == .settings { panelState.settingsActivationRevision += 1 }
        if !hasPlacedWindow || isClosingWindow {
            placeWindow(near: anchorView ?? statusItemButton)
            hasPlacedWindow = true
        }
#if BRAINBAR_UI
        fitPage(newDesignStore.page)
#endif
        activationPolicy?.windowOpened(window)
        if window.isMiniaturized { window.deminiaturize(nil) }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
#if BRAINBAR_UI
        Self.clearInitialFocus(window)
        // SwiftUI may hand focus on after the window becomes key; clear it again on the next turn.
        Task { @MainActor [weak window] in if let window { Self.clearInitialFocus(window) } }
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
        showPage(.overview)
#else
        panelState.selectedTab = .dashboard
        show()
#endif
    }

    func showSettings() {
#if BRAINBAR_UI
        NDSettingsWindowController.show(store: newDesignStore, activationPolicy: activationPolicy)
#else
        panelState.selectedTab = .settings
        show()
#endif
    }

#if BRAINBAR_UI
    func showSettings(section: BrainBarSettingsSection) {
        switch section {
        case .jobs: showPage(.jobs)
        case .backups: showPage(.backups)
        case .advanced: showSettings()
        }
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
#if BRAINBAR_UI
        let restorationMinimum = currentMinSize
#else
        let restorationMinimum = Self.minSize
#endif
        if let saved = frameStore.persistedFrame(),
           let restored = Self.restoredFrame(saved, minSize: restorationMinimum, visibleFrames: screenFrames()) {
            window.setFrame(restored, display: false)
#if BRAINBAR_UI
            fitPage(newDesignStore.page)
#endif
            return
        }
#if !BRAINBAR_UI
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

#if BRAINBAR_UI
    func windowDidChangeScreen(_ notification: Notification) {
        pageFitObserver?.schedule()
    }
#endif

    /// A real window resizes on both axes, down to the size the dashboard needs.
    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
#if BRAINBAR_UI
        // L17 / NDWindowSizing: the new design's window is the fitted size; the delegate holds it even if AppKit's layout clears the
        // content limits (it keeps one pair of size limits, and the toolbar's own layout can reset it).
        if let fitted = fittedFrameSize { return fitted }
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
