import AppKit
import SwiftUI

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
}

/// The one BrainBar window (#963, vNext D5): a real, titled window that stays open until it is
/// closed, like VoiceBar's Settings window. The old menu-bar panel floated above everything and
/// dismissed itself on any click away, which broke click-through testing.
final class BrainBarMainWindow: NSWindow {
    /// Cmd-, from inside the window. The window handles its own key equivalents because an
    /// accessory app shows no menu bar to route them.
    var onSettingsShortcut: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

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
    private var hasPlacedWindow = false
    /// Only used to place the very first window below the menu-bar icon.
    weak var statusItemButton: NSView?

    init(
        runtime: BrainBarRuntime,
        frameStore: BrainBarWindowFrameStore = BrainBarWindowFrameStore(key: frameDefaultsKey),
        screenFrames: @escaping () -> [NSRect] = { NSScreen.screens.map(\.visibleFrame) }
    ) {
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false, panelState: panelState)
                .frame(minWidth: Self.minSize.width)
                .frame(maxWidth: .infinity)
        )
        hostingController.sizingOptions = []
        hostingController.view.frame = NSRect(origin: .zero, size: Self.defaultSize)
        hostingController.view.autoresizingMask = [.width, .height]

        contentViewControllerForTesting = hostingController
        self.frameStore = frameStore
        self.screenFrames = screenFrames
        window = Self.makeWindow(contentViewController: hostingController)
        windowForTesting = window
        super.init()
        window.delegate = self
        window.onSettingsShortcut = { [weak self] in self?.showSettings() }
        BrainBarSettingsActions.installOpenHandler { [weak self] in
            self?.showSettings()
        }
    }

    /// The hotkey and `brainbar://toggle`: open the window, or close it when it is open.
    func toggle(anchoredTo anchorView: NSView? = nil) {
        if window.isVisible {
            dismiss()
        } else {
            show(anchoredTo: anchorView)
        }
    }

    func show(anchoredTo anchorView: NSView? = nil) {
        if panelState.selectedTab == .settings { panelState.settingsActivationRevision += 1 }
        if !hasPlacedWindow {
            placeWindow(near: anchorView ?? statusItemButton)
            hasPlacedWindow = true
        }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    func dismiss() {
        persistFrame()
        window.orderOut(nil)
    }

    func showDashboard() {
        panelState.selectedTab = .dashboard
        show()
    }

    func showSettings() {
        panelState.selectedTab = .settings
        show()
    }

#if BRAINBAR_UI
    func showSettings(section: BrainBarSettingsSection) {
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
        window.setContentSize(Self.defaultSize)
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
        guard hasPlacedWindow else { return }
        frameStore.persist(frame: window.frame)
    }

    func windowDidEndLiveResize(_ notification: Notification) { persistFrame() }
    func windowDidMove(_ notification: Notification) { persistFrame() }
    func windowWillClose(_ notification: Notification) { persistFrame() }

    /// A real window resizes on both axes, down to the size the dashboard needs.
    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        NSSize(width: max(frameSize.width, Self.minSize.width), height: max(frameSize.height, Self.minSize.height))
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
