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

final class BrainBarDashboardPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

@MainActor
final class BrainBarDashboardPanelState: ObservableObject {
    @Published var attentionExpanded = false
    @Published var detailsExpanded = BrainBarOnePageComposition.detailsExpandedByDefault
    @Published var signalCoverageExpanded = false
    @Published var dashboardHeight: CGFloat = 0
    @Published var headerHeight: CGFloat = 0
    @Published var searchOverlayPresented = false
    @Published var graphPresented = false
    @Published var selectedTab: BrainBarTab = .dashboard
    @Published var settingsActivationRevision = 0
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
            searchOverlayPresented || graphPresented ? 640 : 300
        )
    }

    func disclosureAnimationDidComplete() {}
}

@MainActor
final class BrainBarDashboardPanelController: NSObject, NSWindowDelegate {
    static let defaultSize = NSSize(
        width: BrainBarWindowPlacement.defaultSize.width,
        height: BrainBarWindowPlacement.defaultSize.height
    )
    static let minSize = NSSize(width: BrainBarWindowPlacement.minimumSize.width, height: 300)
    static let maxSize = NSSize(width: 1_600, height: 1_200)

    let panelForTesting: NSPanel
    let contentViewControllerForTesting: NSViewController
    var isShownForTesting: Bool { panel.isVisible }
    var naturalDashboardHeightForTesting: CGFloat { panelState.dashboardHeight }

    private let panel: NSPanel
    private let panelState = BrainBarDashboardPanelState()
    private weak var lastShownAnchor: NSView?
    private var lastAnchorScreenRect: NSRect?
    private weak var lastAnchorScreen: NSScreen?
    private var clickOutsideMonitor: Any?
    private var localClickMonitor: Any?
    private var shownAt: Date = .distantPast
    weak var statusItemButton: NSView?

    init(runtime: BrainBarRuntime) {
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false, panelState: panelState)
                .frame(minWidth: Self.minSize.width)
                .frame(maxWidth: .infinity)
        )
        hostingController.sizingOptions = []
        hostingController.view.frame = NSRect(origin: .zero, size: Self.defaultSize)
        hostingController.view.autoresizingMask = [.width, .height]

        contentViewControllerForTesting = hostingController
        panel = Self.makePanel(contentViewController: hostingController)
        panelForTesting = panel
        super.init()
        panel.delegate = self
        BrainBarSettingsActions.installOpenHandler { [weak self] in
            self?.showSettings()
        }
    }

    func toggle(anchoredTo anchorView: NSView? = nil) {
        if panel.isVisible {
            dismiss()
        } else {
            show(anchoredTo: anchorView)
        }
    }

    func show(anchoredTo anchorView: NSView? = nil) {
        guard let anchorView else { return }
        if panelState.selectedTab == .settings { panelState.settingsActivationRevision += 1 }
        let anchorWindow = anchorView.window
        let anchorRect = anchorWindow?.convertToScreen(anchorView.convert(anchorView.bounds, to: nil))
        let anchorScreen = anchorWindow?.screen
        if lastShownAnchor !== anchorView || lastAnchorScreenRect != anchorRect || lastAnchorScreen !== anchorScreen {
            positionPanel(below: anchorView)
            lastShownAnchor = anchorView
            lastAnchorScreenRect = anchorRect
            lastAnchorScreen = anchorScreen
        }
        NSApp.activate(ignoringOtherApps: true)
        panel.makeKeyAndOrderFront(nil)
        panel.orderFrontRegardless()
        shownAt = Date()
        installClickOutsideMonitor()
    }

    func dismiss() {
        removeClickOutsideMonitor()
        panel.orderOut(nil)
    }

    func showSettings() {
        panelState.selectedTab = .settings
        if panel.isVisible {
            NSApp.activate(ignoringOtherApps: true)
            panel.makeKeyAndOrderFront(nil)
        } else {
            show(anchoredTo: statusItemButton)
        }
    }

    private func installClickOutsideMonitor() {
        removeClickOutsideMonitor()
        let mask: NSEvent.EventTypeMask = [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        clickOutsideMonitor = NSEvent.addGlobalMonitorForEvents(matching: mask) { [weak self] _ in
            Task { @MainActor in self?.dismissIfClickOutside() }
        }
        localClickMonitor = NSEvent.addLocalMonitorForEvents(matching: mask) { [weak self] event in
            Task { @MainActor in self?.dismissIfLocalClickOutside(event) }
            return event
        }
    }

    private func removeClickOutsideMonitor() {
        if let clickOutsideMonitor { NSEvent.removeMonitor(clickOutsideMonitor) }
        if let localClickMonitor { NSEvent.removeMonitor(localClickMonitor) }
        clickOutsideMonitor = nil
        localClickMonitor = nil
    }

    private func dismissIfClickOutside() {
        guard panel.isVisible, Date().timeIntervalSince(shownAt) > 0.20 else { return }
        guard NSApp.modalWindow == nil, panel.attachedSheet == nil else { return }
        dismiss()
    }

    private func dismissIfLocalClickOutside(_ event: NSEvent) {
        guard panel.isVisible, Date().timeIntervalSince(shownAt) > 0.20 else { return }
        guard NSApp.modalWindow == nil, panel.attachedSheet == nil else { return }
        if event.window === panel { return }
        if let button = statusItemButton, event.window === button.window { return }   // let toggle() own the menubar click
        dismiss()
    }

    func windowDidResignKey(_ notification: Notification) {
        guard panel.isVisible, Date().timeIntervalSince(shownAt) > 0.20 else { return }
        guard NSApp.modalWindow == nil, panel.attachedSheet == nil else { return }
        dismiss()
    }

    func windowWillClose(_ notification: Notification) {
        removeClickOutsideMonitor()
    }

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        NSSize(width: max(frameSize.width, Self.minSize.width), height: sender.frame.height)
    }

    func setDetailsExpandedForTesting(_ expanded: Bool) { panelState.detailsExpanded = expanded }
    func setSignalCoverageExpandedForTesting(_ expanded: Bool) { panelState.signalCoverageExpanded = expanded }
    func setSearchOverlayPresentedForTesting(_ presented: Bool) { panelState.searchOverlayPresented = presented }
    var selectedTabForTesting: BrainBarTab { panelState.selectedTab }

    private static func makePanel(contentViewController: NSViewController) -> NSPanel {
        let panel = BrainBarDashboardPanel(
            contentRect: NSRect(origin: .zero, size: defaultSize),
            styleMask: [.titled, .fullSizeContentView, .closable, .resizable],
            backing: .buffered,
            defer: false
        )
        panel.title = "BrainBar"
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isReleasedWhenClosed = false
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        panel.level = .statusBar
        panel.becomesKeyOnlyIfNeeded = false
        panel.minSize = minSize
        panel.maxSize = maxSize
        panel.contentViewController = contentViewController
        panel.contentMinSize = minSize
        panel.setContentSize(defaultSize)
        return panel
    }

    private func positionPanel(below anchorView: NSView) {
        guard let anchorWindow = anchorView.window,
              let screen = anchorWindow.screen ?? NSScreen.screens.first else {
            panel.setFrame(NSRect(origin: .zero, size: panel.frame.size), display: false)
            return
        }

        let anchorRectInWindow = anchorView.convert(anchorView.bounds, to: nil)
        let anchorRect = anchorWindow.convertToScreen(anchorRectInWindow)
        panel.setFrameOrigin(Self.anchorOrigin(anchorRect: anchorRect, panelSize: panel.frame.size, visibleFrame: screen.visibleFrame))
    }

    static func anchorOrigin(anchorRect: NSRect, panelSize: NSSize, visibleFrame: NSRect) -> NSPoint {
        let gap = BrainBarWindowPlacement.menuBarIconGap
        let targetX = min(max(anchorRect.maxX - panelSize.width, visibleFrame.minX), visibleFrame.maxX - panelSize.width)
        let targetY = max(min(anchorRect.minY - gap - panelSize.height, visibleFrame.maxY - panelSize.height), visibleFrame.minY)
        return NSPoint(x: targetX, y: targetY)
    }
}
