import AppKit
import Combine
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
    @Published var detailsExpanded = BrainBarOnePageComposition.detailsExpandedByDefault
    @Published var signalCoverageExpanded = false
    @Published var dashboardHeight: CGFloat = 0
    @Published var headerHeight: CGFloat = 0
    @Published var searchOverlayPresented = false
    @Published var graphPresented = false
    @Published private(set) var disclosureAnimationRevision = 0
    var fittingHeight: CGFloat {
        max(
            BrainBarDisclosureAnimation.windowHeight(
                containerHeight: dashboardHeight,
                chromeAndSurroundingContentHeight: headerHeight
            ),
            searchOverlayPresented || graphPresented ? 640 : 300
        )
    }

    func disclosureAnimationDidComplete() {
        disclosureAnimationRevision += 1
    }
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
    var measuredContentHeightForTesting: CGFloat { panelState.headerHeight + panelState.dashboardHeight }
    private(set) var contentSizeApplicationCountForTesting = 0
    private(set) var reentrantFitAttemptCountForTesting = 0
    var contentSizeDidApplyForTesting: (() -> Void)?
    private var visibleFrameOverrideForTesting: CGRect?

    private let panel: NSPanel
    private let panelState = BrainBarDashboardPanelState()
    private var sizingObservations = Set<AnyCancellable>()
    private var isApplyingFittedSize = false
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
        installSizingObservers()
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
        positionPanel(below: anchorView)
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
        guard !BrainBarSettingsActions.suppressDashboardResignDismiss else { return }
        dismiss()
    }

    private func dismissIfLocalClickOutside(_ event: NSEvent) {
        guard panel.isVisible, Date().timeIntervalSince(shownAt) > 0.20 else { return }
        guard !BrainBarSettingsActions.suppressDashboardResignDismiss else { return }
        if event.window === panel { return }
        if let button = statusItemButton, event.window === button.window { return }   // let toggle() own the menubar click
        dismiss()
    }

    func windowDidResignKey(_ notification: Notification) {
        guard panel.isVisible, Date().timeIntervalSince(shownAt) > 0.20 else { return }
        guard !BrainBarSettingsActions.suppressDashboardResignDismiss else { return }
        dismiss()
    }

    func windowWillClose(_ notification: Notification) {
        removeClickOutsideMonitor()
    }

    func windowDidResize(_ notification: Notification) {
        guard !isApplyingFittedSize else { return }
        fitPanelToContent()
    }

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        NSSize(width: max(frameSize.width, Self.minSize.width), height: sender.frame.height)
    }

    func setDetailsExpandedForTesting(_ expanded: Bool) {
        panelState.detailsExpanded = expanded
        DispatchQueue.main.async { [weak self] in self?.panelState.disclosureAnimationDidComplete() }
    }

    func setSignalCoverageExpandedForTesting(_ expanded: Bool) {
        panelState.signalCoverageExpanded = expanded
        DispatchQueue.main.async { [weak self] in self?.panelState.disclosureAnimationDidComplete() }
    }
    func setSearchOverlayPresentedForTesting(_ presented: Bool) { panelState.searchOverlayPresented = presented }

    func resetGeometryMetricsForTesting() {
        contentSizeApplicationCountForTesting = 0
        reentrantFitAttemptCountForTesting = 0
    }

    func setDashboardHeightForTesting(_ height: CGFloat) {
        panelState.dashboardHeight = height
    }

    func setVisibleFrameForTesting(_ visibleFrame: CGRect?) {
        visibleFrameOverrideForTesting = visibleFrame
    }

    func completeDisclosureTransitionForTesting(expanded: Bool) {
        panelState.detailsExpanded = expanded
        panelState.disclosureAnimationDidComplete()
    }

    private func installSizingObservers() {
        panelState.$dashboardHeight
            .combineLatest(panelState.$headerHeight)
            .filter { dashboardHeight, headerHeight in dashboardHeight > 0 && headerHeight > 0 }
            .prefix(1)
            .sink { [weak self] _ in
                DispatchQueue.main.async { self?.fitPanelToContent() }
            }
            .store(in: &sizingObservations)

        panelState.$disclosureAnimationRevision
            .dropFirst()
            .sink { [weak self] _ in
                DispatchQueue.main.async { self?.fitPanelToContent() }
            }
            .store(in: &sizingObservations)

        panelState.$searchOverlayPresented
            .combineLatest(panelState.$graphPresented)
            .removeDuplicates { lhs, rhs in lhs.0 == rhs.0 && lhs.1 == rhs.1 }
            .dropFirst()
            .sink { [weak self] _ in
                DispatchQueue.main.async { self?.fitPanelToContent() }
            }
            .store(in: &sizingObservations)
    }

    private func fitPanelToContent() {
        guard !isApplyingFittedSize else {
            reentrantFitAttemptCountForTesting += 1
            return
        }

        let width = panel.contentLayoutRect.width
        let originalFrame = panel.frame
        let visibleFrame = visibleFrameOverrideForTesting
            ?? statusItemButton?.window?.screen?.visibleFrame
            ?? panel.screen?.visibleFrame
        let titlebarInset = max((panel.contentView?.frame.height ?? panel.contentLayoutRect.height)
            - panel.contentLayoutRect.height, 0)
        let availableFrameHeight = visibleFrame.map { frame in
            let preservedTop = min(max(originalFrame.maxY, frame.minY), frame.maxY)
            return max(preservedTop - frame.minY, 0)
        } ?? Self.maxSize.height
        let maxContentHeight = min(
            max(availableFrameHeight - titlebarInset, 0),
            max(panel.maxSize.height - titlebarInset, 0)
        )
        let usableHeight = min(panelState.fittingHeight, maxContentHeight)
        let contentHeight = usableHeight + titlebarInset
        let needsResize = abs(panel.contentLayoutRect.height - usableHeight) > 0.5
        guard needsResize else {
            lockContentHeight(contentHeight)
            return
        }

        isApplyingFittedSize = true
        contentSizeApplicationCountForTesting += 1
        panel.contentMinSize = NSSize(width: Self.minSize.width, height: 0)
        panel.contentMaxSize = NSSize(width: Self.maxSize.width, height: Self.maxSize.height)

        let currentContentHeight = panel.contentView?.frame.height ?? panel.contentLayoutRect.height
        let targetFrameHeight = originalFrame.height + (contentHeight - currentContentHeight)
        var targetFrame = CGRect(
            x: originalFrame.minX,
            y: originalFrame.maxY - targetFrameHeight,
            width: originalFrame.width,
            height: targetFrameHeight
        )
        if let visibleFrame {
            targetFrame = BrainBarWindowPlacement.clamp(frame: targetFrame, to: visibleFrame)
        }

        panel.setContentSize(NSSize(width: width, height: contentHeight))
        contentSizeDidApplyForTesting?()
        panel.setFrameOrigin(targetFrame.origin)
        lockContentHeight(contentHeight)
        isApplyingFittedSize = false
    }

    private func lockContentHeight(_ contentHeight: CGFloat) {
        panel.contentMinSize = NSSize(width: Self.minSize.width, height: contentHeight)
        panel.contentMaxSize = NSSize(width: Self.maxSize.width, height: contentHeight)
    }

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
        let visibleFrame = screen.visibleFrame
        let panelSize = panel.frame.size
        let gap = BrainBarWindowPlacement.menuBarIconGap
        let targetX = min(
            max(anchorRect.maxX - panelSize.width, visibleFrame.minX),
            visibleFrame.maxX - panelSize.width
        )
        let targetY = max(
            min(anchorRect.minY - gap - panelSize.height, visibleFrame.maxY - panelSize.height),
            visibleFrame.minY
        )
        panel.setFrameOrigin(NSPoint(x: targetX, y: targetY))
    }
}
