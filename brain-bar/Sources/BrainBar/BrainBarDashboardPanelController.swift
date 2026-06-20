import AppKit
import SwiftUI

final class BrainBarDashboardPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
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
    static let maxSize = NSSize(width: 1_600, height: 1_200)

    let panelForTesting: NSPanel
    let contentViewControllerForTesting: NSViewController
    var isShownForTesting: Bool { panel.isVisible }

    private let panel: NSPanel
    private var clickOutsideMonitor: Any?
    private var localClickMonitor: Any?
    private var shownAt: Date = .distantPast
    weak var statusItemButton: NSView?

    init(runtime: BrainBarRuntime) {
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false)
                .frame(minWidth: Self.minSize.width, minHeight: Self.minSize.height)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        )
        hostingController.view.frame = NSRect(origin: .zero, size: Self.defaultSize)
        hostingController.view.autoresizingMask = [.width, .height]

        contentViewControllerForTesting = hostingController
        panel = Self.makePanel(contentViewController: hostingController)
        panelForTesting = panel
        super.init()
        panel.delegate = self
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
        dismiss()
    }

    private func dismissIfLocalClickOutside(_ event: NSEvent) {
        guard panel.isVisible, Date().timeIntervalSince(shownAt) > 0.20 else { return }
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
