import AppKit
import SwiftUI

@MainActor
final class BrainBarDashboardPanelController {
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
    }

    func dismiss() {
        panel.orderOut(nil)
    }

    private static func makePanel(contentViewController: NSViewController) -> NSPanel {
        let panel = NSPanel(
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
        panel.hidesOnDeactivate = true
        panel.level = .statusBar
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
