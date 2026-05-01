import AppKit
import SwiftUI

final class BrainBarDashboardPanel: NSPanel {
    var onEscape: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func cancelOperation(_ sender: Any?) {
        onEscape?()
    }
}

@MainActor
final class BrainBarDashboardPanelController {
    static let autosaveName = "BrainBarPanel"
    static let defaultSize = NSSize(width: 1_348, height: 1_078)
    static let minSize = NSSize(width: 760, height: 560)

    let panelForTesting: BrainBarDashboardPanel

    private let panel: BrainBarDashboardPanel
    private let runtime: BrainBarRuntime

    init(runtime: BrainBarRuntime) {
        self.runtime = runtime
        panel = BrainBarDashboardPanel(
            contentRect: NSRect(origin: .zero, size: Self.defaultSize),
            styleMask: [.titled, .resizable, .closable, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panelForTesting = panel
        configurePanel()
    }

    func toggle() {
        if panel.isVisible {
            dismiss()
        } else {
            show()
        }
    }

    func show() {
        if !panel.isVisible, panel.frame.origin == .zero {
            centerPanel()
        }
        NSApp.activate(ignoringOtherApps: true)
        panel.makeKeyAndOrderFront(nil)
        panel.orderFrontRegardless()
    }

    func dismiss() {
        panel.orderOut(nil)
    }

    private func configurePanel() {
        panel.title = "BrainBar"
        panel.titleVisibility = .visible
        panel.titlebarAppearsTransparent = false
        panel.isMovableByWindowBackground = true
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.minSize = Self.minSize
        panel.isReleasedWhenClosed = false
        panel.backgroundColor = .windowBackgroundColor
        panel.isOpaque = true
        panel.hasShadow = true
        panel.onEscape = { [weak self] in
            self?.dismiss()
        }
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        )
        hostingController.view.autoresizingMask = [.width, .height]
        panel.contentViewController = hostingController
        panel.setFrame(initialFrame(), display: false)
        panel.setFrameAutosaveName(Self.autosaveName)
    }

    private func initialFrame() -> NSRect {
        let size = Self.defaultSize
        guard let screenFrame = NSScreen.main?.visibleFrame ?? NSScreen.screens.first?.visibleFrame else {
            return NSRect(origin: .zero, size: size)
        }

        let origin = NSPoint(
            x: screenFrame.midX - size.width / 2,
            y: screenFrame.midY - size.height / 2
        )
        return NSRect(origin: origin, size: size)
    }

    private func centerPanel() {
        panel.setFrame(initialFrame(), display: true)
    }
}
