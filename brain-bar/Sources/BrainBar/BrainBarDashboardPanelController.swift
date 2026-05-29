import AppKit
import SwiftUI

final class BrainBarDashboardPanel: NSPanel {
    var onEscape: (() -> Void)?
    var onResignKey: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func cancelOperation(_ sender: Any?) {
        onEscape?()
    }

    override func resignKey() {
        super.resignKey()
        onResignKey?()
    }
}

@MainActor
final class BrainBarDashboardPanelController {
    static let autosaveName = BrainBarWindowFrameAutosave.dashboardPanelName
    static let defaultSize = NSSize(width: BrainBarWindowPlacement.defaultSize.width, height: BrainBarWindowPlacement.defaultSize.height)
    static let minSize = NSSize(width: BrainBarWindowPlacement.minimumSize.width, height: BrainBarWindowPlacement.minimumSize.height)

    let panelForTesting: BrainBarDashboardPanel
    var needsInitialPositioningForTesting: Bool { needsInitialPositioning }

    private let panel: BrainBarDashboardPanel
    private let runtime: BrainBarRuntime
    private var needsInitialPositioning: Bool

    init(runtime: BrainBarRuntime) {
        self.runtime = runtime
        needsInitialPositioning = Self.needsInitialPositioning()
        panel = BrainBarDashboardPanel(
            contentRect: NSRect(origin: .zero, size: Self.defaultSize),
            styleMask: [.titled, .resizable, .closable, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panelForTesting = panel
        configurePanel()
    }

    static func needsInitialPositioning(defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: BrainBarWindowFrameAutosave.dashboardPanelDefaultsKey) == nil
    }

    func toggle(anchoredTo anchorView: NSView? = nil) {
        if panel.isVisible {
            dismiss()
        } else {
            show(anchoredTo: anchorView)
        }
    }

    func show(anchoredTo anchorView: NSView? = nil) {
        if !panel.isVisible, needsInitialPositioning {
            centerPanel()
            needsInitialPositioning = false
        }
        positionPanel(anchoredTo: anchorView)
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
        panel.isMovable = false
        panel.isMovableByWindowBackground = false
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.minSize = Self.minSize
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = true
        panel.backgroundColor = .windowBackgroundColor
        panel.isOpaque = true
        panel.hasShadow = true
        panel.onEscape = { [weak self] in
            self?.dismiss()
        }
        panel.onResignKey = { [weak self] in
            self?.dismiss()
        }
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        )
        hostingController.view.autoresizingMask = [.width, .height]
        panel.contentViewController = hostingController
        panel.setFrameAutosaveName(Self.autosaveName)
        if needsInitialPositioning {
            panel.setFrame(initialFrame(), display: false)
        }
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

    private func positionPanel(anchoredTo anchorView: NSView?) {
        guard let statusItemFrame = anchorView.flatMap(Self.statusItemFrame) else { return }
        let anchoredFrame = Self.anchoredFrameBelowStatusItem(
            currentFrame: panel.frame,
            statusItemFrame: statusItemFrame
        )
        panel.setFrame(anchoredFrame, display: true)
    }

    private static func statusItemFrame(anchorView: NSView) -> NSRect? {
        guard let window = anchorView.window else { return nil }
        let frameInWindow = anchorView.convert(anchorView.bounds, to: nil)
        return window.convertToScreen(frameInWindow)
    }

    static func anchoredFrameBelowStatusItem(
        currentFrame: NSRect,
        statusItemFrame: NSRect,
        visibleScreenFrames: [NSRect] = NSScreen.screens.map(\.visibleFrame),
        gap: CGFloat = BrainBarWindowPlacement.menuBarIconGap
    ) -> NSRect {
        let visibleFrame = visibleScreenFrames.first(where: { $0.intersects(statusItemFrame) })
            ?? visibleScreenFrames.first
            ?? currentFrame
        let targetX = statusItemFrame.midX - currentFrame.width / 2
        let targetY = statusItemFrame.minY - gap - currentFrame.height
        let maxOriginX = max(visibleFrame.minX, visibleFrame.maxX - currentFrame.width)
        let maxOriginY = visibleFrame.maxY - gap - currentFrame.height

        return NSRect(
            x: min(max(targetX, visibleFrame.minX), maxOriginX),
            y: min(max(targetY, visibleFrame.minY), maxOriginY),
            width: currentFrame.width,
            height: currentFrame.height
        )
    }
}
