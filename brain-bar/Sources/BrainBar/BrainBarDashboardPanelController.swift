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

    let popoverForTesting: NSPopover
    let contentViewControllerForTesting: NSViewController
    var isShownForTesting: Bool { popover.isShown }

    private let popover: NSPopover

    init(runtime: BrainBarRuntime) {
        let hostingController = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false)
                .frame(minWidth: Self.minSize.width, minHeight: Self.minSize.height)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        )
        hostingController.view.frame = NSRect(origin: .zero, size: Self.defaultSize)
        hostingController.view.autoresizingMask = [.width, .height]

        popover = NSPopover()
        popoverForTesting = popover
        contentViewControllerForTesting = hostingController

        configurePopover(contentViewController: hostingController)
    }

    func toggle(anchoredTo anchorView: NSView? = nil) {
        if popover.isShown {
            dismiss()
        } else {
            show(anchoredTo: anchorView)
        }
    }

    func show(anchoredTo anchorView: NSView? = nil) {
        guard let anchorView else { return }

        popover.show(
            relativeTo: anchorView.bounds,
            of: anchorView,
            preferredEdge: .minY
        )
    }

    func dismiss() {
        popover.performClose(nil)
    }

    private func configurePopover(contentViewController: NSViewController) {
        popover.behavior = .transient
        popover.animates = true
        popover.contentSize = Self.defaultSize
        popover.contentViewController = contentViewController
    }
}
