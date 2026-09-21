#if DEBUG
import AppKit
import Foundation

@MainActor
enum BrainBarGeometryProbe {
    static func runIfRequested() {
        guard ProcessInfo.processInfo.environment["BRAINBAR_GEOMETRY_PROBE"] == "1" else { return }
        NSApplication.shared.setActivationPolicy(.prohibited)
        exit(run())
    }

    private static func run() -> Int32 {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting
        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.4))
        controller.setDashboardHeightForTesting(400)
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))

        let anchorWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 32, height: 24),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        let anchorView = NSView(frame: NSRect(x: 0, y: 0, width: 32, height: 24))
        anchorWindow.contentView = anchorView
        controller.statusItemButton = anchorView

        let visibleFrame = NSScreen.main?.visibleFrame
        panel.setFrameOrigin(NSPoint(
            x: (visibleFrame?.minX ?? 0) + 80,
            y: (visibleFrame?.maxY ?? 1_000) - 80 - panel.frame.height
        ))

        func topLeft() -> NSPoint { NSPoint(x: panel.frame.minX, y: panel.frame.maxY) }
        func samePoint(_ lhs: NSPoint, _ rhs: NSPoint) -> Bool {
            abs(lhs.x - rhs.x) <= 0.5 && abs(lhs.y - rhs.y) <= 0.5
        }

        let beforeExpand = topLeft()
        controller.resetGeometryMetricsForTesting()
        for height in stride(from: CGFloat(420), through: 780, by: 20) {
            controller.setDashboardHeightForTesting(height)
            RunLoop.main.run(until: Date().addingTimeInterval(1.0 / 60.0))
        }
        controller.contentSizeDidApplyForTesting = { [weak controller] in
            controller?.windowDidResize(Notification(name: NSWindow.didResizeNotification, object: panel))
        }
        controller.completeDisclosureTransitionForTesting(expanded: true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        controller.contentSizeDidApplyForTesting = nil
        let expandOK = controller.contentSizeApplicationCountForTesting == 1
            && controller.reentrantFitAttemptCountForTesting == 0
            && samePoint(beforeExpand, topLeft())
        print("GEOMETRY_PROBE expand count=\(controller.contentSizeApplicationCountForTesting) reentry=\(controller.reentrantFitAttemptCountForTesting) topLeftPreserved=\(samePoint(beforeExpand, topLeft()))")

        let beforeCollapse = topLeft()
        controller.resetGeometryMetricsForTesting()
        for height in stride(from: CGFloat(760), through: 360, by: -20) {
            controller.setDashboardHeightForTesting(height)
            RunLoop.main.run(until: Date().addingTimeInterval(1.0 / 60.0))
        }
        controller.contentSizeDidApplyForTesting = { [weak controller] in
            controller?.windowDidResize(Notification(name: NSWindow.didResizeNotification, object: panel))
        }
        controller.completeDisclosureTransitionForTesting(expanded: false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        controller.contentSizeDidApplyForTesting = nil
        let collapseOK = controller.contentSizeApplicationCountForTesting == 1
            && controller.reentrantFitAttemptCountForTesting == 0
            && samePoint(beforeCollapse, topLeft())
        print("GEOMETRY_PROBE collapse count=\(controller.contentSizeApplicationCountForTesting) reentry=\(controller.reentrantFitAttemptCountForTesting) topLeftPreserved=\(samePoint(beforeCollapse, topLeft()))")

        let collapsed = BrainBarDashboardDisclosureState(
            detailsExpanded: false,
            signalCoverageExpanded: false
        )
        let expanded = BrainBarDashboardDisclosureState(
            detailsExpanded: true,
            signalCoverageExpanded: false
        )
        let scrollOK = expanded.opensDisclosure(comparedTo: collapsed)
            && !collapsed.opensDisclosure(comparedTo: expanded)
        print("GEOMETRY_PROBE scroll expandReset=\(expanded.opensDisclosure(comparedTo: collapsed)) collapseReset=\(collapsed.opensDisclosure(comparedTo: expanded))")

        return expandOK && collapseOK && scrollOK ? 0 : 1
    }
}
#endif
