import AppKit
import SwiftUI
import XCTest

@testable import BrainBar

final class BrainBarCardShapeVisualTests: XCTestCase {
    @MainActor
    func testOptionalCardShapeFrames() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_L3_SHAPE_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_L3_SHAPE_RENDER_DIR to capture Dashboard states.")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        NSApplication.shared.setActivationPolicy(.prohibited)

        let states: [BrainBarDashboardFixture.OperatorState] = [
            .live, .loadingCards, .stale, .unavailable, .empty, .error,
        ]
        for (widthName, width) in [("compact", CGFloat(760)), ("default", 960), ("wide", 1_280)] {
            for state in states {
                let panelState = BrainBarDashboardPanelState()
                panelState.detailsExpanded = true
                panelState.signalCoverageExpanded = true
                let view = BrainBarDashboardPreview.make(
                    collector: BrainBarDashboardFixture.makeCollector(state),
                    observabilityResult: state == .unavailable ? .unreadable("Fixture unavailable") : (state == .empty ? BrainBarDashboardFixture.emptyObservabilityResult : BrainBarDashboardFixture.readableObservabilityResult),
                    now: BrainBarDashboardFixture.fetchedAt,
                    panelState: panelState
                )
                let host = NSHostingController(rootView: view)
                let window = NSWindow(
                    contentRect: NSRect(x: -6_000, y: -6_000, width: width, height: 1_200),
                    styleMask: [.titled], backing: .buffered, defer: false
                )
                window.contentViewController = host
                window.setFrame(NSRect(x: -6_000, y: -6_000, width: width, height: 1_200), display: false)
                window.orderFront(nil)
                host.view.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.35))
                host.view.layoutSubtreeIfNeeded()
                let bitmap = try XCTUnwrap(host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds))
                host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
                let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                XCTAssertGreaterThan(png.count, 5_000)
                let file = directory.appendingPathComponent("\(widthName)-\(state).png")
                try png.write(to: file, options: .atomic)
                print("[brainbar-l3-shape] \(file.lastPathComponent) \(bitmap.pixelsWide)x\(bitmap.pixelsHigh) \(png.count) bytes")
                window.orderOut(nil)
            }
        }
    }
}
