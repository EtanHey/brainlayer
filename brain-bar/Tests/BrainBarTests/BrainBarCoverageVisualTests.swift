import AppKit
import SwiftUI
import XCTest

@testable import BrainBar

final class BrainBarCoverageVisualTests: XCTestCase {
    @MainActor
    func testOptionalCoverageFixtureRenders() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_L0_HONESTY_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_L0_HONESTY_RENDER_DIR to capture coverage fixtures.")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        NSApplication.shared.setActivationPolicy(.prohibited)

        let cases: [(String, CGFloat, BrainBarDashboardFixture.OperatorState, Bool)] = [
            ("compact", 760, .live, true),
            ("default", 960, .live, true),
            ("wide", 1_280, .live, true),
            ("stale", 960, .stale, true),
            ("details-collapsed", 960, .live, false),
        ]
        for (name, width, state, detailsExpanded) in cases {
            let panelState = BrainBarDashboardPanelState()
            panelState.detailsExpanded = detailsExpanded
            panelState.signalCoverageExpanded = true
            let view = BrainBarDashboardPreview.make(
                collector: BrainBarDashboardFixture.makeCollector(state),
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
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.6))
            host.view.layoutSubtreeIfNeeded()
            let bitmap = try XCTUnwrap(host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds))
            host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
            let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            XCTAssertGreaterThan(png.count, 5_000)
            let url = directory.appendingPathComponent("\(name).png")
            try png.write(to: url, options: .atomic)
            print("[brainbar-l0-honesty] \(name) \(url.path) \(bitmap.pixelsWide)x\(bitmap.pixelsHigh) \(png.count) bytes")
            window.orderOut(nil)
        }
    }
}
