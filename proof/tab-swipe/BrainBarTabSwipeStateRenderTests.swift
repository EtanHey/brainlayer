import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// State table for #1870 (owner rule 24 Nov, "no state table, no merge"): the tab control in every state it has, at the
/// window's real width and each page's real height, dark and light. Offscreen hosting views only, never a window.
/// `BRAINBAR_SWIPE_STATE_DIR=<dir>` writes the pictures; the assertions run either way.
@MainActor
final class BrainBarTabSwipeStateRenderTests: XCTestCase {
    private var output: URL?

    override func setUp() async throws {
        output = ProcessInfo.processInfo.environment["BRAINBAR_SWIPE_STATE_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        if let output { try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
        NDMarkView.captureClock = Date(timeIntervalSinceReferenceDate: 0.3)
    }

    override func tearDown() async throws { NDMarkView.captureClock = nil }

    /// A click on a tab and a swipe to the same page draw the same window: same selected tab, same page, same height.
    func testAClickAndASwipeToTheSamePageDrawTheSameWindow() throws {
        let cases: [(from: NDPage, dx: CGFloat, to: NDPage)] = [(.overview, -120, .backups), (.backups, -120, .jobs), (.backups, 120, .overview)]
        for (suffix, appearance) in NDRenderHarness.appearances {
            for c in cases {
                let clicked = Rig(fixture: .calm, page: c.from, appearance: appearance)
                let control = try XCTUnwrap(clicked.control)
                control.selectedSegment = try XCTUnwrap(NDPage.allCases.firstIndex(of: c.to))
                XCTAssertTrue(NSApp.sendAction(try XCTUnwrap(control.action), to: control.target, from: control), "the tab's own action")
                let byClick = try clicked.draw()

                let swiped = Rig(fixture: .calm, page: c.from, appearance: appearance)
                var swipe = NDPageSwipe()
                for event in [NDPageSwipe.Event(phase: .began)] + Array(repeating: NDPageSwipe.Event(phase: .changed, dx: c.dx / 10), count: 10)
                    + [NDPageSwipe.Event(phase: .ended)] {
                    swiped.store.swipePage(swipe.route(event).step)
                }
                let bySwipe = try swiped.draw()

                let name = "\(c.to.rawValue)-from-\(c.from.rawValue)"
                XCTAssertEqual(clicked.store.page, c.to, name)
                XCTAssertEqual(swiped.store.page, c.to, name)
                XCTAssertEqual(clicked.control?.selectedSegment, swiped.control?.selectedSegment, "\(name): the same selected tab")
                XCTAssertEqual(byClick.size, bySwipe.size, "\(name): the same height")
                let differing = Self.differingFraction(byClick, bySwipe)
                XCTAssertLessThan(differing, 0.001, "\(name) \(suffix): the click and the swipe draw the same window (\(differing))")
                try write(byClick, "state-\(name)-by-click-\(suffix)")
                try write(bySwipe, "state-\(name)-by-swipe-\(suffix)")
            }
        }
    }

    /// Keyboard focus on the tab control (Full Keyboard Access, Tab key). An offscreen view cannot hold first responder, so
    /// the ring is drawn the way AppKit draws it: the control's own `drawFocusRingMask` under `NSSetFocusRingStyle`.
    func testKeyboardFocusOnTheTabControl() throws {
        for (suffix, appearance) in NDRenderHarness.appearances {
            for page in NDPage.allCases {
                let rig = Rig(fixture: .calm, page: page, appearance: appearance)
                let resting = try rig.draw()
                let control = try XCTUnwrap(rig.control)
                let focused = try rig.draw(focusRingOn: control)
                XCTAssertGreaterThan(Self.differingFraction(resting, focused), 0.0005, "\(page) \(suffix): the ring is drawn")
                try write(focused, "state-focus-\(page.rawValue)-\(suffix)")
            }
        }
    }

    /// Two faults: Backups and Jobs carry their count pills; each page selected, at its real height.
    func testEachPageSelectedWithCountPills() throws {
        for (suffix, appearance) in NDRenderHarness.appearances {
            for page in NDPage.allCases {
                let rig = Rig(fixture: .twoFaults, page: page, appearance: appearance)
                let rep = try rig.draw()
                XCTAssertEqual(rig.control?.selectedSegment, NDPage.allCases.firstIndex(of: page))
                XCTAssertGreaterThan(rig.store.r1.backupsBadge + rig.store.r1.jobsBadge, 0, "the fixture carries counts")
                try write(rep, "state-two-faults-\(page.rawValue)-\(suffix)")
            }
        }
    }

    // MARK: Helpers

    @MainActor private final class Rig {
        let store: NDStore
        let host: NSHostingView<NDWindowRoot>
        init(fixture: BrainBarDesignFixture.Name, page: NDPage, appearance: NSAppearance.Name) {
            store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: page)
            host = NSHostingView(rootView: NDWindowRoot(store: store, drawsTrafficLights: true))
            host.appearance = NSAppearance(named: appearance)
            layOut()
        }
        var control: NSSegmentedControl? { Self.find(in: host) }
        func layOut() {
            let size = NSSize(width: NDGrid.windowWidth, height: NDPageHeight.window(store, page: store.page, screen: nil))
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.3))
            host.layoutSubtreeIfNeeded()
        }
        func draw(focusRingOn control: NSView? = nil) throws -> NSBitmapImageRep {
            layOut()
            let rep = try NDRenderHarness.bitmap(host.bounds.size)
            host.cacheDisplay(in: host.bounds, to: rep)
            guard let control, let context = NSGraphicsContext(bitmapImageRep: rep) else { return rep }
            NSGraphicsContext.saveGraphicsState()
            defer { NSGraphicsContext.restoreGraphicsState() }
            NSGraphicsContext.current = context
            let frame = control.convert(control.bounds, to: host)
            let y = host.isFlipped ? host.bounds.height - frame.maxY : frame.minY
            let transform = NSAffineTransform()
            transform.translateX(by: frame.minX, yBy: y)
            if control.isFlipped {
                transform.translateX(by: 0, yBy: frame.height)
                transform.scaleX(by: 1, yBy: -1)
            }
            transform.concat()
            (host.appearance ?? NSAppearance.currentDrawing()).performAsCurrentDrawingAppearance {
                NSFocusRingPlacement.only.set()
                control.drawFocusRingMask()
            }
            return rep
        }
        static func find(in view: NSView) -> NSSegmentedControl? {
            if let control = view as? NSSegmentedControl { return control }
            for sub in view.subviews { if let found = find(in: sub) { return found } }
            return nil
        }
    }

    private func write(_ rep: NSBitmapImageRep, _ name: String) throws {
        guard let output else { return }
        try NDRenderHarness.write(rep, name: name, in: output)
    }

    /// The share of pixels whose RGBA bytes differ.
    static func differingFraction(_ a: NSBitmapImageRep, _ b: NSBitmapImageRep) -> Double {
        guard a.pixelsWide == b.pixelsWide, a.pixelsHigh == b.pixelsHigh, let pa = a.bitmapData, let pb = b.bitmapData,
              a.bytesPerRow == b.bytesPerRow else { return 1 }
        var differing = 0
        for row in 0..<a.pixelsHigh {
            for column in 0..<a.pixelsWide {
                let offset = row * a.bytesPerRow + column * 4
                if pa[offset] != pb[offset] || pa[offset + 1] != pb[offset + 1] || pa[offset + 2] != pb[offset + 2] { differing += 1 }
            }
        }
        return Double(differing) / Double(a.pixelsWide * a.pixelsHigh)
    }
}
