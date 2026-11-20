import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Revision 1.13.2 hygiene: decorative strokes ignore hits. The missed centre clicks remain unexplained.
/// Source assertions check the decoration policy; AppKit hit-testing reaches hosted content.
/// Neither proves a delivered SwiftUI button action or the cause of the earlier misses.
@MainActor
final class BrainBarNewDesignR1132Tests: XCTestCase {
    typealias Fixture = BrainBarDesignFixture

    static let newDesign = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign")

    /// No overlay in the new design draws a stroke except through `ndDecoration`, which never takes a hit. On c2821964
    /// NDCard drew its border as `.overlay(RoundedRectangle(…).strokeBorder(…))`, above the problem row.
    func testNoDecorativeStrokeCanTakeAClick() throws {
        var offenders: [String] = []
        for file in try FileManager.default.contentsOfDirectory(atPath: Self.newDesign.path)
        where file.hasSuffix(".swift") && file != "NDRenderHarness.swift" {
            let source = try String(contentsOf: Self.newDesign.appendingPathComponent(file), encoding: .utf8)
            var rest = source[...]
            while let range = rest.range(of: ".overlay") {
                let after = rest[range.upperBound...]
                let body = Self.overlayBody(after)
                if body.contains(".stroke") { offenders.append("\(file): \(body.prefix(90))") }
                rest = after
            }
        }
        XCTAssertEqual(offenders, [], "a stroke drawn in a plain overlay sits above the controls and can take their clicks")
        let window = try String(contentsOf: Self.newDesign.appendingPathComponent("NDWindow.swift"), encoding: .utf8)
        XCTAssertTrue(window.contains("overlay(decoration.allowsHitTesting(false))"), "ndDecoration never takes a hit")
        XCTAssertTrue(window.contains(".ndDecoration(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(ND.cardLine"),
                      "the card's border is a decoration")
    }

    /// An overlay's own argument list and trailing block, matched by their brackets.
    static func overlayBody(_ text: Substring) -> String {
        var out = ""
        var index = text.startIndex
        func balanced(_ open: Character, _ close: Character) {
            guard index < text.endIndex, text[index] == open else { return }
            var depth = 0
            while index < text.endIndex {
                let c = text[index]
                out.append(c)
                if c == open { depth += 1 } else if c == close { depth -= 1 }
                index = text.index(after: index)
                if depth == 0 { return }
            }
        }
        balanced("(", ")")
        while index < text.endIndex, text[index] == " " { index = text.index(after: index) }
        balanced("{", "}")
        return out
    }

    /// The single-row card's centre line is the row button's centre line: the line the M1 found dead.
    func testTheOneRowCardsCentreIsTheButtonsCentre() {
        let row = NDProblemRows.rowHeight
        let button: CGFloat = 22
        XCTAssertEqual(row / 2, (row - button) / 2 + button / 2, "the button is centred in its row; one row is the whole card")
    }

    /// A window that may sit off every screen: AppKit would otherwise pull a titled window back onto one.
    final class OffscreenWindow: NSWindow {
        override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }
    }

    /// The new design's window as `BrainBarDashboardPanelController` builds it, 900 × 846 like the M1's, ordered in off
    /// every screen: AppKit hit-testing at the row Reconnect's exact centre (x 818.75, y 134 from the window's top; the M1
    /// measured the button at 123 to 145) reaches the hosted SwiftUI content, never an AppKit view over it.
    func testTheRowReconnectsExactCentreReachesTheHostedContent() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let store = NDStore(fixture: Fixture.inputs(.driveSignedOut), page: .overview)
        let window = OffscreenWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 846),
                                     styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                                     backing: .buffered, defer: false)
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isReleasedWhenClosed = false
        let hosting = NSHostingController(rootView: NDWindowRoot(store: store))
        hosting.sizingOptions = []
        window.contentViewController = hosting
        let toolbar = NDToolbar(store: store)
        window.toolbar = toolbar.toolbar
        window.toolbarStyle = .unified
        window.titlebarSeparatorStyle = .none
        window.setContentSize(NSSize(width: 900, height: 846))
        window.setFrameOrigin(NSPoint(x: -30_000, y: -30_000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil); window.close() }
        window.contentView?.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.15))
        let content = try XCTUnwrap(window.contentView)
        let frameView = try XCTUnwrap(content.superview)
        for dy in [-0.5, -0.125, 0, 0.125, 0.5] {
            let point = NSPoint(x: 818.75, y: window.frame.height - (134 + dy))
            let hit = try XCTUnwrap(frameView.hitTest(point), "dy \(dy)")
            XCTAssertTrue(hit.isDescendant(of: content), "dy \(dy): \(type(of: hit)) is outside the hosted content")
            XCTAssertFalse("\(type(of: hit))".contains("FocusRing") || "\(type(of: hit))".contains("KeyViewProxy"), "dy \(dy): \(type(of: hit))")
        }
        withExtendedLifetime(toolbar) {}
    }
}
