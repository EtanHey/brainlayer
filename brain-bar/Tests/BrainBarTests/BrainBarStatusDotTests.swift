import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Etan row E4 (2026-10-01): the green dot beside "Google Drive" was not vertically centred.
/// Measured in pixels: the dot's centre must sit within 0.5 pt of the label's cap-height centre
/// (the bare Circle it replaces measured 1 pt low).
@MainActor
final class BrainBarStatusDotTests: XCTestCase {
    private struct Measured { let dotCenter: CGFloat, capCenter: CGFloat }

    /// Renders `dot` beside "HHHH" (cap letters only, so the ink spans exactly the cap height)
    /// and returns both vertical centres in points.
    private func measure(font: Font, @ViewBuilder dot: () -> some View) throws -> Measured {
        let row = HStack(alignment: .firstTextBaseline, spacing: 8) {
            dot()
            Text("HHHH").font(font).foregroundStyle(Color.white)
        }
        .padding(12)
        .background(Color.black)
        .environment(\.colorScheme, .dark)
        let host = NSHostingView(rootView: row)
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        host.layoutSubtreeIfNeeded()
        let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        let scale = CGFloat(rep.pixelsHigh) / host.bounds.height
        var red: [Int] = [], white: [Int] = []
        for y in 0 ..< rep.pixelsHigh {
            for x in 0 ..< rep.pixelsWide {
                guard let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                if c.redComponent > 0.6, c.greenComponent < 0.3, c.blueComponent < 0.3 { red.append(y) }
                if c.redComponent > 0.6, c.greenComponent > 0.6, c.blueComponent > 0.6 { white.append(y) }
            }
        }
        let dotRows = [try XCTUnwrap(red.min()), try XCTUnwrap(red.max())]
        let capRows = [try XCTUnwrap(white.min()), try XCTUnwrap(white.max())]
        return Measured(
            dotCenter: CGFloat(dotRows[0] + dotRows[1]) / 2 / scale,
            capCenter: CGFloat(capRows[0] + capRows[1]) / 2 / scale
        )
    }

    func test_the_dot_is_centred_on_its_label_at_every_size_the_app_uses() throws {
        let cases: [(Font, CGFloat)] = [
            (.system(size: 14, weight: .semibold), 8),  // Google Drive card
            (.system(size: 13, weight: .semibold), 9),  // Dashboard status strip
            (.system(size: 13), 7),                     // Dashboard backup tile
            (.system(size: 12, weight: .medium), 5),    // Dashboard attention items
            (.system(size: 11, weight: .medium), 7),    // Settings rows
        ]
        for (font, size) in cases {
            let measured = try measure(font: font) {
                BrainBarStatusDot(color: Color(red: 1, green: 0, blue: 0), size: size).font(font)
            }
            XCTAssertEqual(measured.dotCenter, measured.capCenter, accuracy: 0.5,
                           "dot \(size) pt beside \(font): dot centre \(measured.dotCenter), cap centre \(measured.capCenter)")
        }
    }

    /// The test can tell: the bare Circle the Drive card used before E4 is measurably off-centre.
    func test_a_bare_circle_on_the_baseline_is_measurably_off_centre() throws {
        let font = Font.system(size: 14, weight: .semibold)
        let measured = try measure(font: font) {
            Circle().fill(Color(red: 1, green: 0, blue: 0)).frame(width: 8, height: 8)
        }
        XCTAssertGreaterThan(abs(measured.dotCenter - measured.capCenter), 0.5)
    }
}
