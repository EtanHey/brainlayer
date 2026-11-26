import AppKit
import XCTest
@testable import BrainBar

/// Headless: an in-memory button, never a system status item, menu tracking or window.
@MainActor final class BrainBarBrainIconTests: XCTestCase {
    func testFixtureIconsAndHeadlessRenders() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let output = ProcessInfo.processInfo.environment["BRAINBAR_ICON_RENDER_DIR"].map { URL(fileURLWithPath: $0) }
        if let output { try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
        let fixtures = try BrainBarMenuPresentationTests.loadCases().filter { ["S1", "S3", "S8"].contains($0["id"] as? String ?? "") }
        XCTAssertEqual(fixtures.count, 3)
        for fixture in fixtures {
            let id = fixture["id"] as! String
            let capture = try BrainBarMenuCases.capture(fixture["inputs"] as! [String: Any])
            let p = BrainBarMenuPresenter.present(capture)
            XCTAssertEqual(p.icon, BrainBarMenuPresentationTests.expected(fixture).icon)
            let image = BrainBarBrainIcon.image(for: p)
            XCTAssertEqual(image.size, NSSize(width: 22, height: 22))
            XCTAssertEqual(image.isTemplate, p.icon == .resting)
            XCTAssertEqual(image.accessibilityDescription, "BrainBar — " + p.headline)
            for dark in [false, true] {
                let pixels = try bitmap(image, tint: image.isTemplate ? (dark ? .white : .black) : nil)
                var red = 0, white = 0, ink = 0
                var firstInkRow = pixels.pixelsHigh, lastInkRow = -1
                for y in 0..<pixels.pixelsHigh {
                    for x in 0..<pixels.pixelsWide {
                        let c = try XCTUnwrap(pixels.colorAt(x: x, y: y)?.usingColorSpace(.sRGB))
                        if c.alphaComponent > 0 {
                            firstInkRow = min(firstInkRow, y); lastInkRow = max(lastInkRow, y)
                        }
                        if c.alphaComponent > 0.99 {
                            ink += 1
                            if isAlertRed(c) { red += 1 }
                            if c.redComponent > 0.99 && c.greenComponent > 0.99 && c.blueComponent > 0.99 { white += 1 }
                        }
                    }
                }
                // Centred 20-point glyph: ink rows 3...40 within the 44-pixel slot.
                XCTAssertEqual(firstInkRow, 3); XCTAssertEqual(lastInkRow, 40)
                XCTAssertGreaterThan(ink, 50)
                if p.icon == .marked { XCTAssertGreaterThan(red, 100); XCTAssertGreaterThan(white, 5) }
                else { XCTAssertEqual(red, 0); if dark { XCTAssertEqual(white, ink) } }
                if let output {
                    try XCTUnwrap(pixels.representation(using: .png, properties: [:]))
                        .write(to: output.appendingPathComponent(id + (dark ? "-icon-dark.png" : "-icon-light.png")))
                }
            }
        }
    }

    func testRedToleranceRejectsDifferentRed() {
        XCTAssertTrue(isAlertRed(NSColor(srgbRed: 1, green: 59.0 / 255.0, blue: 48.0 / 255.0, alpha: 1)))
        XCTAssertFalse(isAlertRed(NSColor(srgbRed: 1, green: 0, blue: 0, alpha: 1)))
    }

    private func isAlertRed(_ color: NSColor) -> Bool {
        let tolerance: CGFloat = 6.0 / 255.0
        return abs(color.redComponent - 1) <= tolerance
            && abs(color.greenComponent - 59.0 / 255.0) <= tolerance
            && abs(color.blueComponent - 48.0 / 255.0) <= tolerance
    }

    private func bitmap(_ image: NSImage, tint: NSColor? = nil) throws -> NSBitmapImageRep {
        let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(image.size.width * 2), pixelsHigh: Int(image.size.height * 2),
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0))
        rep.size = image.size
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = try XCTUnwrap(NSGraphicsContext(bitmapImageRep: rep))
        let rect = NSRect(origin: .zero, size: image.size)
        image.draw(in: rect)
        if let tint { tint.setFill(); rect.fill(using: .sourceIn) }
        NSGraphicsContext.current?.flushGraphics()
        return rep
    }
}
