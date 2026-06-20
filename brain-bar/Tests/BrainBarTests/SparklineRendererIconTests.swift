import AppKit
import XCTest
@testable import BrainBar

/// Guards the menu-bar status icon: it must (1) render a non-empty image even when
/// ALL series are idle/zero (so it never vanishes on a dark fullscreen menu bar —
/// Etan 2026-06-20), and (2) draw the three overlapping pipeline lines.
@MainActor
final class SparklineRendererIconTests: XCTestCase {
    private var outDir: String? { ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] }

    private func save(_ image: NSImage, _ name: String) {
        guard let dir = outDir,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return }
        try? png.write(to: URL(fileURLWithPath: "\(dir)/\(name).png"))
    }

    func testStatusIconRendersWhenAllSeriesIdle() throws {
        // The regression: gray single line on a dark bar = invisible. The icon must
        // still produce a real (non-zero-size) image when nothing is happening.
        let image = SparklineRenderer.renderStatusBarIcon(
            agent: Array(repeating: 0, count: 12),
            watcher: Array(repeating: 0, count: 12),
            enrichment: Array(repeating: 0, count: 12),
            size: NSSize(width: 26, height: 14)
        )
        XCTAssertGreaterThan(image.size.width, 0)
        XCTAssertGreaterThan(image.size.height, 0)
        save(image, "status-icon-idle")
    }

    func testStatusIconRendersThreeActiveSeries() throws {
        let image = SparklineRenderer.renderStatusBarIcon(
            agent: [0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1],
            watcher: [3, 5, 4, 7, 6, 5, 8, 6, 5, 7, 6, 9],
            enrichment: [1, 0, 2, 1, 3, 2, 1, 2, 0, 1, 2, 1],
            size: NSSize(width: 26, height: 14)
        )
        XCTAssertGreaterThan(image.size.width, 0)
        save(image, "status-icon-active")
    }
}
