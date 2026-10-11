#if DEBUG
import AppKit
import CryptoKit
import SwiftUI

/// Pixel comparator for the production status-item image factory. The reference
/// constructs only the two retained series, independently of that factory.
@MainActor
enum BrainBarMenuIconProof {
    static let names = ["status-icon-idle", "status-icon-active", "status-icon-badged"]

    static func measure(_ name: String, directory: URL? = nil) throws -> [String: Any] {
        let active = name == "status-icon-active"
        let badge = name == "status-icon-badged"
        let agent = active ? [0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1] : Array(repeating: 0, count: 12)
        let watcher = active ? [3, 5, 4, 7, 6, 5, 8, 6, 5, 7, 6, 9] : Array(repeating: 0, count: 12)
        let stats = DashboardStats(
            chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 274847,
            enrichmentPercent: 50, enrichmentRatePerMinute: 1, databaseSizeBytes: 0,
            recentActivityBuckets: agent, recentAgentWriteBuckets: agent,
            recentWatcherWriteBuckets: watcher,
            recentEnrichmentBuckets: active ? [9, 8, 10, 4, 7, 3, 10, 5, 9, 4, 8, 10] : Array(repeating: 0, count: 12)
        )
        let actual = BrainBarStatusPopoverController.statusIconImage(stats: stats, badgeOn: badge)
        let reference = ImageRenderer(content: ZStack(alignment: .topTrailing) {
            MenuBarSparklineIcon(series: [
                .init(values: agent, color: Color(nsColor: BrainBarDesignTokens.Colors.seriesAgent)),
                .init(values: watcher, color: Color(nsColor: BrainBarDesignTokens.Colors.seriesWatcher)),
            ])
            if badge {
                Circle().fill(Color.red).overlay(Circle().stroke(Color.white, lineWidth: 0.75))
                    .frame(width: 6, height: 6)
            }
        }.frame(width: 26, height: 14))
        reference.scale = NSScreen.main?.backingScaleFactor ?? 2
        guard let expected = reference.nsImage else { throw Failure.missingImage }
        let (actualBytes, nontransparent, red) = try pixels(actual)
        let (expectedBytes, _, _) = try pixels(expected)
        guard let tiff = actual.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else { throw Failure.missingImage }
        if let directory { try png.write(to: directory.appendingPathComponent(name + ".png")) }
        return [
            "name": name, "png": name + ".png", "text": "",
            "width": bitmap.pixelsWide, "height": bitmap.pixelsHigh,
            "pixel_check": [
                "series": ["Agent", "Watcher"], "matches_reference": actualBytes == expectedBytes,
                "actual_rgba_sha256": digest(actualBytes), "reference_rgba_sha256": digest(expectedBytes),
                "nontransparent_pixels": nontransparent, "red_pixels": red,
            ],
        ]
    }

    private enum Failure: Error { case missingImage }
    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private static func pixels(_ image: NSImage) throws -> (Data, Int, Int) {
        guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff) else {
            throw Failure.missingImage
        }
        var data = Data(), nontransparent = 0, red = 0
        for y in 0..<bitmap.pixelsHigh {
            for x in 0..<bitmap.pixelsWide {
                guard let c = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { throw Failure.missingImage }
                for value in [c.redComponent, c.greenComponent, c.blueComponent, c.alphaComponent] {
                    data.append(UInt8(max(0, min(255, (value * 255).rounded()))))
                }
                if c.alphaComponent > 0.1 { nontransparent += 1 }
                if c.redComponent > 0.7 && c.greenComponent < 0.4 && c.blueComponent < 0.4 && c.alphaComponent > 0.5 { red += 1 }
            }
        }
        return (data, nontransparent, red)
    }
}
#endif
