import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Renders the real SparklineChart views to PNGs for visual QA against the Aldante
/// reference frames. Not an assertion test — it produces images in the qa dir.
/// Run with: swift test --filter SparklineSnapshotTests
@MainActor
final class SparklineSnapshotTests: XCTestCase {
    // Opt-in only: set BRAINBAR_SNAPSHOT_DIR to render the QA PNGs. Skipped in CI by default.
    private var outDir: String? { ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] }

    private func save<V: View>(_ view: V, _ name: String, size: CGSize = CGSize(width: 360, height: 200)) {
        guard let outDir else { return }
        let renderer = ImageRenderer(content:
            view
                .frame(width: size.width, height: size.height)
                .padding(18)
                .background(Color(nsColor: NSColor(calibratedRed: 0.07, green: 0.08, blue: 0.10, alpha: 1)))
        )
        renderer.scale = 2
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            XCTFail("render failed for \(name)")
            return
        }
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        let path = "\(outDir)/\(name).png"
        try? png.write(to: URL(fileURLWithPath: path))
        print("SNAPSHOT \(path)")
    }

    func testRenderSparklines() throws {
        try XCTSkipIf(outDir == nil, "Set BRAINBAR_SNAPSHOT_DIR to render QA snapshots")
        let accent = BrainBarDesignTokens.Colors.signalFTS5 // green-ish, like Aldante battery card
        let watcher = BrainBarDesignTokens.Colors.seriesWatcher

        // SPARSE: the [0,0,3,0,0] trap — must show a soft-floor lit band + spike mass, no overshoot.
        let sparse = SparklineChartPresentation(
            label: "Enrichments",
            values: [0, 0, 3, 0, 0, 1, 0, 0],
            primarySeriesLabel: "Enrichments"
        )
        save(SparklineChart(presentation: sparse, accentColor: accent), "sparse-rest")

        // DENSE: continuous-ish signal — full gradient mass + catmull smoothing.
        let dense = SparklineChartPresentation(
            label: "Coverage",
            values: [42, 48, 55, 53, 60, 66, 71, 69, 74, 80],
            primarySeriesLabel: "Coverage"
        )
        save(SparklineChart(presentation: dense, accentColor: accent), "dense-rest")

        // LOW-VALUE tight scale: peak of 3 should fill toward the TOP, not pin at 60%.
        let low = SparklineChartPresentation(
            label: "Writes",
            values: [0, 1, 2, 3, 2, 1, 0, 1],
            primarySeriesLabel: "Writes"
        )
        save(SparklineChart(presentation: low, accentColor: accent), "low-tight")

        // DUAL series: green primary + rose secondary, each its own faint fill.
        let dual = SparklineChartPresentation(
            label: "Writes",
            values: [0, 2, 0, 4, 1, 3, 0, 2],
            secondaryValues: [1, 0, 2, 0, 3, 0, 1, 0],
            primarySeriesLabel: "Agent",
            secondarySeriesLabel: "Watcher"
        )
        save(SparklineChart(presentation: dual, accentColor: accent, secondaryAccentColor: watcher), "dual-rest")

        // COMPACT (menubar) regression: must stay a clean stroke, no fill/chrome.
        save(
            SparklineChart(presentation: sparse, accentColor: accent, compact: true),
            "sparse-compact",
            size: CGSize(width: 44, height: 18)
        )

        // HOVER over a mid bucket: crosshair + on-curve dot + tooltip + revealed axes.
        save(
            SparklineChart(presentation: dense, accentColor: accent,
                           previewHoveredBucket: 5, previewHoverX: 190),
            "dense-hover-mid"
        )

        // HOVER DIRECTLY ON THE TOP SPIKE (the reviewer-flagged failure):
        // the card must NOT collapse to the bottom — it should ride above/beside the dot.
        save(
            SparklineChart(presentation: sparse, accentColor: accent,
                           previewHoveredBucket: 2, previewHoverX: 110),
            "sparse-hover-spike"
        )

        // REFERENCE LINE (Aldante signature dashed benchmark).
        save(
            SparklineChart(presentation: low, accentColor: accent, referenceValue: 3),
            "low-reference"
        )

        // FULL NUMBER-FIRST CARD (the defining Aldante trait): caption + hero number,
        // status pill, graph as evidence, single muted caption at the bottom.
        let enrichLane = DashboardFlowLane(
            name: "Enrichments",
            status: .live,
            statusText: "Enrichment is draining backlog steadily.",
            windowLabel: "Last 30m",
            activityWindowMinutes: 30,
            rateText: "0.5/min",
            volumeText: "14 enriched",
            lastEventText: "1m ago",
            values: [0, 0, 3, 0, 0, 1, 0, 2],
            sparklineLabel: "Enrichments",
            latestBucketName: "latest bucket count",
            accentColor: BrainBarDesignTokens.Colors.signalFTS5,
            primarySeriesLabel: "Enrichments",
            secondaryValues: [],
            secondarySeriesLabel: nil,
            secondaryAccentColor: nil,
            tertiaryValues: [],
            tertiarySeriesLabel: nil,
            tertiaryAccentColor: nil
        )
        save(BrainBarFlowLaneCardPreview.make(lane: enrichLane),
             "card-enrichments", size: CGSize(width: 380, height: 300))

        let writesLane = DashboardFlowLane(
            name: "Writes",
            status: .live,
            statusText: "Writes are landing across agent + watcher.",
            windowLabel: "Last 30m",
            activityWindowMinutes: 30,
            rateText: "1.2/min",
            volumeText: "37 writes",
            lastEventText: "just now",
            values: [0, 2, 0, 4, 1, 3, 0, 2],
            sparklineLabel: "Writes",
            latestBucketName: "latest bucket count",
            accentColor: BrainBarDesignTokens.Colors.signalFTS5,
            primarySeriesLabel: "Agent",
            secondaryValues: [1, 0, 2, 0, 3, 0, 1, 0],
            secondarySeriesLabel: "Watcher",
            secondaryAccentColor: BrainBarDesignTokens.Colors.seriesWatcher,
            tertiaryValues: [],
            tertiarySeriesLabel: nil,
            tertiaryAccentColor: nil
        )
        save(BrainBarFlowLaneCardPreview.make(lane: writesLane),
             "card-writes", size: CGSize(width: 380, height: 320))

        // HOVER where ONLY the secondary (rose) series has the spike and primary is
        // zero at that bucket: the dot/crosshair/tooltip must ride the ROSE curve, not
        // the green baseline. (Reviewer multi-series anchor fix.) Bucket 4 -> secondary=3.
        let secondaryOnly = SparklineChartPresentation(
            label: "Writes",
            values: [0, 0, 0, 0, 0, 0, 0, 0],
            secondaryValues: [0, 1, 0, 2, 3, 0, 1, 0],
            primarySeriesLabel: "Agent",
            secondarySeriesLabel: "Watcher"
        )
        save(
            SparklineChart(presentation: secondaryOnly, accentColor: accent,
                           secondaryAccentColor: watcher,
                           previewHoveredBucket: 4, previewHoverX: 160),
            "secondary-spike-hover"
        )
    }
}
