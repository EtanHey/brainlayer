import AppKit
import SwiftUI
import Vision
import XCTest
@testable import BrainBar

@MainActor
final class SparklineTooltipOpacityTests: XCTestCase {
    private let values = [42, 48, 55, 53, 60, 66, 71, 69, 74, 80]

    private func renderStack(height: CGFloat, hovered: PipelineSeries, bucket: Int, siblings: Bool,
                             scheme: ColorScheme) throws -> NSBitmapImageRep {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 900, height: 900))
        XCTAssertTrue(layout.compactCards)
        let width = BrainBarIngestBandLayout.chartSizes(containerWidth: 900)[0].width
        let lane = { (series: PipelineSeries) in
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(series == .allCommits ? "All chunks · chunk rows" :
                         series == .agentStores ? "Agent · chunk rows" : "Watcher · unique chunk IDs")
                        .font(.system(size: 11))
                    Spacer()
                    Text("66").font(.system(size: 13, weight: .semibold))
                }.frame(height: 16).opacity(siblings || series == hovered ? 1 : 0)
                SparklineChart(presentation: SparklineChartPresentation(
                    label: "Ingest", values: self.values, primarySeriesLabel: "Watcher",
                    fetchedAt: Date(timeIntervalSince1970: 1782860400),
                    metricDisclosure: "Window: Last 30m · Count: hovered value below · Unit: chunks · Clock: ingest time",
                    showsRestingAxes: true
                ), accentColor: .systemGreen,
                   previewHoveredBucket: series == hovered ? bucket : nil, previewHoverX: 0)
                    .frame(height: height)
            }
        }
        let view = brainBarIngestLaneStack(compact: layout.compactCards,
            allCommits: { lane(.allCommits) }, agentStores: { lane(.agentStores) },
            jsonlWatcher: { lane(.jsonlWatcher) })
            .frame(width: width, height: height * 3 + 110, alignment: .topLeading)
            .padding(.horizontal, (900 - width) / 2)
            .background(Color.brainBarBackgroundBase)
            .environment(\.colorScheme, scheme)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        return try XCTUnwrap(try XCTUnwrap(renderer.nsImage).tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
    }

    func testRealIngestLaneStackHasNoSiblingPixelsInsideHoveredCard() throws {
        let width = BrainBarIngestBandLayout.chartSizes(containerWidth: 900)[0].width
        for height in [BrainBarIngestBandLayout.plotHeight, CGFloat(112)] {
            for hovered in [PipelineSeries.allCommits, .agentStores] {
                for bucket in [0, 9] {
                    for scheme in [ColorScheme.light, .dark] {
                        let with = try renderStack(height: height, hovered: hovered, bucket: bucket, siblings: true, scheme: scheme)
                        let without = try renderStack(height: height, hovered: hovered, bucket: bucket, siblings: false, scheme: scheme)
                        let chartSize = CGSize(width: width, height: height - 14)
                        let cardSize = SparklineTooltipPlacement.tooltipSize(in: chartSize, showsMetricDisclosure: true)
                        let anchorY = 10 + (chartSize.height - 20) * (1 - CGFloat(values[bucket]) / 90)
                        let center = SparklineTooltipPlacement.position(near: CGPoint(x: 0, y: anchorY),
                            anchorY: anchorY, hoveredX: 40 + CGFloat(bucket) * (width - 50) / 9,
                            containerBounds: chartSize, tooltipSize: cardSize)
                        let row = hovered == .allCommits ? 0 : 1
                        let card = CGRect(x: (900 - width) / 2 + center.x - cardSize.width / 2,
                            y: CGFloat(row) * (height + 36) + 22 + center.y - cardSize.height / 2,
                            width: cardSize.width, height: cardSize.height)
                        // Prove a real card occupies the sampled bounds before comparing siblings.
                        let fill = BrainBarDesignTokens.Colors.backgroundRaised
                        let padding = try XCTUnwrap(without.colorAt(x: Int((card.minX + 6) * 2), y: Int(card.midY * 2)))
                        XCTAssertEqual(padding.redComponent, fill.redComponent, accuracy: 0.01)
                        XCTAssertEqual(padding.greenComponent, fill.greenComponent, accuracy: 0.01)
                        XCTAssertEqual(padding.blueComponent, fill.blueComponent, accuracy: 0.01)
                        XCTAssertEqual(padding.alphaComponent, 1, accuracy: 0.005)
                        var changed = 0
                        for y in Int(card.minY + 2)..<Int(card.maxY - 2) {
                            for x in Int(card.minX + 8)..<Int(card.maxX - 8) {
                                let a = try XCTUnwrap(with.colorAt(x: x * 2, y: y * 2))
                                let b = try XCTUnwrap(without.colorAt(x: x * 2, y: y * 2))
                                if abs(a.redComponent - b.redComponent) > 0.03 ||
                                   abs(a.greenComponent - b.greenComponent) > 0.03 ||
                                   abs(a.blueComponent - b.blueComponent) > 0.03 { changed += 1 }
                            }
                        }
                        print("LANE_STACK height=\(height) hovered=\(hovered) bucket=\(bucket) scheme=\(scheme) siblingPixels=\(changed)")
                        XCTAssertEqual(changed, 0, "A sibling title must never overpaint the hover card")
                        if let output = ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] {
                            try XCTUnwrap(with.representation(using: .png, properties: [:])).write(
                                to: URL(fileURLWithPath: output).appendingPathComponent("stack-\(Int(height))-\(hovered)-\(bucket)-\(scheme).png"))
                        }
                    }
                }
            }
        }
    }

    func testHoverTooltipHasOpaqueBackgroundInBothAppearances() throws {
        let size = CGSize(width: 360, height: 200)
        let presentation = SparklineChartPresentation(
            label: "Ingest", values: [42, 48, 55, 53, 60, 66, 71, 69, 74, 80],
            primarySeriesLabel: "Watcher"
        )
        // The VStack reserves 12 points plus 2 spacing for the x axis.
        let chartSize = CGSize(width: size.width, height: size.height - 14)
        let anchorY = 10 + (chartSize.height - 20) * (1 - 66.0 / Double(presentation.tightAxisMax))
        let tooltipSize = SparklineTooltipPlacement.tooltipSize(in: chartSize)
        let center = SparklineTooltipPlacement.position(
            near: CGPoint(x: 190, y: anchorY), anchorY: anchorY,
            hoveredX: 40 + 5 * (size.width - 50) / 9,
            containerBounds: chartSize, tooltipSize: tooltipSize
        )
        // Sample the card's empty bottom padding, safely inside its rounded corners.
        let sample = CGPoint(x: center.x, y: center.y + tooltipSize.height / 2 - 4)
        for scheme in [ColorScheme.light, .dark] {
            let chart = SparklineChart(
                presentation: presentation, accentColor: .systemGreen,
                previewHoveredBucket: 5, previewHoverX: 190
            ).frame(width: size.width, height: size.height).environment(\.colorScheme, scheme)
            let renderer = ImageRenderer(content: chart)
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.nsImage)
            let bitmap = try XCTUnwrap(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
            let color = try XCTUnwrap(bitmap.colorAt(x: Int(sample.x * 2), y: Int(sample.y * 2)))
            XCTAssertEqual(color.alphaComponent, 1, accuracy: 0.005,
                           "Hover tooltip must block the chart beneath it in \(scheme)")
            if let output = ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] {
                let url = URL(fileURLWithPath: output).appendingPathComponent("tooltip-\(scheme).png")
                try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: url)
            }
        }
    }

    func testCompact112PointHoverRetainsAllThreeSeriesValues() throws {
        for scheme in [ColorScheme.light, .dark] {
            let chart = SparklineChart(presentation: SparklineChartPresentation(
                label: "Ingest", values: values, secondaryValues: Array(repeating: 67, count: 10),
                tertiaryValues: Array(repeating: 68, count: 10), primarySeriesLabel: "Agent",
                secondarySeriesLabel: "Watcher", tertiarySeriesLabel: "Replay",
                metricDisclosure: "Window: Last 30m · Count: hovered value below · Unit: chunks · Clock: ingest time"
            ), accentColor: .systemGreen, previewHoveredBucket: 5, previewHoverX: 0)
                .frame(width: 360, height: 112).environment(\.colorScheme, scheme)
            let renderer = ImageRenderer(content: chart)
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.nsImage)
            let cg = try XCTUnwrap(image.cgImage(forProposedRect: nil, context: nil, hints: nil))
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
            let text = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
            for value in ["Agent", "Watcher", "Replay", "66", "67", "68"] {
                XCTAssertTrue(text.contains(value), "Missing \(value) from rendered compact card: \(text)")
            }
            if let output = ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] {
                try XCTUnwrap(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:))?
                    .representation(using: .png, properties: [:])).write(
                        to: URL(fileURLWithPath: output).appendingPathComponent("compact-three-series-\(scheme).png"))
            }
        }
    }
}
