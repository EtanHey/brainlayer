import AppKit
import Charts
import Foundation
import SwiftUI

struct SparklineChartPoint: Identifiable, Equatable, Sendable {
    let bucket: Int
    let value: Int

    var id: Int { bucket }
}

struct SparklineChartPresentation: Equatable, Sendable {
    let label: String
    let values: [Int]
    let latestBucketName: String

    init(
        label: String,
        values: [Int],
        latestBucketName: String = "latest bucket count"
    ) {
        self.label = label
        self.values = values
        self.latestBucketName = latestBucketName
    }

    var points: [SparklineChartPoint] {
        values.enumerated().map { index, value in
            SparklineChartPoint(bucket: index, value: value)
        }
    }

    var accessibilityLabel: String {
        label
    }

    var accessibilityValue: String {
        "\(latestBucketName) \(values.last ?? 0), \(trendDescription)"
    }

    var maxValue: Int {
        max(values.max() ?? 0, 1)
    }

    private var trendDescription: String {
        guard let first = values.first, let last = values.last else {
            return "no trend"
        }
        if last > first {
            return "trending up"
        }
        if last < first {
            return "trending down"
        }
        return "steady"
    }
}

struct SparklineChart: View {
    let presentation: SparklineChartPresentation
    let accentColor: NSColor
    let compact: Bool

    init(
        presentation: SparklineChartPresentation,
        accentColor: NSColor,
        compact: Bool = false
    ) {
        self.presentation = presentation
        self.accentColor = accentColor
        self.compact = compact
    }

    var body: some View {
        Chart(presentation.points) { point in
            if !compact {
                AreaMark(
                    x: .value("Bucket", point.bucket),
                    y: .value("Count", point.value)
                )
                .foregroundStyle(Color(nsColor: accentColor).opacity(0.10))
            }

            LineMark(
                x: .value("Bucket", point.bucket),
                y: .value("Count", point.value)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(Color(nsColor: accentColor).opacity(0.85))
            .lineStyle(StrokeStyle(lineWidth: compact ? 1.6 : 2, lineCap: .round, lineJoin: .round))
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: 0...presentation.maxValue)
        .chartPlotStyle { plotArea in
            plotArea
                .background(Color.clear)
                .padding(compact ? 2 : 10)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(presentation.accessibilityLabel))
        .accessibilityValue(Text(presentation.accessibilityValue))
    }
}

enum SparklineRenderer {
    static func endpoint(
        values: [Int],
        size: NSSize = NSSize(width: 44, height: 18)
    ) -> CGPoint? {
        let width = max(Int(size.width.rounded(.up)), 1)
        let height = max(Int(size.height.rounded(.up)), 1)
        let isCompact = height <= 20 || width <= 52

        guard values.count > 1 else { return nil }

        let maxValue = max(values.max() ?? 0, 1)
        let horizontalInset: CGFloat = isCompact ? 2 : 10
        let verticalInset: CGFloat = isCompact ? 2 : 10
        let chartRect = CGRect(
            x: horizontalInset,
            y: verticalInset,
            width: max(CGFloat(width) - (horizontalInset * 2), 1),
            height: max(CGFloat(height) - (verticalInset * 2), 1)
        )
        let step = chartRect.width / CGFloat(max(values.count - 1, 1))

        guard let lastValue = values.last else { return nil }
        let normalized = CGFloat(lastValue) / CGFloat(maxValue)
        return CGPoint(
            x: chartRect.minX + CGFloat(values.count - 1) * step,
            y: chartRect.minY + normalized * chartRect.height
        )
    }

    @MainActor
    static func render(
        state: PipelineState,
        values: [Int],
        size: NSSize = NSSize(width: 44, height: 18),
        accentColor: NSColor? = nil
    ) -> NSImage {
        let width = max(size.width.rounded(.up), 1)
        let height = max(size.height.rounded(.up), 1)
        let chart = SparklineChart(
            presentation: SparklineChartPresentation(
                label: "Recent activity sparkline",
                values: values
            ),
            accentColor: accentColor ?? state.color,
            compact: height <= 20 || width <= 52
        )
        .frame(width: width, height: height)

        let renderer = ImageRenderer(content: chart)
        renderer.scale = NSScreen.main?.backingScaleFactor ?? 2
        if let image = renderer.nsImage {
            image.isTemplate = false
            return image
        }
        return NSImage(size: NSSize(width: width, height: height))
    }
}
