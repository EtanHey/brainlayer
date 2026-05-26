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
    let activityWindowMinutes: Int
    let latestBucketName: String

    init(
        label: String,
        values: [Int],
        activityWindowMinutes: Int = 30,
        latestBucketName: String = "latest bucket count"
    ) {
        self.label = label
        self.values = values
        self.activityWindowMinutes = activityWindowMinutes
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

    func bucketLabel(for bucket: Int) -> String {
        guard !values.isEmpty else { return "no bucket" }
        let clampedBucket = min(max(bucket, 0), values.count - 1)
        let bucketWidth = max(1, Int(ceil(Double(activityWindowMinutes) / Double(values.count))))
        let newerMinutesAgo = max(values.count - 1 - clampedBucket, 0) * bucketWidth
        let olderMinutesAgo = min(newerMinutesAgo + bucketWidth, activityWindowMinutes)

        if newerMinutesAgo == 0 {
            return "last \(olderMinutesAgo)m"
        }
        return "\(olderMinutesAgo)-\(newerMinutesAgo)m ago"
    }

    func tooltipText(for point: SparklineChartPoint) -> String {
        "\(bucketLabel(for: point.bucket)): \(point.value)"
    }

    private var trendDescription: String {
        guard let last = values.last else {
            return "no trend"
        }
        guard values.count > 1 else {
            return "steady"
        }

        let previous = values[values.count - 2]
        if last > previous {
            return "trending up"
        }
        if last < previous {
            return "trending down"
        }
        return "steady"
    }
}

struct SparklineChart: View {
    let presentation: SparklineChartPresentation
    let accentColor: NSColor
    let compact: Bool
    @State private var hoveredPoint: SparklineChartPoint?
    @State private var hoverLocation: CGPoint?

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
        .chartOverlay { chartProxy in
            GeometryReader { geometry in
                if let plotAnchor = chartProxy.plotFrame {
                    let plotFrame = geometry[plotAnchor]

                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let location):
                                hoveredPoint = nearestPoint(
                                    to: location,
                                    plotFrame: plotFrame,
                                    chartProxy: chartProxy
                                )
                                hoverLocation = location
                            case .ended:
                                hoveredPoint = nil
                                hoverLocation = nil
                            }
                        }

                    if let hoveredPoint,
                       let hoverLocation,
                       !compact {
                        sparklineTooltip(for: hoveredPoint)
                            .position(tooltipPosition(near: hoverLocation, in: geometry.size))
                            .allowsHitTesting(false)
                    }
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(presentation.accessibilityLabel))
        .accessibilityValue(Text(presentation.accessibilityValue))
    }

    private func nearestPoint(
        to location: CGPoint,
        plotFrame: CGRect,
        chartProxy: ChartProxy
    ) -> SparklineChartPoint? {
        guard !presentation.points.isEmpty,
              plotFrame.contains(location) else {
            return nil
        }

        let plotX = location.x - plotFrame.minX
        guard let bucket: Double = chartProxy.value(atX: plotX, as: Double.self) else {
            return nil
        }
        let index = min(max(Int(bucket.rounded()), 0), presentation.points.count - 1)
        return presentation.points[index]
    }

    private func tooltipPosition(near location: CGPoint, in size: CGSize) -> CGPoint {
        let x = min(max(location.x, 58), max(size.width - 58, 58))
        let y = max(location.y - 28, 16)
        return CGPoint(x: x, y: y)
    }

    @ViewBuilder
    private func sparklineTooltip(for point: SparklineChartPoint) -> some View {
        Text(presentation.tooltipText(for: point))
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.primary)
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: true)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(nsColor: accentColor).opacity(0.35), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.14), radius: 8, y: 3)
    }
}

enum SparklineRenderer {
    static func isCompact(size: NSSize) -> Bool {
        let width = max(size.width.rounded(.up), 1)
        let height = max(size.height.rounded(.up), 1)
        return height <= 20 || width <= 52
    }

    static func endpoint(
        values: [Int],
        size: NSSize = NSSize(width: 44, height: 18)
    ) -> CGPoint? {
        let width = max(Int(size.width.rounded(.up)), 1)
        let height = max(Int(size.height.rounded(.up)), 1)
        let isCompact = isCompact(size: size)

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
            compact: isCompact(size: size)
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
