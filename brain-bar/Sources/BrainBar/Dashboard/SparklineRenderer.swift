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
    let fetchedAt: Date

    init(
        label: String,
        values: [Int],
        activityWindowMinutes: Int = 30,
        latestBucketName: String = "latest bucket count",
        fetchedAt: Date = Date()
    ) {
        self.label = label
        self.values = values
        self.activityWindowMinutes = activityWindowMinutes
        self.latestBucketName = latestBucketName
        self.fetchedAt = fetchedAt
    }

    var points: [SparklineChartPoint] {
        values.enumerated().map { index, value in
            SparklineChartPoint(bucket: index, value: value)
        }
    }

    var latestPoint: SparklineChartPoint? {
        points.last
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
        let range = bucketRange(for: clampedBucket)
        return "\(DashboardMetricFormatter.shortAbsoluteTimeString(range.start))-\(DashboardMetricFormatter.shortAbsoluteTimeString(range.end))"
    }

    func relativeBucketLabel(for bucket: Int) -> String {
        guard !values.isEmpty else { return "no bucket" }
        let clampedBucket = min(max(bucket, 0), values.count - 1)
        let totalSeconds = max(activityWindowMinutes * 60, 1)
        let bucketWidthSeconds = max(1, Double(totalSeconds) / Double(values.count))
        let bucketStart = Double(clampedBucket) * bucketWidthSeconds
        let bucketEnd = min(Double(clampedBucket + 1) * bucketWidthSeconds, Double(totalSeconds))
        let olderSecondsAgo = max(0, Int(round(Double(totalSeconds) - bucketStart)))
        let newerSecondsAgo = max(0, Int(round(Double(totalSeconds) - bucketEnd)))

        if newerSecondsAgo == 0 {
            return "last \(Self.durationLabel(seconds: olderSecondsAgo))"
        }
        return "\(Self.durationLabel(seconds: olderSecondsAgo))-\(Self.durationLabel(seconds: newerSecondsAgo)) ago"
    }

    func tooltipText(forBucket bucket: Int) -> String {
        let clampedBucket = min(max(bucket, 0), max(values.count - 1, 0))
        let value = values.indices.contains(clampedBucket) ? values[clampedBucket] : 0
        return "\(bucketLabel(for: clampedBucket)) (\(relativeBucketLabel(for: clampedBucket))): \(value)"
    }

    private func bucketRange(for bucket: Int) -> (start: Date, end: Date) {
        let totalSeconds = max(activityWindowMinutes * 60, 1)
        let bucketWidthSeconds = max(1, Double(totalSeconds) / Double(max(values.count, 1)))
        let bucketStart = Double(bucket) * bucketWidthSeconds
        let bucketEnd = min(Double(bucket + 1) * bucketWidthSeconds, Double(totalSeconds))
        let start = fetchedAt.addingTimeInterval(-(Double(totalSeconds) - bucketStart))
        let end = fetchedAt.addingTimeInterval(-(Double(totalSeconds) - bucketEnd))
        return (start, end)
    }

    private static func durationLabel(seconds: Int) -> String {
        let minutes = seconds / 60
        let remainingSeconds = seconds % 60
        if remainingSeconds == 0 {
            return "\(minutes)m"
        }
        if minutes == 0 {
            return "\(remainingSeconds)s"
        }
        return "\(minutes)m \(remainingSeconds)s"
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
    @State private var hoveredBucket: Int?
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
        VStack(spacing: 2) {
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

                if point == presentation.latestPoint {
                    PointMark(
                        x: .value("Bucket", point.bucket),
                        y: .value("Count", point.value)
                    )
                    .foregroundStyle(Color(nsColor: accentColor))
                    .symbolSize(compact ? 18 : 42)
                }
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
                                    hoveredBucket = nearestBucket(
                                        to: location,
                                        plotFrame: plotFrame,
                                        chartProxy: chartProxy
                                    )
                                    hoverLocation = location
                                case .ended:
                                    hoveredBucket = nil
                                    hoverLocation = nil
                                }
                            }

                        if let hoveredBucket,
                           let hoverLocation,
                           !compact {
                            sparklineTooltip(forBucket: hoveredBucket)
                                .position(tooltipPosition(near: hoverLocation, in: geometry.size))
                                .allowsHitTesting(false)
                        }
                    }
                }
            }

            if !compact {
                HStack {
                    ForEach(xAxisBuckets, id: \.self) { bucket in
                        Text(presentation.bucketLabel(for: bucket))
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                        if bucket != xAxisBuckets.last {
                            Spacer(minLength: 8)
                        }
                    }
                }
                .frame(height: 12)
                .padding(.horizontal, 16)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(presentation.accessibilityLabel))
        .accessibilityValue(Text(presentation.accessibilityValue))
    }

    private func nearestBucket(
        to location: CGPoint,
        plotFrame: CGRect,
        chartProxy: ChartProxy
    ) -> Int? {
        guard !presentation.points.isEmpty,
              plotFrame.contains(location) else {
            return nil
        }

        let plotX = location.x - plotFrame.minX
        guard let bucket: Double = chartProxy.value(atX: plotX, as: Double.self) else {
            return nil
        }
        return min(max(Int(bucket.rounded()), 0), presentation.points.count - 1)
    }

    private var xAxisBuckets: [Int] {
        guard !presentation.values.isEmpty else { return [] }
        let last = presentation.values.count - 1
        if last <= 0 { return [0] }
        return Array(Set([0, last / 2, last])).sorted()
    }

    private func tooltipPosition(near location: CGPoint, in size: CGSize) -> CGPoint {
        let x = min(max(location.x, 58), max(size.width - 58, 58))
        let y = max(location.y - 28, 16)
        return CGPoint(x: x, y: y)
    }

    @ViewBuilder
    private func sparklineTooltip(forBucket bucket: Int) -> some View {
        Text(presentation.tooltipText(forBucket: bucket))
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
