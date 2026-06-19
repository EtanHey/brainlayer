import AppKit
import Charts
import Foundation
import SwiftUI

struct SparklineChartPoint: Identifiable, Equatable, Sendable {
    let bucket: Int
    let value: Int
    let timestamp: Date

    var id: Int { bucket }
}

struct SparklineChartPresentation: Equatable, Sendable {
    let label: String
    let values: [Int]
    let secondaryValues: [Int]
    let tertiaryValues: [Int]
    let primarySeriesLabel: String?
    let secondarySeriesLabel: String?
    let tertiarySeriesLabel: String?
    let activityWindowMinutes: Int
    let latestBucketName: String
    let fetchedAt: Date

    init(
        label: String,
        values: [Int],
        secondaryValues: [Int] = [],
        tertiaryValues: [Int] = [],
        primarySeriesLabel: String? = nil,
        secondarySeriesLabel: String? = nil,
        tertiarySeriesLabel: String? = nil,
        activityWindowMinutes: Int = 30,
        latestBucketName: String = "latest bucket count",
        fetchedAt: Date = Date()
    ) {
        self.label = label
        self.values = values
        self.secondaryValues = secondaryValues
        self.tertiaryValues = tertiaryValues
        self.primarySeriesLabel = primarySeriesLabel
        self.secondarySeriesLabel = secondarySeriesLabel
        self.tertiarySeriesLabel = tertiarySeriesLabel
        self.activityWindowMinutes = activityWindowMinutes
        self.latestBucketName = latestBucketName
        self.fetchedAt = fetchedAt
    }

    var points: [SparklineChartPoint] {
        values.enumerated().map { index, value in
            SparklineChartPoint(bucket: index, value: value, timestamp: bucketMidpoint(for: index))
        }
    }

    var latestPoint: SparklineChartPoint? {
        points.last
    }

    var secondaryPoints: [SparklineChartPoint] {
        secondaryValues.enumerated().map { index, value in
            SparklineChartPoint(bucket: index, value: value, timestamp: bucketMidpoint(for: index))
        }
    }

    var tertiaryPoints: [SparklineChartPoint] {
        tertiaryValues.enumerated().map { index, value in
            SparklineChartPoint(bucket: index, value: value, timestamp: bucketMidpoint(for: index))
        }
    }

    var hasSecondarySeries: Bool {
        !secondaryValues.isEmpty
    }

    var hasTertiarySeries: Bool {
        !tertiaryValues.isEmpty
    }

    var hasMultipleSeries: Bool {
        hasSecondarySeries || hasTertiarySeries
    }

    var primaryLegendLabel: String {
        primarySeriesLabel ?? "Primary"
    }

    var secondaryLegendLabel: String {
        secondarySeriesLabel ?? "Secondary"
    }

    var tertiaryLegendLabel: String {
        tertiarySeriesLabel ?? "Tertiary"
    }

    var accessibilityLabel: String {
        label
    }

    var accessibilityValue: String {
        var components = ["\(latestBucketName) \(values.last ?? 0)", trendDescription]
        if let primarySeriesLabel {
            components.append("\(primarySeriesLabel) latest bucket \(values.last ?? 0)")
        }
        if let secondarySeriesLabel, hasSecondarySeries {
            components.append("\(secondarySeriesLabel) latest bucket \(secondaryValues.last ?? 0)")
        }
        if let tertiarySeriesLabel, hasTertiarySeries {
            components.append("\(tertiarySeriesLabel) latest bucket \(tertiaryValues.last ?? 0)")
        }
        return components.joined(separator: ", ")
    }

    var maxValue: Int {
        max(values.max() ?? 0, secondaryValues.max() ?? 0, tertiaryValues.max() ?? 0, 1)
    }

    var xAxisDomainStart: Date {
        fetchedAt.addingTimeInterval(-Double(max(activityWindowMinutes * 60, 1)))
    }

    var xAxisDomainEnd: Date {
        fetchedAt
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
        let primaryValue = values.indices.contains(clampedBucket) ? values[clampedBucket] : 0
        guard hasMultipleSeries else {
            return "\(bucketLabel(for: clampedBucket)) (\(relativeBucketLabel(for: clampedBucket))): \(primaryValue)"
        }

        var seriesComponents = ["\(primarySeriesLabel ?? "Primary") \(primaryValue)"]
        if let secondarySeriesLabel, hasSecondarySeries {
            let value = secondaryValues.indices.contains(clampedBucket) ? secondaryValues[clampedBucket] : 0
            seriesComponents.append("\(secondarySeriesLabel) \(value)")
        }
        if let tertiarySeriesLabel, hasTertiarySeries {
            let value = tertiaryValues.indices.contains(clampedBucket) ? tertiaryValues[clampedBucket] : 0
            seriesComponents.append("\(tertiarySeriesLabel) \(value)")
        }
        return "\(bucketLabel(for: clampedBucket)) (\(relativeBucketLabel(for: clampedBucket))): \(seriesComponents.joined(separator: ", "))"
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

    private func bucketMidpoint(for bucket: Int) -> Date {
        let range = bucketRange(for: bucket)
        return range.start.addingTimeInterval(range.end.timeIntervalSince(range.start) / 2)
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
    let secondaryAccentColor: NSColor?
    let tertiaryAccentColor: NSColor?
    let compact: Bool
    @State private var hoveredBucket: Int?
    @State private var hoverLocation: CGPoint?

    init(
        presentation: SparklineChartPresentation,
        accentColor: NSColor,
        secondaryAccentColor: NSColor? = nil,
        tertiaryAccentColor: NSColor? = nil,
        compact: Bool = false
    ) {
        self.presentation = presentation
        self.accentColor = accentColor
        self.secondaryAccentColor = secondaryAccentColor
        self.tertiaryAccentColor = tertiaryAccentColor
        self.compact = compact
    }

    var body: some View {
        VStack(spacing: 2) {
            Chart {
                ForEach(presentation.points) { point in
                    if !compact && !presentation.hasMultipleSeries {
                        AreaMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Count", point.value)
                        )
                        .foregroundStyle(Color.brainBar(nsColor: accentColor).opacity(0.10))
                    }

                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Count", point.value),
                        series: .value("Series", presentation.primaryLegendLabel)
                    )
                    .interpolationMethod(.linear)
                    .foregroundStyle(by: .value("Series", presentation.primaryLegendLabel))
                    .lineStyle(StrokeStyle(lineWidth: compact ? 1.6 : 2, lineCap: .round, lineJoin: .round))

                    if point == presentation.latestPoint {
                        PointMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Count", point.value)
                        )
                        .foregroundStyle(Color.brainBar(nsColor: accentColor))
                        .symbolSize(compact ? 18 : 42)
                    }
                }

                if presentation.hasSecondarySeries {
                    ForEach(presentation.secondaryPoints) { point in
                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Count", point.value),
                            series: .value("Series", presentation.secondaryLegendLabel)
                        )
                        .interpolationMethod(.linear)
                        .foregroundStyle(by: .value("Series", presentation.secondaryLegendLabel))
                        .lineStyle(StrokeStyle(lineWidth: compact ? 1.4 : 2, lineCap: .round, lineJoin: .round, dash: compact ? [] : [4, 3]))
                    }
                }

                if presentation.hasTertiarySeries {
                    ForEach(presentation.tertiaryPoints) { point in
                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Count", point.value),
                            series: .value("Series", presentation.tertiaryLegendLabel)
                        )
                        .interpolationMethod(.linear)
                        .foregroundStyle(by: .value("Series", presentation.tertiaryLegendLabel))
                        .lineStyle(StrokeStyle(lineWidth: compact ? 1.4 : 2, lineCap: .round, lineJoin: .round, dash: compact ? [] : [2, 3]))
                    }
                }
            }
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
            .chartLegend(.hidden)
            .chartForegroundStyleScale(chartForegroundStyleScale)
            .chartXScale(domain: presentation.xAxisDomainStart...presentation.xAxisDomainEnd)
            .chartYScale(domain: 0...presentation.maxValue)
            .chartPlotStyle { plotArea in
                plotArea
                    .background(Color.brainBarClear)
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
                            let tooltipSize = SparklineTooltipPlacement.tooltipSize(in: geometry.size)
                            sparklineTooltip(forBucket: hoveredBucket)
                                .frame(width: tooltipSize.width, alignment: .leading)
                                .position(
                                    SparklineTooltipPlacement.position(
                                        near: hoverLocation,
                                        in: geometry.size,
                                        tooltipSize: tooltipSize
                                    )
                                )
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
        guard let hoveredDate: Date = chartProxy.value(atX: plotX, as: Date.self) else {
            return nil
        }
        let distances = presentation.points.map { abs($0.timestamp.timeIntervalSince(hoveredDate)) }
        guard let minDistance = distances.min(),
              let bucket = distances.firstIndex(of: minDistance) else {
            return nil
        }
        return bucket
    }

    private var xAxisBuckets: [Int] {
        guard !presentation.values.isEmpty else { return [] }
        let last = presentation.values.count - 1
        if last <= 0 { return [0] }
        return Array(Set([0, last / 2, last])).sorted()
    }

    private var chartForegroundStyleScale: KeyValuePairs<String, Color> {
        let primaryColor = Color.brainBar(nsColor: accentColor)
        let secondaryColor = Color.brainBar(nsColor: secondaryAccentColor ?? .systemOrange)
        let tertiaryColor = Color.brainBar(nsColor: tertiaryAccentColor ?? .systemPurple)

        if presentation.hasSecondarySeries && presentation.hasTertiarySeries {
            return [
                presentation.primaryLegendLabel: primaryColor,
                presentation.secondaryLegendLabel: secondaryColor,
                presentation.tertiaryLegendLabel: tertiaryColor,
            ]
        }
        if presentation.hasSecondarySeries {
            return [
                presentation.primaryLegendLabel: primaryColor,
                presentation.secondaryLegendLabel: secondaryColor,
            ]
        }
        if presentation.hasTertiarySeries {
            return [
                presentation.primaryLegendLabel: primaryColor,
                presentation.tertiaryLegendLabel: tertiaryColor,
            ]
        }
        return [presentation.primaryLegendLabel: primaryColor]
    }

    @ViewBuilder
    private func sparklineTooltip(forBucket bucket: Int) -> some View {
        Text(presentation.tooltipText(forBucket: bucket))
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.primary)
            .lineLimit(2)
            .truncationMode(.tail)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.brainBar(nsColor: accentColor).opacity(0.35), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.14), radius: 8, y: 3)
    }
}

enum SparklineTooltipPlacement {
    private static let margin: CGFloat = 8
    private static let preferredWidth: CGFloat = 260
    private static let minimumWidth: CGFloat = 112
    private static let estimatedHeight: CGFloat = 44
    private static let cursorGap: CGFloat = 12

    static func tooltipSize(in containerSize: CGSize) -> CGSize {
        let availableWidth = max(containerSize.width - margin * 2, minimumWidth)
        return CGSize(
            width: min(preferredWidth, availableWidth),
            height: estimatedHeight
        )
    }

    static func position(
        near location: CGPoint,
        in containerSize: CGSize,
        tooltipSize: CGSize
    ) -> CGPoint {
        let halfWidth = tooltipSize.width / 2
        let halfHeight = tooltipSize.height / 2
        let minX = margin + halfWidth
        let maxX = max(containerSize.width - margin - halfWidth, minX)
        let x = min(max(location.x, minX), maxX)

        let yAbove = location.y - cursorGap - halfHeight
        let yBelow = location.y + cursorGap + halfHeight
        let minY = margin + halfHeight
        let maxY = max(containerSize.height - margin - halfHeight, minY)
        let preferredY = yAbove >= minY ? yAbove : yBelow
        let y = min(max(preferredY, minY), maxY)

        return CGPoint(x: x, y: y)
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
