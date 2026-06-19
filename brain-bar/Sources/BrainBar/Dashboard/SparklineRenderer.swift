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

struct SparklineLegendEntry: Equatable, Sendable {
    let label: String
    let isActive: Bool
}

enum SparklineSeriesRole: CaseIterable, Equatable, Sendable {
    case primary
    case secondary
    case tertiary
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

    var legendEntries: [SparklineLegendEntry] {
        var entries = [
            SparklineLegendEntry(label: primaryLegendLabel, isActive: isSeriesActive(.primary)),
        ]
        if hasSecondarySeries {
            entries.append(SparklineLegendEntry(label: secondaryLegendLabel, isActive: isSeriesActive(.secondary)))
        }
        if hasTertiarySeries {
            entries.append(SparklineLegendEntry(label: tertiaryLegendLabel, isActive: isSeriesActive(.tertiary)))
        }
        return entries
    }

    var visibleSeriesLabels: [String] {
        SparklineSeriesRole.allCases.compactMap { role in
            guard shouldPlotSeries(role) else { return nil }
            return label(for: role)
        }
    }

    var showsListeningForWritesCaption: Bool {
        hasMultipleSeries &&
            label.localizedCaseInsensitiveContains("writes") &&
            !legendEntries.contains(where: \.isActive)
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

    func points(for role: SparklineSeriesRole) -> [SparklineChartPoint] {
        switch role {
        case .primary:
            points
        case .secondary:
            secondaryPoints
        case .tertiary:
            tertiaryPoints
        }
    }

    func label(for role: SparklineSeriesRole) -> String? {
        switch role {
        case .primary:
            primaryLegendLabel
        case .secondary:
            hasSecondarySeries ? secondaryLegendLabel : nil
        case .tertiary:
            hasTertiarySeries ? tertiaryLegendLabel : nil
        }
    }

    func isSeriesActive(_ role: SparklineSeriesRole) -> Bool {
        switch role {
        case .primary:
            values.reduce(0, +) > 0
        case .secondary:
            secondaryValues.reduce(0, +) > 0
        case .tertiary:
            tertiaryValues.reduce(0, +) > 0
        }
    }

    func shouldPlotSeries(_ role: SparklineSeriesRole) -> Bool {
        guard label(for: role) != nil else { return false }
        if hasMultipleSeries {
            return isSeriesActive(role)
        }
        return !points(for: role).isEmpty
    }

    func shouldEmphasizeSparsePoints(_ role: SparklineSeriesRole) -> Bool {
        guard shouldPlotSeries(role) else { return false }
        let total = points(for: role).reduce(0) { $0 + $1.value }
        return (1...2).contains(total)
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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hoveredBucket: Int?
    @State private var hoverLocation: CGPoint?
    @State private var lineRevealProgress: CGFloat = 1

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
            if presentation.showsListeningForWritesCaption {
                Text("Listening for writes...")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.60))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .transition(.opacity)
            } else {
                chartBody
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
        .onAppear {
            if reduceMotion {
                lineRevealProgress = 1
            } else {
                lineRevealProgress = 0
                withAnimation(.easeOut(duration: 0.6)) {
                    lineRevealProgress = 1
                }
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: presentation)
    }

    private var chartBody: some View {
        GeometryReader { geometry in
            let plotFrame = plotFrame(in: geometry.size)

            ZStack(alignment: .topLeading) {
                ForEach(SparklineSeriesRole.allCases, id: \.self) { role in
                    if presentation.shouldPlotSeries(role) {
                        SparklineSeriesPathShape(
                            points: presentation.points(for: role),
                            maxValue: presentation.maxValue,
                            plotFrame: plotFrame
                        )
                        .stroke(color(for: role), style: lineStyle(for: role))

                        ForEach(visiblePointMarkers(for: role)) { point in
                            Circle()
                                .fill(color(for: role))
                                .frame(width: compact ? 4.4 : 9, height: compact ? 4.4 : 9)
                                .position(
                                    self.point(
                                        for: point,
                                        bucketCount: presentation.points(for: role).count,
                                        in: plotFrame
                                    )
                                )
                        }
                    }
                }

                Rectangle()
                    .fill(.clear)
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            hoveredBucket = nearestBucket(to: location, plotFrame: plotFrame)
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

    private func visiblePointMarkers(for role: SparklineSeriesRole) -> [SparklineChartPoint] {
        let points = presentation.points(for: role)
        guard let last = points.last else { return [] }
        if presentation.shouldEmphasizeSparsePoints(role) {
            return points.filter { $0.value > 0 }
        }
        return [last]
    }

    private func point(for point: SparklineChartPoint, bucketCount: Int, in plotFrame: CGRect) -> CGPoint {
        let x: CGFloat
        if bucketCount <= 1 {
            x = plotFrame.midX
        } else {
            x = plotFrame.minX + CGFloat(point.bucket) * (plotFrame.width / CGFloat(bucketCount - 1))
        }
        let normalizedValue = CGFloat(point.value) / CGFloat(max(presentation.maxValue, 1))
        return CGPoint(x: x, y: plotFrame.maxY - (normalizedValue * plotFrame.height))
    }

    private func plotFrame(in size: CGSize) -> CGRect {
        let inset: CGFloat = compact ? 2 : 10
        return CGRect(
            x: inset,
            y: inset,
            width: max(size.width - inset * 2, 1),
            height: max(size.height - inset * 2, 1)
        )
    }

    private func nearestBucket(to location: CGPoint, plotFrame: CGRect) -> Int? {
        guard !presentation.points.isEmpty, plotFrame.contains(location) else { return nil }
        let bucketCount = max(presentation.points.count, 1)
        if bucketCount == 1 { return 0 }
        let normalizedX = min(max((location.x - plotFrame.minX) / max(plotFrame.width, 1), 0), 1)
        return Int((normalizedX * CGFloat(bucketCount - 1)).rounded())
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
        let primaryColor = color(for: .primary)
        let secondaryColor = color(for: .secondary)
        let tertiaryColor = color(for: .tertiary)

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

    private func color(for role: SparklineSeriesRole) -> Color {
        switch role {
        case .primary:
            Color.brainBar(nsColor: accentColor)
        case .secondary:
            Color.brainBar(nsColor: secondaryAccentColor ?? BrainBarDesignTokens.Colors.seriesWatcher)
        case .tertiary:
            Color.brainBar(nsColor: tertiaryAccentColor ?? BrainBarDesignTokens.Colors.signalTrigram)
        }
    }

    private func lineStyle(for role: SparklineSeriesRole) -> StrokeStyle {
        switch role {
        case .primary:
            StrokeStyle(lineWidth: compact ? 1.6 : 2.0, lineCap: .round, lineJoin: .round)
        case .secondary:
            StrokeStyle(lineWidth: compact ? 1.4 : 1.75, lineCap: .round, lineJoin: .round)
        case .tertiary:
            StrokeStyle(lineWidth: compact ? 1.25 : 1.5, lineCap: .round, lineJoin: .round)
        }
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

private struct SparklineSeriesPathShape: Shape {
    let points: [SparklineChartPoint]
    let maxValue: Int
    let plotFrame: CGRect

    func path(in rect: CGRect) -> Path {
        var path = Path()
        for (index, point) in points.enumerated() {
            let renderedPoint = renderedPoint(for: point, bucketCount: points.count)
            if index == 0 {
                path.move(to: renderedPoint)
            } else {
                path.addLine(to: renderedPoint)
            }
        }
        return path
    }

    private func renderedPoint(for point: SparklineChartPoint, bucketCount: Int) -> CGPoint {
        let x: CGFloat
        if bucketCount <= 1 {
            x = plotFrame.midX
        } else {
            x = plotFrame.minX + CGFloat(point.bucket) * (plotFrame.width / CGFloat(bucketCount - 1))
        }
        let normalizedValue = CGFloat(point.value) / CGFloat(max(maxValue, 1))
        return CGPoint(x: x, y: plotFrame.maxY - (normalizedValue * plotFrame.height))
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
