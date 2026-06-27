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
        if let primarySeriesLabel, !primarySeriesLabel.isEmpty { return primarySeriesLabel }
        let trimmed = label.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? "Activity" : trimmed
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

    var tightAxisMax: Int {
        maxValue <= 5 ? max(maxValue, 1) : axisMax
    }

    /// Rounds the raw peak up to a glanceable 1/2/5 x 10^n tick so y-axis labels are round.
    var axisMax: Int {
        let raw = maxValue
        guard raw > 1 else { return 1 }
        let exponent = floor(log10(Double(raw)))
        let base = pow(10, exponent)
        let fraction = Double(raw) / base
        let niceFraction: Double = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10
        return max(Int((niceFraction * base).rounded(.up)), 1)
    }
    /// Tick values bottom->top: always includes 0 (baseline) and axisMax (peak).
    var yAxisTicks: [Int] {
        let top = axisMax
        if top <= 2 { return Array(0...top) }
        let mid = Int((Double(top) / 2).rounded())
        return Array(Set([0, mid, top])).sorted()
    }

    var tightYAxisTicks: [Int] {
        let top = tightAxisMax
        if top <= 2 { return Array(0...top) }
        return Array(Set([0, Int(ceil(Double(top) / 2)), top])).sorted()
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

    func visiblePointMarkers(for role: SparklineSeriesRole, compact: Bool) -> [SparklineChartPoint] {
        let rolePoints = points(for: role)
        guard let latest = rolePoints.last else { return [] }
        if compact {
            return [latest]
        }
        if shouldEmphasizeSparsePoints(role) {
            return rolePoints.filter { $0.value > 0 }
        }
        return latest.value > 0 ? [latest] : []
    }

    func nonZeroFraction(_ role: SparklineSeriesRole) -> Double {
        let rolePoints = points(for: role)
        guard !rolePoints.isEmpty else { return 0 }
        return Double(rolePoints.filter { $0.value > 0 }.count) / Double(rolePoints.count)
    }

    func isDense(_ role: SparklineSeriesRole) -> Bool {
        nonZeroFraction(role) >= 0.5
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

    func bucketRecencyLabel(for bucket: Int) -> String {
        guard !values.isEmpty else { return "no bucket" }
        let clampedBucket = min(max(bucket, 0), values.count - 1)
        if clampedBucket == values.count - 1 {
            return "now"
        }
        let range = bucketRange(for: clampedBucket)
        return DashboardMetricFormatter.relativeEventString(lastEventAt: range.end, now: fetchedAt)
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
        let hours = seconds / 3600
        let minutesAfterHours = (seconds % 3600) / 60
        if hours > 0 {
            if minutesAfterHours == 0 {
                return "\(hours)h"
            }
            return "\(hours)h \(minutesAfterHours)m"
        }
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

enum SparklineSmoothing {
    case linear
    case monotone
    case catmullRom
}

/// Decides the area-fill floor for a series. Pure + testable.
enum SparklineBaseline {
    /// Sparse series with at least one nonzero bucket get a lifted soft floor so their
    /// runs of zeros read as a designed band around the spike. Dense series fill to the
    /// true zero baseline, and a series with NO data at all also stays on the true
    /// baseline so it never implies a phantom band of activity.
    static func usesSoftFloor(isDense: Bool, nonZeroFraction: Double) -> Bool {
        !isDense && nonZeroFraction > 0
    }
}

/// Picks which plotted series the hover indicator rides. Pure + testable.
enum SparklineHoverAnchor {
    /// Among the plotted series, return the one with the highest value at the hovered
    /// bucket; ties keep declaration order (primary first). Falls back to the first
    /// plotted series, then `.primary` when nothing is plotted — so the dot, crosshair,
    /// and tooltip never anchor to an empty baseline on a multi-series chart where only
    /// a secondary/tertiary lane has data.
    static func dominantRole(
        plotted: [SparklineSeriesRole],
        valueAtBucket: [SparklineSeriesRole: Int]
    ) -> SparklineSeriesRole {
        guard let first = plotted.first else { return .primary }
        var best = first
        var bestValue = valueAtBucket[first] ?? 0
        for role in plotted.dropFirst() {
            let value = valueAtBucket[role] ?? 0
            if value > bestValue {
                best = role
                bestValue = value
            }
        }
        return best
    }
}

struct SparklineChart: View {
    let presentation: SparklineChartPresentation
    let accentColor: NSColor
    let secondaryAccentColor: NSColor?
    let tertiaryAccentColor: NSColor?
    let compact: Bool
    let referenceValue: Int?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hoveredBucket: Int?
    @State private var hoverLocation: CGPoint?

    init(
        presentation: SparklineChartPresentation,
        accentColor: NSColor,
        secondaryAccentColor: NSColor? = nil,
        tertiaryAccentColor: NSColor? = nil,
        compact: Bool = false,
        referenceValue: Int? = nil
    ) {
        self.presentation = presentation
        self.accentColor = accentColor
        self.secondaryAccentColor = secondaryAccentColor
        self.tertiaryAccentColor = tertiaryAccentColor
        self.compact = compact
        self.referenceValue = referenceValue
    }

#if DEBUG
    /// Debug-only seam: render a fixed hover state (dot + crosshair + tooltip) without
    /// a live cursor, for snapshot/visual QA. Never compiled into a release build.
    init(
        presentation: SparklineChartPresentation,
        accentColor: NSColor,
        secondaryAccentColor: NSColor? = nil,
        tertiaryAccentColor: NSColor? = nil,
        compact: Bool = false,
        referenceValue: Int? = nil,
        previewHoveredBucket: Int?,
        previewHoverX: CGFloat
    ) {
        self.presentation = presentation
        self.accentColor = accentColor
        self.secondaryAccentColor = secondaryAccentColor
        self.tertiaryAccentColor = tertiaryAccentColor
        self.compact = compact
        self.referenceValue = referenceValue
        _hoveredBucket = State(initialValue: previewHoveredBucket)
        if previewHoveredBucket != nil {
            _hoverLocation = State(initialValue: CGPoint(x: previewHoverX, y: 0))
        }
    }
#endif

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
                .opacity(hoveredBucket == nil ? 0 : 1)
                .animation(reduceMotion ? nil : .easeInOut(duration: 0.15), value: hoveredBucket)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(presentation.accessibilityLabel))
        .accessibilityValue(Text(presentation.accessibilityValue))
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: presentation)
    }

    private var chartBody: some View {
        GeometryReader { geometry in
            let plotFrame = plotFrame(in: geometry.size)
            let isHovering = hoveredBucket != nil

            ZStack(alignment: .topLeading) {
                if !compact {
                    ForEach(presentation.tightYAxisTicks, id: \.self) { tick in
                        let y = yPosition(forValue: tick, in: plotFrame)
                        if tick == 0 || isHovering {
                            Path { p in
                                p.move(to: CGPoint(x: plotFrame.minX, y: y))
                                p.addLine(to: CGPoint(x: plotFrame.maxX, y: y))
                            }
                            .stroke(
                                Color.brainBarBorderSoft.opacity(tick == 0 ? 0.55 : 0.22),
                                style: StrokeStyle(lineWidth: tick == 0 ? 1 : 0.75,
                                                   dash: tick == 0 ? [] : [2, 3])
                            )
                            .transition(.opacity)

                            if tick != 0 {
                                Text(DashboardMetricFormatter.axisTickString(tick))
                                    .font(.system(size: 9, weight: .medium))
                                    .monospacedDigit()
                                    .foregroundStyle(Color.brainBarTextMuted)
                                    .frame(width: yAxisGutter - 4, alignment: .trailing)
                                    .position(x: (yAxisGutter - 4) / 2, y: y)
                                    .transition(.opacity)
                            }
                        }
                    }
                    .animation(reduceMotion ? nil : .easeInOut(duration: 0.15), value: hoveredBucket)
                }

                ForEach(SparklineSeriesRole.allCases, id: \.self) { role in
                    if !compact, presentation.shouldPlotSeries(role) {
                        SparklineSeriesAreaShape(
                            points: presentation.points(for: role),
                            maxValue: plotMax,
                            plotFrame: plotFrame,
                            baselineY: baselineY(for: role, in: plotFrame),
                            smoothing: smoothing(for: role)
                        )
                        .fill(
                            LinearGradient(
                                colors: [
                                    color(for: role).opacity(role == .primary ? 0.34 : 0.16),
                                    color(for: role).opacity(0.0),
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                    }
                }

                if !compact, let referenceValue {
                    let clampedReference = min(max(referenceValue, 0), plotMax)
                    let y = yPosition(forValue: clampedReference, in: plotFrame)
                    Path { p in
                        p.move(to: CGPoint(x: plotFrame.minX, y: y))
                        p.addLine(to: CGPoint(x: plotFrame.maxX, y: y))
                    }
                    .stroke(
                        color(for: .primary).opacity(0.4),
                        style: StrokeStyle(lineWidth: 1, lineCap: .round, dash: [4, 4])
                    )
                }

                if let hoveredBucket, !compact, !presentation.points.isEmpty {
                    let clampedBucket = min(max(hoveredBucket, 0), presentation.values.count - 1)
                    let anchorRole = hoverAnchorRole(forBucket: clampedBucket)
                    let anchorPoint = hoverAnchorPoint(forBucket: clampedBucket, in: plotFrame)
                    Path { p in
                        p.move(to: CGPoint(x: anchorPoint.x, y: plotFrame.minY))
                        p.addLine(to: CGPoint(x: anchorPoint.x, y: plotFrame.maxY))
                    }
                    .stroke(
                        color(for: anchorRole).opacity(0.3),
                        style: StrokeStyle(lineWidth: 1, lineCap: .round, dash: [3, 3])
                    )
                }

                ForEach(SparklineSeriesRole.allCases, id: \.self) { role in
                    if presentation.shouldPlotSeries(role) {
                        SparklineSeriesPathShape(
                            points: presentation.points(for: role),
                            maxValue: plotMax,
                            plotFrame: plotFrame,
                            smoothing: smoothing(for: role)
                        )
                        .stroke(color(for: role), style: lineStyle(for: role))
                        .shadow(
                            color: !compact && role == .primary ? color(for: role).opacity(0.25) : .clear,
                            radius: !compact && role == .primary ? 4 : 0,
                            y: !compact && role == .primary ? 1 : 0
                        )

                        ForEach(presentation.visiblePointMarkers(for: role, compact: compact)) { point in
                            Circle()
                                .fill(color(for: role))
                                .frame(width: compact ? 4.4 : 6.5, height: compact ? 4.4 : 6.5)
                                .overlay {
                                    if !compact {
                                        Circle()
                                            .stroke(Color.brainBarBackgroundRaised.opacity(0.9), lineWidth: 1.25)
                                    }
                                }
                                .shadow(
                                    color: !compact ? color(for: role).opacity(0.45) : .clear,
                                    radius: !compact ? 4 : 0,
                                    y: !compact ? 1 : 0
                                )
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

                if let hoveredBucket, !compact, !presentation.points.isEmpty {
                    let clampedBucket = min(max(hoveredBucket, 0), presentation.values.count - 1)
                    let anchorRole = hoverAnchorRole(forBucket: clampedBucket)
                    let anchorPoint = hoverAnchorPoint(forBucket: clampedBucket, in: plotFrame)
                    Circle()
                        .fill(color(for: anchorRole))
                        .frame(width: 8, height: 8)
                        .overlay(
                            Circle()
                                .stroke(Color.brainBarBackgroundRaised, lineWidth: 1.5)
                        )
                        .position(anchorPoint)
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
                   !compact,
                   !presentation.points.isEmpty {
                    let clampedBucket = min(max(hoveredBucket, 0), max(presentation.values.count - 1, 0))
                    let anchorPoint = hoverAnchorPoint(forBucket: clampedBucket, in: plotFrame)
                    let tooltipSize = SparklineTooltipPlacement.tooltipSize(
                        in: geometry.size,
                        seriesCount: tooltipRows(forBucket: clampedBucket).count
                    )
                    sparklineTooltip(forBucket: clampedBucket, size: tooltipSize)
                        .position(
                            SparklineTooltipPlacement.position(
                                near: CGPoint(x: hoverLocation.x, y: anchorPoint.y),
                                anchorY: anchorPoint.y,
                                hoveredX: anchorPoint.x,
                                containerBounds: geometry.size,
                                tooltipSize: tooltipSize
                            )
                        )
                        .allowsHitTesting(false)
                }
            }
        }
    }

    private func point(for point: SparklineChartPoint, bucketCount: Int, in plotFrame: CGRect) -> CGPoint {
        let x: CGFloat
        if bucketCount <= 1 {
            x = plotFrame.midX
        } else {
            x = plotFrame.minX + CGFloat(point.bucket) * (plotFrame.width / CGFloat(bucketCount - 1))
        }
        let normalizedValue = CGFloat(point.value) / CGFloat(max(plotMax, 1))
        return CGPoint(x: x, y: plotFrame.maxY - (normalizedValue * plotFrame.height))
    }

    /// The plotted series the hover indicator should ride at the given bucket (the
    /// visible spike may be on a secondary/tertiary lane, not the primary).
    private func hoverAnchorRole(forBucket bucket: Int) -> SparklineSeriesRole {
        let plotted = SparklineSeriesRole.allCases.filter { presentation.shouldPlotSeries($0) }
        let values = Dictionary(uniqueKeysWithValues: plotted.map { role -> (SparklineSeriesRole, Int) in
            let pts = presentation.points(for: role)
            return (role, pts.indices.contains(bucket) ? pts[bucket].value : 0)
        })
        return SparklineHoverAnchor.dominantRole(plotted: plotted, valueAtBucket: values)
    }

    /// On-curve anchor point (x = bucket position, y = dominant plotted series' value
    /// at that bucket) for the crosshair, ring dot, and tooltip vertical placement.
    private func hoverAnchorPoint(forBucket bucket: Int, in plotFrame: CGRect) -> CGPoint {
        let role = hoverAnchorRole(forBucket: bucket)
        let pts = presentation.points(for: role)
        let clamped = min(max(bucket, 0), max(pts.count - 1, 0))
        guard pts.indices.contains(clamped) else {
            return CGPoint(x: plotFrame.midX, y: plotFrame.maxY)
        }
        return point(for: pts[clamped], bucketCount: pts.count, in: plotFrame)
    }

    private func yPosition(forValue value: Int, in plotFrame: CGRect) -> CGFloat {
        let normalized = CGFloat(value) / CGFloat(max(plotMax, 1))
        return plotFrame.maxY - normalized * plotFrame.height
    }

    private var yAxisGutter: CGFloat { compact ? 0 : 30 }
    private var plotMax: Int { compact ? presentation.maxValue : presentation.tightAxisMax }

    private func smoothing(for role: SparklineSeriesRole) -> SparklineSmoothing {
        guard !compact else { return .linear }
        return presentation.isDense(role) ? .catmullRom : .monotone
    }

    private func baselineY(for role: SparklineSeriesRole, in plotFrame: CGRect) -> CGFloat {
        let softFloor = SparklineBaseline.usesSoftFloor(
            isDense: presentation.isDense(role),
            nonZeroFraction: presentation.nonZeroFraction(role)
        )
        return softFloor ? plotFrame.maxY - plotFrame.height * 0.10 : plotFrame.maxY
    }

    private func plotFrame(in size: CGSize) -> CGRect {
        let inset: CGFloat = compact ? 2 : 10
        let leftInset = inset + yAxisGutter
        return CGRect(
            x: leftInset,
            y: inset,
            width: max(size.width - leftInset - inset, 1),
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

    private func tooltipRows(forBucket bucket: Int) -> [(role: SparklineSeriesRole, label: String, value: Int)] {
        return SparklineSeriesRole.allCases.compactMap { role in
            guard presentation.shouldPlotSeries(role),
                  let label = presentation.label(for: role) else { return nil }
            let series: [Int]
            switch role {
            case .primary:   series = presentation.values
            case .secondary: series = presentation.secondaryValues
            case .tertiary:  series = presentation.tertiaryValues
            }
            let value = series.indices.contains(bucket) ? series[bucket] : 0
            return (role, label, value)
        }
    }

    @ViewBuilder
    private func sparklineTooltip(forBucket bucket: Int, size: CGSize) -> some View {
        let rows = tooltipRows(forBucket: bucket)
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(presentation.bucketRecencyLabel(for: bucket))
                    .font(.system(size: 11, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(Color.brainBarTextPrimary)
                Text("\(presentation.relativeBucketLabel(for: bucket)) · \(presentation.bucketLabel(for: bucket))")
                    .font(.system(size: 9.5, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.70))
            }
            .lineLimit(1)
            .fixedSize(horizontal: false, vertical: true)

            if !rows.isEmpty {
                Rectangle()
                    .fill(Color.brainBarBorderSoft)
                    .frame(height: 1)
                    .padding(.horizontal, 2)
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(rows, id: \.role) { row in
                        HStack(spacing: 6) {
                            RoundedRectangle(cornerRadius: 2, style: .continuous)
                                .fill(color(for: row.role))
                                .frame(width: 7, height: 7)
                            Text(row.label)
                                .font(.system(size: 10.5, weight: .medium))
                                .foregroundStyle(Color.brainBarTextSecondary)
                                .lineLimit(1).truncationMode(.tail)
                            Spacer(minLength: 8)
                            Text("\(row.value)")
                                .font(.system(size: 11, weight: .semibold))
                                .monospacedDigit()
                                .foregroundStyle(Color.brainBarTextPrimary)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(width: size.width, height: size.height, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Color.brainBarBackgroundRaised)
        )
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .strokeBorder(Color.brainBar(nsColor: accentColor).opacity(0.35), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.28), radius: 9, y: 3)
        .accessibilityHidden(true)
    }
}

private struct SparklineSeriesPathShape: Shape {
    let points: [SparklineChartPoint]
    let maxValue: Int
    let plotFrame: CGRect
    let smoothing: SparklineSmoothing

    func path(in rect: CGRect) -> Path {
        SparklineSeriesPathBuilder.topPath(
            for: points,
            maxValue: maxValue,
            plotFrame: plotFrame,
            smoothing: smoothing
        )
    }
}

private struct SparklineSeriesAreaShape: Shape {
    let points: [SparklineChartPoint]
    let maxValue: Int
    let plotFrame: CGRect
    let baselineY: CGFloat
    let smoothing: SparklineSmoothing

    func path(in rect: CGRect) -> Path {
        let renderedPoints = SparklineSeriesPathBuilder.renderedPoints(
            for: points,
            maxValue: maxValue,
            plotFrame: plotFrame
        )
        guard let first = renderedPoints.first, let last = renderedPoints.last else {
            return Path()
        }

        var path = SparklineSeriesPathBuilder.topPath(
            for: renderedPoints,
            plotFrame: plotFrame,
            smoothing: smoothing
        )
        let clampedBaseline = min(max(baselineY, plotFrame.minY), plotFrame.maxY)
        path.addLine(to: CGPoint(x: last.x, y: clampedBaseline))
        path.addLine(to: CGPoint(x: first.x, y: clampedBaseline))
        path.closeSubpath()
        return path
    }
}

private enum SparklineSeriesPathBuilder {
    static func topPath(
        for points: [SparklineChartPoint],
        maxValue: Int,
        plotFrame: CGRect,
        smoothing: SparklineSmoothing
    ) -> Path {
        topPath(
            for: renderedPoints(for: points, maxValue: maxValue, plotFrame: plotFrame),
            plotFrame: plotFrame,
            smoothing: smoothing
        )
    }

    static func renderedPoints(
        for points: [SparklineChartPoint],
        maxValue: Int,
        plotFrame: CGRect
    ) -> [CGPoint] {
        points.map { point in
            let x: CGFloat
            if points.count <= 1 {
                x = plotFrame.midX
            } else {
                x = plotFrame.minX + CGFloat(point.bucket) * (plotFrame.width / CGFloat(points.count - 1))
            }
            let normalizedValue = CGFloat(point.value) / CGFloat(max(maxValue, 1))
            return CGPoint(x: x, y: plotFrame.maxY - (normalizedValue * plotFrame.height))
        }
    }

    static func topPath(
        for renderedPoints: [CGPoint],
        plotFrame: CGRect,
        smoothing: SparklineSmoothing
    ) -> Path {
        guard let first = renderedPoints.first else { return Path() }
        guard renderedPoints.count > 1 else {
            var path = Path()
            path.move(to: first)
            return path
        }

        switch smoothing {
        case .linear:
            return linearPath(for: renderedPoints)
        case .monotone:
            return monotonePath(for: renderedPoints, plotFrame: plotFrame)
        case .catmullRom:
            return catmullRomPath(for: renderedPoints, plotFrame: plotFrame)
        }
    }

    private static func linearPath(for points: [CGPoint]) -> Path {
        var path = Path()
        for (index, point) in points.enumerated() {
            if index == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        return path
    }

    private static func monotonePath(for points: [CGPoint], plotFrame: CGRect) -> Path {
        let count = points.count
        var secants = Array(repeating: CGFloat.zero, count: count - 1)
        for index in 0..<(count - 1) {
            let dx = points[index + 1].x - points[index].x
            secants[index] = dx == 0 ? 0 : (points[index + 1].y - points[index].y) / dx
        }

        var tangents = Array(repeating: CGFloat.zero, count: count)
        tangents[0] = secants[0]
        tangents[count - 1] = secants[count - 2]
        if count > 2 {
            for index in 1..<(count - 1) {
                let previous = secants[index - 1]
                let next = secants[index]
                if previous == 0 || next == 0 || (previous > 0) != (next > 0) {
                    tangents[index] = 0
                } else {
                    tangents[index] = (previous + next) / 2
                }
            }
        }

        for index in 0..<(count - 1) {
            let secant = secants[index]
            guard secant != 0 else {
                tangents[index] = 0
                tangents[index + 1] = 0
                continue
            }
            let alpha = tangents[index] / secant
            let beta = tangents[index + 1] / secant
            let magnitude = alpha * alpha + beta * beta
            if magnitude > 9 {
                let scale = 3 / sqrt(magnitude)
                tangents[index] = scale * alpha * secant
                tangents[index + 1] = scale * beta * secant
            }
        }

        var path = Path()
        path.move(to: points[0])
        for index in 0..<(count - 1) {
            let dx = points[index + 1].x - points[index].x
            let control1 = CGPoint(
                x: points[index].x + dx / 3,
                y: clampY(points[index].y + tangents[index] * dx / 3, in: plotFrame)
            )
            let control2 = CGPoint(
                x: points[index + 1].x - dx / 3,
                y: clampY(points[index + 1].y - tangents[index + 1] * dx / 3, in: plotFrame)
            )
            path.addCurve(to: points[index + 1], control1: control1, control2: control2)
        }
        return path
    }

    private static func catmullRomPath(for points: [CGPoint], plotFrame: CGRect) -> Path {
        var path = Path()
        path.move(to: points[0])
        for index in 0..<(points.count - 1) {
            let p0 = points[max(index - 1, 0)]
            let p1 = points[index]
            let p2 = points[index + 1]
            let p3 = points[min(index + 2, points.count - 1)]
            let control1 = CGPoint(
                x: p1.x + (p2.x - p0.x) / 6,
                y: clampY(p1.y + (p2.y - p0.y) / 6, in: plotFrame)
            )
            let control2 = CGPoint(
                x: p2.x - (p3.x - p1.x) / 6,
                y: clampY(p2.y - (p3.y - p1.y) / 6, in: plotFrame)
            )
            path.addCurve(to: p2, control1: control1, control2: control2)
        }
        return path
    }

    private static func clampY(_ y: CGFloat, in plotFrame: CGRect) -> CGFloat {
        min(max(y, plotFrame.minY), plotFrame.maxY)
    }
}

enum SparklineTooltipPlacement {
    private static let margin: CGFloat = 8
    private static let preferredWidth: CGFloat = 260
    private static let minimumWidth: CGFloat = 112
    private static let cursorGap: CGFloat = 12
    // Structured-card metrics = the PINNED card height (matches sparklineTooltip's paddings/spacing).
    private static let verticalPadding: CGFloat = 9
    private static let headerHeight: CGFloat = 24
    private static let dividerBlock: CGFloat = 14
    private static let rowHeight: CGFloat = 16

    static func tooltipSize(in containerSize: CGSize, seriesCount: Int = 1) -> CGSize {
        let availableWidth = max(containerSize.width - margin * 2, minimumWidth)
        let rows = max(seriesCount, 0)
        let rowsBlock = rows > 0 ? dividerBlock + CGFloat(rows) * rowHeight : 0
        let height = verticalPadding * 2 + headerHeight + rowsBlock
        let availableHeight = max(containerSize.height - margin * 2, height)
        return CGSize(width: min(preferredWidth, availableWidth),
                      height: min(height, availableHeight))
    }

    static func position(
        near location: CGPoint,
        anchorY: CGFloat,
        hoveredX: CGFloat,
        containerBounds: CGSize,
        tooltipSize: CGSize
    ) -> CGPoint {
        let halfWidth = tooltipSize.width / 2
        let halfHeight = tooltipSize.height / 2
        let minX = margin + halfWidth
        let maxX = max(containerBounds.width - margin - halfWidth, minX)
        let x = min(max(location.x, minX), maxX)
        let minY = margin + halfHeight
        let maxY = max(containerBounds.height - margin - halfHeight, minY)

        let yAbove = anchorY - cursorGap - halfHeight
        if yAbove >= minY {
            return CGPoint(x: x, y: yAbove)
        }

        let yBelow = anchorY + cursorGap + halfHeight
        if yBelow <= maxY {
            return CGPoint(x: x, y: yBelow)
        }

        let rightX = hoveredX + halfWidth + cursorGap
        let leftX = hoveredX - halfWidth - cursorGap
        let sideX: CGFloat
        if rightX >= minX, rightX <= maxX {
            sideX = rightX
        } else if leftX >= minX, leftX <= maxX {
            sideX = leftX
        } else {
            // No clean side fits at the anchor column — pin to whichever edge has
            // more room so the card still clears the on-curve dot instead of
            // landing on the anchor column and overlapping the curve (a top-edge
            // spike in a narrow / tall-tooltip container).
            sideX = (hoveredX - minX) >= (maxX - hoveredX) ? minX : maxX
        }
        let y = min(max(anchorY, minY), maxY)

        return CGPoint(x: sideX, y: y)
    }
}

/// The menu-bar status icon: three overlapping pipeline sparklines (Agent stores /
/// JSONL watcher / Enrichment) over an always-visible baseline. Bright colors on a
/// transparent ground (isTemplate=false) keep it visible on a dark fullscreen menu
/// bar — where the old single gray line vanished. Even when every series is idle,
/// the baseline keeps the icon on screen.
struct MenuBarSparklineIcon: View {
    struct Series: Equatable {
        let values: [Int]
        let color: Color
    }

    let series: [Series]

    var body: some View {
        Canvas { ctx, size in
            let inset: CGFloat = 1.5
            let rect = CGRect(
                x: inset,
                y: inset,
                width: max(size.width - inset * 2, 1),
                height: max(size.height - inset * 2, 1)
            )

            // Always-visible baseline so the icon never disappears on any background.
            var baseline = Path()
            baseline.move(to: CGPoint(x: rect.minX, y: rect.maxY))
            baseline.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
            ctx.stroke(baseline, with: .color(.white.opacity(0.28)), lineWidth: 0.75)

            // Shared scale across series so relative activity reads at a glance.
            let globalMax = max(series.flatMap { $0.values }.max() ?? 0, 1)
            for s in series {
                guard s.values.count > 1 else { continue }
                var path = Path()
                for (index, value) in s.values.enumerated() {
                    let x = rect.minX + CGFloat(index) * (rect.width / CGFloat(s.values.count - 1))
                    let normalized = CGFloat(value) / CGFloat(globalMax)
                    let y = rect.maxY - normalized * rect.height
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                ctx.stroke(
                    path,
                    with: .color(s.color),
                    style: StrokeStyle(lineWidth: 1.2, lineCap: .round, lineJoin: .round)
                )
            }
        }
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

    /// Render the three-line pipeline status-bar icon (Agent / Watcher / Enrichment).
    /// isTemplate=false: we want the bright colors, and the baseline guarantees the
    /// icon stays visible on a dark fullscreen menu bar even when all series are idle.
    @MainActor
    static func renderStatusBarIcon(
        agent: [Int],
        watcher: [Int],
        enrichment: [Int],
        size: NSSize = NSSize(width: 26, height: 14)
    ) -> NSImage {
        let width = max(size.width.rounded(.up), 1)
        let height = max(size.height.rounded(.up), 1)
        let icon = MenuBarSparklineIcon(series: [
            .init(values: agent, color: Color(nsColor: BrainBarDesignTokens.Colors.seriesAgent)),
            .init(values: watcher, color: Color(nsColor: BrainBarDesignTokens.Colors.seriesWatcher)),
            .init(values: enrichment, color: Color(nsColor: BrainBarDesignTokens.Colors.signalFTS5)),
        ])
        .frame(width: width, height: height)

        let renderer = ImageRenderer(content: icon)
        renderer.scale = NSScreen.main?.backingScaleFactor ?? 2
        if let image = renderer.nsImage {
            image.isTemplate = false
            return image
        }
        return NSImage(size: NSSize(width: width, height: height))
    }
}
