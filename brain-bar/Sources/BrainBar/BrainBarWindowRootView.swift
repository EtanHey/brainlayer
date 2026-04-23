import AppKit
import Combine
import SwiftUI

struct BrainBarWindowRootView: View {
    @ObservedObject var runtime: BrainBarRuntime

    @State private var selectedTab: BrainBarTab = .dashboard
    @State private var hasActivatedInjectionsTab = false
    @State private var hasActivatedGraphTab = false
    @State private var commandBarPanelState = QuickCapturePanelState()
    @State private var commandBarViewModel: QuickCaptureViewModel?
    @StateObject private var windowObserver: BrainBarWindowObserver

    init(runtime: BrainBarRuntime) {
        self.runtime = runtime
        _windowObserver = StateObject(
            wrappedValue: BrainBarWindowObserver(coordinator: runtime.windowCoordinator)
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            BrainBarWindowHeader(
                selectedTab: $selectedTab,
                hotkeyStatus: runtime.hotkeyStatus.statusLine,
                commandBarViewModel: commandBarViewModel
            )

            Divider()

            ZStack {
                dashboardContent
                    .brainBarTabVisibility(selectedTab == .dashboard)

                if hasActivatedInjectionsTab || selectedTab == .injections {
                    injectionsContent
                        .brainBarTabVisibility(selectedTab == .injections)
                }

                if hasActivatedGraphTab || selectedTab == .graph {
                    graphContent
                        .brainBarTabVisibility(selectedTab == .graph)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay {
                // Overlay carries its own full-area tap-catcher and only
                // renders when the user is on the Dashboard tab with a
                // non-empty search query that hasn't been dismissed.
                BrainBarCommandBarResultsOverlay(
                    viewModel: commandBarViewModel,
                    isOnActiveTab: selectedTab == .dashboard
                )
            }
        }
        .frame(
            minWidth: 760,
            idealWidth: 900,
            maxWidth: 1_600,
            minHeight: 560,
            idealHeight: 640,
            maxHeight: 1_200
        )
        .opacity(windowObserver.isContentReady ? 1 : 0)
        .background(Color(nsColor: .windowBackgroundColor))
        .background(WindowAttachmentView { window in
            windowObserver.attach(window: window)
        })
        .onAppear {
            ensureCommandBarViewModel()
            activate(tab: selectedTab)
            if let action = runtime.requestedQuickAction {
                handleRequestedQuickAction(action)
            }
        }
        .onChange(of: selectedTab) { _, newTab in
            activate(tab: newTab)
        }
        .onReceive(runtime.$database) { _ in
            ensureCommandBarViewModel()
            // DB just became available — replay any pending request that was
            // left in runtime.requestedQuickAction while we were still warming.
            if let action = runtime.requestedQuickAction {
                handleRequestedQuickAction(action)
            }
        }
        .onReceive(runtime.$requestedQuickAction.compactMap { $0 }) { action in
            handleRequestedQuickAction(action)
        }
    }

    @ViewBuilder
    private var dashboardContent: some View {
        if let collector = runtime.collector {
            BrainBarDashboardView(
                collector: collector,
                hotkeyStatus: runtime.hotkeyStatus.statusLine
            )
        } else {
            BrainBarLoadingView(title: "BrainBar", subtitle: "Opening database and warming the dashboard...")
        }
    }

    @ViewBuilder
    private var injectionsContent: some View {
        if let store = runtime.injectionStore {
            BrainBarInjectionTab(store: store)
        } else {
            BrainBarLoadingView(title: "Injections", subtitle: "Injection store unavailable.")
        }
    }

    @ViewBuilder
    private var graphContent: some View {
        if let database = runtime.database {
            BrainBarGraphTab(database: database)
        } else {
            BrainBarLoadingView(title: "Graph", subtitle: "Knowledge graph unavailable.")
        }
    }

    private func ensureCommandBarViewModel() {
        guard commandBarViewModel == nil, let database = runtime.database else { return }
        commandBarViewModel = QuickCaptureViewModel(db: database, panelState: commandBarPanelState)
    }

    private func handleRequestedQuickAction(_ action: BrainBarQuickAction) {
        ensureCommandBarViewModel()
        // If the DB isn't ready yet, leave the request in flight and replay
        // when `onReceive(runtime.$database)` fires.
        guard let vm = commandBarViewModel else { return }
        selectedTab = .dashboard
        vm.setMode(action == .capture ? .capture : .search)
        vm.panelDidAppear()
        runtime.clearQuickActionRequest()
    }

    private func activate(tab: BrainBarTab) {
        switch tab {
        case .dashboard:
            break
        case .injections:
            hasActivatedInjectionsTab = true
        case .graph:
            hasActivatedGraphTab = true
        }
    }
}

private extension View {
    @ViewBuilder
    func brainBarTabVisibility(_ isVisible: Bool) -> some View {
        opacity(isVisible ? 1 : 0)
            .allowsHitTesting(isVisible)
            .accessibilityHidden(!isVisible)
    }
}

private struct BrainBarWindowHeader: View {
    @Binding var selectedTab: BrainBarTab

    let hotkeyStatus: String
    let commandBarViewModel: QuickCaptureViewModel?

    var body: some View {
        VStack(spacing: 10) {
            HStack(alignment: .center, spacing: 16) {
                Label("BrainBar", systemImage: "brain")
                    .font(.system(size: 18, weight: .semibold))
                    .labelStyle(.titleAndIcon)

                Spacer(minLength: 16)

                Picker("Section", selection: $selectedTab) {
                    ForEach(BrainBarTab.allCases) { tab in
                        Text(tab.title).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 280)
                .labelsHidden()

                Spacer(minLength: 16)

                Text(hotkeyStatus)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }

            BrainBarCommandBar(viewModel: commandBarViewModel)
        }
        .padding(.horizontal, 20)
        .padding(.top, 14)
        .padding(.bottom, 12)
        .background(.regularMaterial)
        .background(WindowDragHandle())
    }
}

private struct BrainBarDashboardView: View {
    @ObservedObject var collector: StatsCollector
    let hotkeyStatus: String

    @State private var previousWriteBuckets: [Int] = []
    @State private var previousEnrichmentBuckets: [Int] = []
    @State private var writePulseRevision = 0
    @State private var enrichmentPulseRevision = 0
    @State private var detailsExpanded = false

    private var flowSummary: DashboardFlowSummary {
        DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats)
    }

    var body: some View {
        GeometryReader { proxy in
            let layout = BrainBarDashboardLayout(containerSize: proxy.size)

            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: layout.sectionSpacing) {
                    overviewCard(layout: layout)
                    chartCards(layout: layout)
                    agentPresenceStrip(layout: layout)
                    if layout.usesQueueRail {
                        queueRail(layout: layout)
                    }
                    diagnostics(layout: layout)
                }
                .padding(layout.outerPadding)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .onAppear {
            previousWriteBuckets = collector.stats.recentActivityBuckets
            previousEnrichmentBuckets = collector.stats.recentEnrichmentBuckets
        }
        .onChange(of: collector.stats.recentActivityBuckets) { _, newBuckets in
            if BrainBarLivePulse.shouldPulse(previous: previousWriteBuckets, current: newBuckets) {
                writePulseRevision += 1
            }
            previousWriteBuckets = newBuckets
        }
        .onChange(of: collector.stats.recentEnrichmentBuckets) { _, newBuckets in
            if BrainBarLivePulse.shouldPulse(previous: previousEnrichmentBuckets, current: newBuckets) {
                enrichmentPulseRevision += 1
            }
            previousEnrichmentBuckets = newBuckets
        }
    }

    @ViewBuilder
    private func overviewCard(layout: BrainBarDashboardLayout) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: layout.gridSpacing) {
                VStack(alignment: .leading, spacing: 12) {
                    overviewNarrative(layout: layout)
                    overviewCaption
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                overviewMetaRow(layout: layout)
                    .frame(width: layout.overviewStatsWidth)
            }

            VStack(alignment: .leading, spacing: layout.gridSpacing) {
                overviewNarrative(layout: layout)
                overviewMetaRow(layout: layout)
                overviewCaption
            }
        }
        .padding(layout.cardPadding)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(nsColor: .controlBackgroundColor),
                            Color(nsColor: flowAccentColor).opacity(0.16),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(Color(nsColor: flowAccentColor).opacity(0.18), lineWidth: 1)
        )
    }

    private func overviewNarrative(layout: BrainBarDashboardLayout) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(flowSummary.headline)
                .font(.system(size: layout.overviewTitleFontSize, weight: .bold))
                .fixedSize(horizontal: false, vertical: true)

            Text(flowSummary.detail)
                .font(.system(size: layout.overviewSubtitleFontSize, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    overviewBadgeRow
                }

                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        BrainBarHeroBadge(text: flowSummary.windowLabel)
                        BrainBarHeroBadge(text: flowSummary.queue.status.label)
                    }

                    HStack(spacing: 8) {
                        BrainBarHeroBadge(text: "Writes \(flowSummary.ingress.lastEventText)")
                        BrainBarHeroBadge(text: "Enrichments \(flowSummary.enrichment.lastEventText)")
                    }
                }
            }
        }
    }

    private var overviewCaption: some View {
        Text("Live uses the last \(collector.stats.liveWindowMinutes)m. Trend cards use \(flowSummary.windowLabel.lowercased()).")
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var overviewBadgeRow: some View {
        Group {
            BrainBarHeroBadge(text: flowSummary.windowLabel)
            BrainBarHeroBadge(text: flowSummary.queue.status.label)
            BrainBarHeroBadge(text: "Writes \(flowSummary.ingress.lastEventText)")
            BrainBarHeroBadge(text: "Enrichments \(flowSummary.enrichment.lastEventText)")
        }
    }

    private func overviewMetaRow(layout: BrainBarDashboardLayout) -> some View {
        let columns = Array(
            repeating: GridItem(.flexible(minimum: 130), spacing: layout.gridSpacing, alignment: .leading),
            count: 2
        )

        return LazyVGrid(columns: columns, spacing: layout.gridSpacing) {
            BrainBarOverviewStat(label: "Chunks", value: "\(collector.stats.chunkCount)")
            BrainBarOverviewStat(label: "Enriched", value: "\(collector.stats.enrichedChunkCount)")
            BrainBarOverviewStat(label: "Backlog", value: "\(collector.stats.pendingEnrichmentCount)")
            BrainBarOverviewStat(label: "Coverage", value: "\(Int(collector.stats.enrichmentPercent.rounded()))%")
        }
    }

    @ViewBuilder
    private func chartCards(layout: BrainBarDashboardLayout) -> some View {
        if layout.chartColumns == 2 {
            HStack(alignment: .top, spacing: layout.gridSpacing) {
                writesCard(layout: layout)
                enrichmentsCard(layout: layout)
            }
        } else {
            VStack(spacing: layout.gridSpacing) {
                writesCard(layout: layout)
                enrichmentsCard(layout: layout)
            }
        }
    }

    private func writesCard(layout: BrainBarDashboardLayout) -> some View {
        BrainBarFlowLaneCard(
            lane: flowSummary.ingress,
            pulseRevision: writePulseRevision,
            compact: layout.compactCards,
            chartHeight: layout.sparklineHeight,
            emphasize: true
        )
    }

    private func enrichmentsCard(layout: BrainBarDashboardLayout) -> some View {
        BrainBarFlowLaneCard(
            lane: flowSummary.enrichment,
            pulseRevision: enrichmentPulseRevision,
            compact: layout.compactCards,
            chartHeight: layout.sparklineHeight,
            emphasize: true
        )
    }

    private func queueRail(layout: BrainBarDashboardLayout) -> some View {
        BrainBarQueueRail(
            summary: flowSummary.queue,
            coverageText: "\(Int(collector.stats.enrichmentPercent.rounded()))% enriched",
            compact: layout.compactCards
        )
    }

    private func agentPresenceStrip(layout: BrainBarDashboardLayout) -> some View {
        BrainBarAgentPresenceStrip(
            activity: collector.agentActivity,
            compact: layout.compactCards
        )
    }

    @ViewBuilder
    private func diagnostics(layout: BrainBarDashboardLayout) -> some View {
        let flowCard = BrainBarDiagnosticCard(
            title: "Flow",
            rows: [
                ("Writes", flowSummary.ingress.statusText),
                ("Enrichment", flowSummary.enrichment.statusText),
                ("Queue", flowSummary.queue.title),
                ("Window", flowSummary.windowLabel),
                ("DB", ByteCountFormatter.string(
                    fromByteCount: collector.stats.databaseSizeBytes,
                    countStyle: .file
                )),
            ],
            columns: layout.diagnosticItemColumns
        )

        let runtimeCard = BrainBarDiagnosticCard(
            title: "Runtime",
            rows: [
                ("Hotkey", hotkeyStatus),
                ("Daemon", daemonSummary),
                ("Agents", collector.agentActivity.summaryText),
                ("State", collector.state.label),
                ("Last seen", daemonLastSeenSummary),
            ],
            columns: layout.diagnosticItemColumns
        )

        VStack(alignment: .leading, spacing: 12) {
            DisclosureGroup(isExpanded: $detailsExpanded) {
                Group {
                    if layout.diagnosticColumns == 2 {
                        HStack(alignment: .top, spacing: layout.gridSpacing) {
                            flowCard
                            runtimeCard
                        }
                    } else {
                        VStack(spacing: layout.gridSpacing) {
                            flowCard
                            runtimeCard
                        }
                    }
                }
                .padding(.top, 4)
            } label: {
                HStack(alignment: .center, spacing: 12) {
                    Text("Runtime & Details")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer(minLength: 8)
                    Text("\(daemonSummary) · \(ByteCountFormatter.string(fromByteCount: collector.stats.databaseSizeBytes, countStyle: .file))")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(BrainBarDashboardCardStyle())
    }

    private var flowAccentColor: NSColor {
        switch collector.state {
        case .indexing:
            return .systemBlue
        case .enriching:
            return .systemGreen
        case .idle:
            return .systemGray
        case .degraded:
            return .systemOrange
        }
    }

    private var daemonSummary: String {
        guard let daemon = collector.daemon else { return "unavailable" }
        return "PID \(daemon.pid) · \(daemon.openConnections) sockets"
    }

    private var daemonLastSeenSummary: String {
        guard let daemon = collector.daemon else { return "Unavailable" }
        return DashboardMetricFormatter.lastEventString(
            lastEventAt: daemon.lastSeenAt,
            activityWindowMinutes: collector.stats.activityWindowMinutes
        )
    }
}

private struct BrainBarFlowLaneCard: View {
    let lane: DashboardFlowLane
    let pulseRevision: Int
    let compact: Bool
    let chartHeight: CGFloat
    let emphasize: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 10 : 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(lane.name)
                        .font(.system(size: emphasize ? (compact ? 16 : 18) : (compact ? 15 : 16), weight: .semibold))
                    Text(lane.windowLabel)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 12)
                BrainBarFlowStatusPill(
                    text: lane.status.label,
                    accentColor: Color(nsColor: lane.accentColor)
                )
            }

            BrainBarHeroSparkline(
                values: lane.values,
                accentColor: lane.accentColor,
                pulseRevision: pulseRevision
            )
            .frame(height: chartHeight)

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    BrainBarLaneMetric(label: "Rate", value: lane.rateText)
                    BrainBarLaneMetric(label: "Volume", value: lane.volumeText)
                    BrainBarLaneMetric(label: "Last event", value: lane.lastEventText)
                    Spacer(minLength: 0)
                }

                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 12) {
                        BrainBarLaneMetric(label: "Rate", value: lane.rateText)
                        BrainBarLaneMetric(label: "Volume", value: lane.volumeText)
                        Spacer(minLength: 0)
                    }
                    BrainBarLaneMetric(label: "Last event", value: lane.lastEventText)
                }
            }

            Text(lane.statusText)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(compact ? 16 : 18)
        .background(BrainBarDashboardCardStyle(emphasized: emphasize))
    }
}

private struct BrainBarQueueRail: View {
    let summary: DashboardQueueSummary
    let coverageText: String
    let compact: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 6 : 8) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 12) {
                    queueHeader
                    Spacer(minLength: 8)
                    queueMetrics
                }

                VStack(alignment: .leading, spacing: compact ? 6 : 8) {
                    queueHeader
                    queueMetrics
                }
            }

            Text(summary.detail)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(compact ? 10 : 12)
        .background(BrainBarDashboardCardStyle())
    }

    private var queueHeader: some View {
        HStack(alignment: .center, spacing: 10) {
            Label("Queue", systemImage: directionSymbolName)
                .font(.system(size: compact ? 12 : 13, weight: .semibold))
                .foregroundStyle(queueColor)

            BrainBarFlowStatusPill(
                text: summary.status.label,
                accentColor: queueColor
            )
        }
    }

    private var queueMetrics: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) {
                BrainBarLaneMetric(label: "Backlog", value: "\(summary.backlogCount)")
                BrainBarLaneMetric(label: "Coverage", value: coverageText)
            }

            VStack(alignment: .leading, spacing: 8) {
                BrainBarLaneMetric(label: "Backlog", value: "\(summary.backlogCount)")
                BrainBarLaneMetric(label: "Coverage", value: coverageText)
            }
        }
    }

    private var queueColor: Color {
        switch summary.status {
        case .empty, .stable:
            return Color(nsColor: .systemTeal)
        case .draining:
            return Color(nsColor: .systemGreen)
        case .growing, .backlogged:
            return Color(nsColor: .systemOrange)
        case .unavailable:
            return Color(nsColor: .systemRed)
        }
    }

    private var directionSymbolName: String {
        switch summary.status {
        case .empty, .stable:
            return "arrow.left.and.right"
        case .draining:
            return "arrow.down.right"
        case .growing, .backlogged:
            return "arrow.up.right"
        case .unavailable:
            return "exclamationmark.triangle"
        }
    }

    private var directionText: String {
        switch summary.status {
        case .empty:
            return "Empty"
        case .stable:
            return "Balanced"
        case .growing:
            return "Growing"
        case .draining:
            return "Draining"
        case .backlogged:
            return "Stalled"
        case .unavailable:
            return "Offline"
        }
    }
}

struct BrainBarDashboardLayout {
    let chartColumns: Int
    let overviewMetricColumns: Int
    let diagnosticColumns: Int
    let diagnosticItemColumns: Int
    let usesQueueRail: Bool
    let compactCards: Bool
    let outerPadding: CGFloat
    let sectionSpacing: CGFloat
    let gridSpacing: CGFloat
    let cardPadding: CGFloat
    let overviewTitleFontSize: CGFloat
    let overviewSubtitleFontSize: CGFloat
    let overviewStatsWidth: CGFloat
    let metricCardMinHeight: CGFloat
    let metricValueFontSize: CGFloat
    let sparklineHeight: CGFloat

    init(containerSize: CGSize) {
        let compactHeight = containerSize.height < 620
        let compactWidth = containerSize.width < 920

        chartColumns = containerSize.width >= 980 ? 2 : 1
        overviewMetricColumns = containerSize.width >= 980 ? 4 : 2
        diagnosticColumns = containerSize.width >= 960 ? 2 : 1
        diagnosticItemColumns = containerSize.width >= 820 ? 2 : 1
        usesQueueRail = true

        compactCards = compactWidth || compactHeight
        outerPadding = compactCards ? 14 : 18
        sectionSpacing = compactCards ? 12 : 16
        gridSpacing = compactCards ? 10 : 14
        cardPadding = compactCards ? 12 : 16
        overviewTitleFontSize = compactCards ? 20 : 24
        overviewSubtitleFontSize = compactCards ? 13 : 14
        overviewStatsWidth = compactCards ? 292 : 340
        metricCardMinHeight = compactCards ? 70 : 82
        metricValueFontSize = compactCards ? 20 : 24
        sparklineHeight = compactCards ? 102 : 116
    }
}

private struct BrainBarInjectionTab: View {
    @ObservedObject var store: InjectionStore
    @State private var filterText = ""

    var body: some View {
        InjectionFeedView(store: store, filterText: $filterText)
            .padding(16)
            .onAppear { store.start() }
    }
}

private struct BrainBarGraphTab: View {
    @StateObject private var viewModel: KGViewModel

    init(database: BrainDatabase) {
        _viewModel = StateObject(wrappedValue: KGViewModel(database: database))
    }

    var body: some View {
        KGCanvasView(viewModel: viewModel)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct BrainBarMetricCard: View {
    let title: String
    let value: String
    let valueFontSize: CGFloat
    let minHeight: CGFloat
    let cardPadding: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            Text(value)
                .font(.system(size: valueFontSize, weight: .semibold, design: .rounded))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, minHeight: minHeight, alignment: .leading)
        .padding(cardPadding)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
    }
}

private struct BrainBarLaneMetric: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
    }
}

private struct BrainBarDiagnosticCard: View {
    let title: String
    let rows: [(String, String)]
    let columns: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))

            LazyVGrid(
                columns: Array(
                    repeating: GridItem(.flexible(minimum: 150), spacing: 10, alignment: .leading),
                    count: columns
                ),
                alignment: .leading,
                spacing: 10
            ) {
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    BrainBarDiagnosticTile(label: row.0, value: row.1)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(BrainBarDashboardCardStyle())
    }
}

private struct BrainBarHeroBadge: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.white.opacity(0.18), in: Capsule())
    }
}

private struct BrainBarOverviewStat: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .lineLimit(1)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(nsColor: .windowBackgroundColor).opacity(0.66))
        )
    }
}

private struct BrainBarAgentPresenceStrip: View {
    let activity: AgentActivitySnapshot
    let compact: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 10) {
            HStack(alignment: .center, spacing: 12) {
                Text("Live agents")
                    .font(.system(size: compact ? 13 : 14, weight: .semibold))
                Spacer(minLength: 8)
                Text(activity.summaryText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    ForEach(activity.presences, id: \.family) { presence in
                        BrainBarAgentPresencePill(presence: presence)
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    ForEach(activity.presences, id: \.family) { presence in
                        BrainBarAgentPresencePill(presence: presence)
                    }
                }
            }

            Text("Agent presence is separate from indexed writes. A live CLI does not imply new chunks landed.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(compact ? 12 : 14)
        .background(BrainBarDashboardCardStyle())
    }
}

private struct BrainBarAgentPresencePill: View {
    let presence: AgentPresence

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color(nsColor: presence.family.accentColor))
                .frame(width: 8, height: 8)
                .opacity(presence.isActive ? 1 : 0.25)
            Text(presence.family.label)
                .font(.system(size: 11, weight: .semibold))
            Text("\(presence.count)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(
                    Capsule()
                        .fill(Color(nsColor: presence.family.accentColor).opacity(presence.isActive ? 0.18 : 0.08))
                )
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            Capsule()
                .fill(Color(nsColor: .windowBackgroundColor).opacity(0.66))
        )
        .overlay(
            Capsule()
                .stroke(Color(nsColor: presence.family.accentColor).opacity(presence.isActive ? 0.28 : 0.1), lineWidth: 1)
        )
    }
}

private struct BrainBarDiagnosticTile: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(nsColor: .windowBackgroundColor).opacity(0.56))
        )
    }
}

private struct BrainBarDashboardCardStyle: View {
    var emphasized = false

    var body: some View {
        RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(
                LinearGradient(
                    colors: emphasized
                        ? [
                            Color(nsColor: .controlBackgroundColor).opacity(0.98),
                            Color(nsColor: .windowBackgroundColor).opacity(0.92),
                        ]
                        : [
                            Color(nsColor: .controlBackgroundColor).opacity(0.96),
                            Color(nsColor: .controlBackgroundColor).opacity(0.94),
                        ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.white.opacity(emphasized ? 0.08 : 0.04), lineWidth: 1)
            )
    }
}

private struct BrainBarFlowStatusPill: View {
    let text: String
    let accentColor: Color

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(accentColor.opacity(0.16), in: Capsule())
            .overlay(
                Capsule()
                    .stroke(accentColor.opacity(0.28), lineWidth: 1)
            )
    }
}

private struct BrainBarHeroSparkline: View {
    let values: [Int]
    let accentColor: NSColor
    let pulseRevision: Int

    @State private var ringScale: CGFloat = 0.8
    @State private var ringOpacity = 0.0

    var body: some View {
        GeometryReader { proxy in
            let renderSize = NSSize(
                width: max(proxy.size.width.rounded(.up), 1),
                height: max(proxy.size.height.rounded(.up), 1)
            )
            let image = SparklineRenderer.render(
                state: .idle,
                values: values,
                size: renderSize,
                accentColor: accentColor
            )
            let endpoint = SparklineRenderer.endpoint(
                values: values,
                size: renderSize
            )

            ZStack(alignment: .topLeading) {
                Image(nsImage: image)
                    .interpolation(.high)
                    .resizable()
                    .frame(width: proxy.size.width, height: proxy.size.height)

                if let endpoint {
                    let point = CGPoint(
                        x: endpoint.x,
                        y: proxy.size.height - endpoint.y
                    )

                    Circle()
                        .stroke(Color(nsColor: accentColor).opacity(0.45), lineWidth: 2)
                        .frame(width: 26, height: 26)
                        .scaleEffect(ringScale)
                        .opacity(ringOpacity)
                        .position(point)

                    Circle()
                        .fill(Color(nsColor: accentColor))
                        .frame(width: 9, height: 9)
                        .shadow(color: Color(nsColor: accentColor).opacity(0.65), radius: 6)
                        .position(point)
                }
            }
        }
        .onAppear {
            triggerPulse()
        }
        .onChange(of: pulseRevision) { _, _ in
            triggerPulse()
        }
    }

    private func triggerPulse() {
        ringScale = 0.8
        ringOpacity = 0.7
        withAnimation(.easeOut(duration: 0.75)) {
            ringScale = 1.75
            ringOpacity = 0
        }
    }
}

struct BrainBarLoadingView: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.large)
            Text(title)
                .font(.system(size: 20, weight: .semibold))
            Text(subtitle)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(32)
    }
}

private struct WindowDragHandle: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        DragHandleView(frame: .zero)
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

private final class DragHandleView: NSView {
    override var mouseDownCanMoveWindow: Bool {
        true
    }
}

private struct WindowAttachmentView: NSViewRepresentable {
    let onResolve: (NSWindow) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        DispatchQueue.main.async {
            if let window = view.window {
                onResolve(window)
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            if let window = nsView.window {
                onResolve(window)
            }
        }
    }
}

@MainActor
private final class BrainBarWindowObserver: ObservableObject {
    @Published private(set) var isContentReady = false

    private let coordinator: BrainBarWindowCoordinator
    private var observers: [NSObjectProtocol] = []
    private var preparedWindowNumber: Int?

    init(coordinator: BrainBarWindowCoordinator) {
        self.coordinator = coordinator
    }

    func attach(window: NSWindow) {
        let needsPreparation = preparedWindowNumber != window.windowNumber
        if needsPreparation {
            preparedWindowNumber = window.windowNumber
            isContentReady = false
            window.alphaValue = 0
        }

        removeObservers()
        configure(window: window)
        coordinator.attach(window: window)

        if needsPreparation {
            DispatchQueue.main.async { [weak self, weak window] in
                self?.isContentReady = true
                window?.alphaValue = 1
            }
        } else if !isContentReady {
            isContentReady = true
            window.alphaValue = 1
        }

        let center = NotificationCenter.default
        observers = [
            center.addObserver(
                forName: NSWindow.didMoveNotification,
                object: window,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.coordinator.captureCurrentFrame()
                }
            },
            center.addObserver(
                forName: NSWindow.didEndLiveResizeNotification,
                object: window,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.coordinator.captureCurrentFrame()
                }
            },
        ]
    }

    private func configure(window: NSWindow) {
        window.title = "BrainBar"
        window.minSize = NSSize(width: 760, height: 560)
        window.maxSize = NSSize(width: 1_600, height: 1_200)
        window.isMovableByWindowBackground = true
        window.styleMask.insert(.resizable)
        if let resolvedFrame = BrainBarWindowPlacement.resolvedFrame(
            persistedFrame: BrainBarWindowFrameStore().persistedFrame()
        ) {
            window.setFrame(resolvedFrame, display: true)
        }
    }

    private func removeObservers() {
        observers.forEach(NotificationCenter.default.removeObserver)
        observers.removeAll()
    }
}
