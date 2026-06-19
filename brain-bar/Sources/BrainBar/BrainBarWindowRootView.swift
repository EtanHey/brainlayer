import AppKit
import Combine
import SwiftUI

enum BrainBarPlaceholderCopy {
    static let injectionFeedNotWired = "Injection feed not yet wired in this build."
}

struct BrainBarWindowRootView: View {
    @ObservedObject var runtime: BrainBarRuntime
    private let managesWindowFrame: Bool

    @State private var selectedTab: BrainBarTab = .dashboard
    @State private var hasActivatedInjectionsTab = false
    @State private var hasActivatedGraphTab = false
    @State private var commandBarProvider = BrainBarCommandBarViewModelProvider()
    @StateObject private var windowObserver: BrainBarWindowObserver

    init(runtime: BrainBarRuntime, managesWindowFrame: Bool = true) {
        self.runtime = runtime
        self.managesWindowFrame = managesWindowFrame
        _windowObserver = StateObject(
            wrappedValue: BrainBarWindowObserver(coordinator: runtime.windowCoordinator)
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            BrainBarWindowHeader(
                selectedTab: $selectedTab,
                collector: runtime.collector,
                hotkeyStatus: runtime.hotkeyStatus.statusLine,
                commandBarViewModel: commandBarViewModel
            )

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
            maxWidth: .infinity,
            minHeight: 560,
            idealHeight: 640,
            maxHeight: .infinity
        )
        .opacity(managesWindowFrame ? (windowObserver.isContentReady ? 1 : 0) : 1)
        .background(BrainBarAppBackground())
        .environment(\.colorScheme, .dark)
        .background(windowAttachment)
        .onAppear {
            activate(tab: selectedTab)
            if let action = runtime.requestedQuickAction {
                handleRequestedQuickAction(action)
            }
        }
        .onChange(of: selectedTab) { _, newTab in
            activate(tab: newTab)
        }
        .onChange(of: runtime.database != nil, initial: true) { _, _ in
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
    private var windowAttachment: some View {
        if managesWindowFrame {
            WindowAttachmentView { window in
                windowObserver.attach(window: window)
            }
        }
    }

    @ViewBuilder
    private var dashboardContent: some View {
        if let collector = runtime.collector {
            if collector.lastDataFetchedAt == nil {
                BrainBarLoadingView(title: "BrainBar", subtitle: "Connecting to daemon and loading dashboard data...")
            } else {
                BrainBarDashboardView(
                    collector: collector,
                    hotkeyStatus: runtime.hotkeyStatus.statusLine
                )
            }
        } else {
            BrainBarLoadingView(title: "BrainBar", subtitle: "Opening database and warming the dashboard...")
        }
    }

    @ViewBuilder
    private var injectionsContent: some View {
        if let store = runtime.injectionStore {
            BrainBarInjectionTab(store: store, isActive: selectedTab == .injections && windowObserver.isWindowVisible)
        } else {
            BrainBarLoadingView(title: "Injections", subtitle: BrainBarPlaceholderCopy.injectionFeedNotWired)
        }
    }

    @ViewBuilder
    private var graphContent: some View {
        if let database = runtime.database {
            BrainBarGraphTab(database: database, isActive: selectedTab == .graph && windowObserver.isWindowVisible)
        } else {
            BrainBarLoadingView(title: "Graph", subtitle: "Knowledge graph unavailable.")
        }
    }

    private var commandBarViewModel: QuickCaptureViewModel? {
        commandBarProvider.viewModel(database: runtime.database)
    }

    private func handleRequestedQuickAction(_ action: BrainBarQuickAction) {
        // If the DB isn't ready yet, leave the request in flight and replay
        // when the runtime database readiness token changes.
        guard let vm = commandBarViewModel else { return }
        selectedTab = .dashboard
        vm.setMode(action == .capture ? .capture : .search)
        vm.panelDidAppear()
        runtime.clearQuickActionRequest()
    }

    private func activate(tab: BrainBarTab) {
        switch tab {
        case .dashboard:
            runtime.collector?.requestRefresh(force: true, trigger: .tabSwitch)
        case .injections:
            runtime.ensureInjectionStore()
            hasActivatedInjectionsTab = true
        case .graph:
            hasActivatedGraphTab = true
        }
    }
}

@MainActor
private final class BrainBarCommandBarViewModelProvider {
    private let panelState = QuickCapturePanelState()
    private var currentViewModel: QuickCaptureViewModel?

    func viewModel(database: BrainDatabase?) -> QuickCaptureViewModel? {
        guard let database else { return currentViewModel }
        if currentViewModel == nil {
            currentViewModel = QuickCaptureViewModel(db: database, panelState: panelState)
        }
        return currentViewModel
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

    let collector: StatsCollector?
    let hotkeyStatus: String
    let commandBarViewModel: QuickCaptureViewModel?

    var body: some View {
        VStack(spacing: 10) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 16) {
                    brand
                    Spacer(minLength: 16)
                    sectionPicker(maxWidth: 280)
                    refreshControls
                    BrainBarAppControlMenu()
                    Spacer(minLength: 12)
                    hotkeyLabel
                }

                VStack(alignment: .leading, spacing: 10) {
                    HStack(alignment: .center, spacing: 12) {
                        brand
                        Spacer(minLength: 12)
                        BrainBarAppControlMenu()
                        hotkeyLabel
                            .frame(maxWidth: 180, alignment: .trailing)
                    }

                    HStack(alignment: .center, spacing: 10) {
                        sectionPicker(maxWidth: .infinity)
                        refreshControls
                    }
                }
            }

            BrainBarCommandBar(viewModel: commandBarViewModel)
        }
        .padding(.horizontal, 20)
        .padding(.top, 14)
        .padding(.bottom, 12)
        .background(BrainBarDesignTokens.Glass.primaryMaterial)
        .background(WindowDragHandle())
    }

    private var brand: some View {
        Label("BrainBar", systemImage: "brain")
            .font(.system(size: 18, weight: .semibold))
            .labelStyle(.titleAndIcon)
            .lineLimit(1)
    }

    private var hotkeyLabel: some View {
        Text(hotkeyStatus)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.tail)
    }

    @ViewBuilder
    private var refreshControls: some View {
        if let collector {
            BrainBarHeaderRefreshControls(collector: collector)
        }
    }

    private func sectionPicker(maxWidth: CGFloat) -> some View {
        Picker("Section", selection: $selectedTab) {
            ForEach(BrainBarTab.allCases) { tab in
                Text(tab.title).tag(tab)
            }
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: maxWidth)
        .labelsHidden()
    }
}

private struct BrainBarHeaderRefreshControls: View {
    @ObservedObject private var collector: StatsCollector

    init(collector: StatsCollector) {
        self.collector = collector
    }

    var body: some View {
        Button {
            collector.manualRefresh()
        } label: {
            Label("Refresh now", systemImage: "arrow.clockwise")
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .keyboardShortcut("r", modifiers: .command)
        .disabled(collector.isManualRefreshInProgress)
        .help("Refresh dashboard now")

        if collector.isManualRefreshInProgress {
            ProgressView()
                .controlSize(.small)
                .frame(width: 16, height: 16)
        }
    }
}

private struct BrainBarAppControlMenu: View {
    var body: some View {
        Menu {
            Button("Settings...") {
                BrainBarSettingsActions.openSettingsWindow()
            }
            Divider()
            Button("Restart BrainBar") {
                BrainBarProcessControl.restart()
            }
            Button("Quit BrainBar") {
                BrainBarProcessControl.quit()
            }
        } label: {
            Image(systemName: "power")
                .frame(width: 18, height: 18)
        }
        .menuStyle(.borderlessButton)
        .controlSize(.small)
        .help("Restart or quit BrainBar")
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
    @State private var signalCoverageExpanded = false
    @State private var vectorSignalDetailExpanded = false

    private var flowSummary: DashboardFlowSummary {
        DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats)
    }

    var body: some View {
        GeometryReader { proxy in
            let layout = BrainBarDashboardLayout(containerSize: proxy.size)

            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: layout.sectionSpacing) {
                    overviewCard(layout: layout)
                    freshnessLine
                    pipelinePanel(layout: layout)
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
            BrainBarGlassPanel(
                cornerRadius: layout.panelCornerRadius,
                tint: flowStateTheme.theme.swiftUIColor,
                emphasized: true
            )
        )
        .shadow(color: flowStateTheme.theme.glowSwiftUIColor.opacity(0.16), radius: 30, y: 12)
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
                WrappingPillLayout(spacing: 8, lineSpacing: 8) {
                    overviewBadgeRow
                }

                WrappingPillLayout(spacing: 8, lineSpacing: 8) {
                    overviewBadgeRow
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var overviewCaption: some View {
        Text("Live uses the last \(collector.stats.liveWindowMinutes)m. Trend cards use \(flowSummary.windowLabel.lowercased()).")
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var freshnessLine: some View {
        HStack(spacing: 8) {
            Image(systemName: "clock")
                .font(.system(size: 11, weight: .semibold))
            Text("Data fetched at: \(dataFetchedText)")
                .font(.system(size: 11, weight: .semibold))
                .monospacedDigit()
            if let heartbeatText {
                Text("Heartbeat: \(heartbeatText)")
                    .font(.system(size: 11, weight: .medium))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                if collector.isHeartbeatAheadOfStats {
                    Text("updating...")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 4)
    }

    private var dataFetchedText: String {
        guard let lastDataFetchedAt = collector.lastDataFetchedAt else { return "not yet" }
        return DashboardMetricFormatter.absoluteTimeString(lastDataFetchedAt)
    }

    private var heartbeatText: String? {
        guard let updatedAt = collector.heartbeat.updatedAt else { return nil }
        let type = collector.heartbeat.lastEvent?.type.rawValue ?? "db"
        return "\(type) \(DashboardMetricFormatter.absoluteTimeString(updatedAt))"
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
            repeating: GridItem(.flexible(minimum: 150), spacing: layout.gridSpacing, alignment: .leading),
            count: 2
        )

        return LazyVGrid(columns: columns, spacing: layout.gridSpacing) {
            BrainBarOverviewStat(label: "Chunks", value: "\(collector.stats.chunkCount)", isHero: true)
            BrainBarOverviewStat(label: "Enriched", value: "\(collector.stats.enrichedChunkCount)", isHero: false)
            BrainBarOverviewStat(label: "Backlog", value: "\(collector.stats.pendingEnrichmentCount)", isHero: false)
            BrainBarOverviewStat(label: "Coverage", value: "\(Int(collector.stats.enrichmentPercent.rounded()))%", isHero: false)
        }
    }

    @ViewBuilder
    private func pipelinePanel(layout: BrainBarDashboardLayout) -> some View {
        VStack(alignment: .leading, spacing: layout.gridSpacing) {
            BrainBarSectionLabel("Pipeline")

            if layout.chartColumns == 2 {
                HStack(alignment: .top, spacing: layout.gridSpacing) {
                    VStack(alignment: .leading, spacing: 12) {
                        writesCard(layout: layout)
                        signalCoveragePanel(layout: layout)
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    enrichmentsCard(layout: layout)
                }
            } else {
                VStack(spacing: layout.gridSpacing) {
                    writesCard(layout: layout)
                    signalCoveragePanel(layout: layout)
                    enrichmentsCard(layout: layout)
                }
            }

            queueRail(layout: layout)
            agentPresenceStrip(layout: layout)
        }
        .padding(layout.cardPadding)
        .background(
            BrainBarGlassPanel(cornerRadius: layout.panelCornerRadius, tint: .brainBarAccent)
        )
    }

    private func writesCard(layout: BrainBarDashboardLayout) -> some View {
        BrainBarFlowLaneCard(
            lane: flowSummary.ingress,
            pulseRevision: writePulseRevision,
            compact: layout.compactCards,
            chartHeight: layout.sparklineHeight,
            fetchedAt: collector.lastDataFetchedAt ?? Date(),
            emphasize: true
        )
    }

    private func enrichmentsCard(layout: BrainBarDashboardLayout) -> some View {
        BrainBarFlowLaneCard(
            lane: flowSummary.enrichment,
            pulseRevision: enrichmentPulseRevision,
            compact: layout.compactCards,
            chartHeight: layout.sparklineHeight,
            fetchedAt: collector.lastDataFetchedAt ?? Date(),
            emphasize: true
        )
    }

    private func queueRail(layout: BrainBarDashboardLayout) -> some View {
        BrainBarQueueRail(
            summary: flowSummary.queue,
            coverageText: "\(Int(collector.stats.enrichmentPercent.rounded()))% enriched",
            watcherText: collector.stats.watcherHealth?.summaryText ?? "unknown",
            compact: layout.compactCards
        )
    }

    private func agentPresenceStrip(layout: BrainBarDashboardLayout) -> some View {
        BrainBarAgentPresenceStrip(
            activity: collector.agentActivity,
            compact: layout.compactCards
        )
    }

    private func signalCoveragePanel(layout: BrainBarDashboardLayout) -> some View {
        BrainBarSignalCoveragePanel(
            stats: collector.stats,
            compact: layout.compactCards,
            isExpanded: $signalCoverageExpanded,
            isVectorDetailExpanded: $vectorSignalDetailExpanded
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
                ("Vector", vectorSummary),
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
                ("FTS", ftsSummary),
                ("Trigram", trigramSummary),
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
                    BrainBarTrigramProgress(stats: collector.stats)
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
        .background(
            BrainBarGlassPanel(cornerRadius: layout.panelCornerRadius, tint: .brainBarAccentViolet)
        )
    }

    private var flowStateTheme: BrainBarStateTheme {
        collector.state.stateTheme
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

    private var vectorSummary: String {
        signalSummary(
            indexedCount: collector.stats.vectorIndexedChunkCount,
            backlogCount: collector.stats.vectorBacklogCount,
            coveragePercent: collector.stats.vectorCoveragePercent
        )
    }

    private var ftsSummary: String {
        signalSummary(
            indexedCount: collector.stats.ftsIndexedChunkCount,
            backlogCount: collector.stats.ftsBacklogCount,
            coveragePercent: collector.stats.ftsCoveragePercent
        )
    }

    private var trigramSummary: String {
        signalSummary(
            indexedCount: collector.stats.trigramIndexedChunkCount,
            backlogCount: collector.stats.trigramBacklogCount,
            coveragePercent: collector.stats.trigramCoveragePercent
        )
    }

    private func signalSummary(indexedCount: Int, backlogCount: Int, coveragePercent: Double) -> String {
        "\(indexedCount)/\(collector.stats.signalEligibleChunkCount) · " +
            String(format: "%.0f%%", coveragePercent) +
            " · backlog \(backlogCount)"
    }
}

private struct BrainBarTrigramProgress: View {
    let stats: BrainDatabase.DashboardStats

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Trigram maintenance")
                    .font(.system(size: 11, weight: .semibold))
                Spacer(minLength: 8)
                Text("\(stats.trigramIndexedChunkCount)/\(stats.signalEligibleChunkCount)")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            ProgressView(value: min(stats.trigramCoveragePercent, 100), total: 100)
        }
    }
}

private struct BrainBarSignalCoveragePanel: View {
    let stats: BrainDatabase.DashboardStats
    let compact: Bool
    @Binding var isExpanded: Bool
    @Binding var isVectorDetailExpanded: Bool

    private var signals: [BrainBarSignalCoverage] {
        let green = BrainBarStateTheme.active.theme.swiftUIColor
        let amber = BrainBarStateTheme.degraded.theme.swiftUIColor

        return [
            BrainBarSignalCoverage(
                name: "Vector",
                indexedCount: stats.vectorIndexedChunkCount,
                totalCount: stats.signalEligibleChunkCount,
                backlogCount: stats.vectorBacklogCount,
                coveragePercent: stats.vectorCoveragePercent,
                accentColor: amber,
                showsDetail: true,
                vectorNetDrainRatePerHour: stats.vectorNetDrainRatePerHour,
                vectorBacklogETAHours: stats.vectorBacklogETAHours
            ),
            BrainBarSignalCoverage(
                name: "FTS5",
                indexedCount: stats.ftsIndexedChunkCount,
                totalCount: stats.signalEligibleChunkCount,
                backlogCount: stats.ftsBacklogCount,
                coveragePercent: stats.ftsCoveragePercent,
                accentColor: green,
                showsDetail: false,
                vectorNetDrainRatePerHour: nil,
                vectorBacklogETAHours: nil
            ),
            BrainBarSignalCoverage(
                name: "Trigram",
                indexedCount: stats.trigramIndexedChunkCount,
                totalCount: stats.signalEligibleChunkCount,
                backlogCount: stats.trigramBacklogCount,
                coveragePercent: stats.trigramCoveragePercent,
                accentColor: green,
                showsDetail: false,
                vectorNetDrainRatePerHour: nil,
                vectorBacklogETAHours: nil
            ),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 10 : 12) {
            disclosureButton

            if isExpanded {
                signalBars
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var disclosureButton: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) {
                isExpanded.toggle()
                if !isExpanded {
                    isVectorDetailExpanded = false
                }
            }
        } label: {
            HStack(spacing: 7) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .bold))
                    .frame(width: 12, height: 12)

                Text("see under the hood")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.brainBarTextSecondary)

                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Show retrieval signal coverage")
    }

    @ViewBuilder
    private var signalBars: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: compact ? 8 : 10) {
                ForEach(signals) { signal in
                    signalColumn(for: signal)
                }
            }

            VStack(spacing: 8) {
                ForEach(signals) { signal in
                    signalColumn(for: signal)
                }
            }
        }
    }

    @ViewBuilder
    private func signalColumn(for signal: BrainBarSignalCoverage) -> some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 10) {
            if signal.showsDetail {
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isVectorDetailExpanded.toggle()
                    }
                } label: {
                    BrainBarSignalCoverageRow(
                        signal: signal,
                        compact: compact,
                        isSelected: isVectorDetailExpanded
                    )
                }
                .buttonStyle(.plain)
                .help("Show Vector backlog details")

                if isVectorDetailExpanded {
                    BrainBarVectorSignalDetail(signal: signal, compact: compact)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            } else {
                BrainBarSignalCoverageRow(signal: signal, compact: compact, isSelected: false)
            }
        }
        .frame(minWidth: compact ? 150 : 170, maxWidth: .infinity, alignment: .topLeading)
    }
}

private struct BrainBarSignalCoverage: Identifiable {
    let name: String
    let indexedCount: Int
    let totalCount: Int
    let backlogCount: Int
    let coveragePercent: Double
    let accentColor: Color
    let showsDetail: Bool
    let vectorNetDrainRatePerHour: Double?
    let vectorBacklogETAHours: Double?

    var id: String { name }

    var percentText: String {
        String(format: "%.0f%%", coveragePercent)
    }

    var clampedCoveragePercent: Double {
        min(max(coveragePercent, 0), 100)
    }

    var backlogText: String {
        NumberFormatter.localizedString(from: NSNumber(value: backlogCount), number: .decimal)
    }
}

private struct BrainBarSignalCoverageRow: View {
    let signal: BrainBarSignalCoverage
    let compact: Bool
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 7 : 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(signal.name)
                    .font(.system(size: compact ? 12 : 13, weight: .semibold))
                    .foregroundStyle(Color.brainBarTextPrimary)
                Spacer(minLength: 8)
                Text(signal.percentText)
                    .font(.system(size: compact ? 13 : 15, weight: .bold, design: .rounded))
                    .foregroundStyle(signal.accentColor)
                    .monospacedDigit()
            }

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(signal.accentColor.opacity(0.16))
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [signal.accentColor.opacity(0.68), signal.accentColor],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: proxy.size.width * signal.clampedCoveragePercent / 100)
                }
            }
            .frame(height: 6)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, compact ? 10 : 11)
        .padding(.horizontal, compact ? 11 : 12)
        .background(BrainBarDashboardCardStyle(emphasized: isSelected))
        .overlay {
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .stroke(signal.accentColor.opacity(isSelected ? 0.48 : 0.2), lineWidth: isSelected ? 1.2 : 1)
        }
    }
}

private struct BrainBarVectorSignalDetail: View {
    let signal: BrainBarSignalCoverage
    let compact: Bool

    var body: some View {
        Group {
            if compact {
                VStack(alignment: .leading, spacing: 12) {
                    metrics
                    trend
                }
            } else {
                HStack(alignment: .center, spacing: 18) {
                    metrics
                    Spacer(minLength: 8)
                    trend
                }
            }
        }
        .padding(.vertical, compact ? 12 : 14)
        .padding(.horizontal, compact ? 12 : 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .fill(signal.accentColor.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                        .stroke(signal.accentColor.opacity(0.28), lineWidth: 1)
                )
        )
    }

    private var metrics: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: compact ? 14 : 22) {
                BrainBarSignalDetailMetric(label: "drain", value: drainText, tint: signal.accentColor)
                BrainBarSignalDetailMetric(label: "ETA", value: etaText, tint: signal.accentColor)
                BrainBarSignalDetailMetric(label: "backlog", value: signal.backlogText, tint: Color.brainBarTextPrimary)
            }

            VStack(alignment: .leading, spacing: 10) {
                BrainBarSignalDetailMetric(label: "drain", value: drainText, tint: signal.accentColor)
                BrainBarSignalDetailMetric(label: "ETA", value: etaText, tint: signal.accentColor)
                BrainBarSignalDetailMetric(label: "backlog", value: signal.backlogText, tint: Color.brainBarTextPrimary)
            }
        }
    }

    private var trend: some View {
        Label(isFalling ? "falling" : "waiting", systemImage: isFalling ? "arrow.down.right" : "clock")
            .font(.system(size: 11, weight: .bold))
            .foregroundStyle(signal.accentColor)
            .padding(.vertical, 5)
            .padding(.horizontal, 8)
            .background(Capsule().fill(signal.accentColor.opacity(0.12)))
            .overlay(Capsule().stroke(signal.accentColor.opacity(0.32), lineWidth: 1))
            .help("Vector backlog trend")
    }

    private var isFalling: Bool {
        (signal.vectorNetDrainRatePerHour ?? 0) > 0
    }

    private var drainText: String {
        guard let rate = signal.vectorNetDrainRatePerHour, rate > 0 else { return "n/a" }
        return "~\(formatted(Int(rate.rounded())))/hr"
    }

    private var etaText: String {
        guard let hours = signal.vectorBacklogETAHours, hours.isFinite, hours > 0 else { return "n/a" }
        if hours < 1 {
            return "~\(max(1, Int((hours * 60).rounded())))m"
        }
        if hours < 10 {
            let rounded = (hours * 10).rounded() / 10
            return rounded == rounded.rounded() ? "~\(Int(rounded))h" : String(format: "~%.1fh", rounded)
        }
        return "~\(Int(hours.rounded()))h"
    }

    private func formatted(_ value: Int) -> String {
        NumberFormatter.localizedString(from: NSNumber(value: value), number: .decimal)
    }
}

private struct BrainBarSignalDetailMetric: View {
    let label: String
    let value: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundStyle(tint)
                .monospacedDigit()
                .lineLimit(1)

            Text(label)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(Color.brainBarTextMuted)
                .textCase(.uppercase)
        }
    }
}

private struct BrainBarFlowLaneCard: View {
    let lane: DashboardFlowLane
    let pulseRevision: Int
    let compact: Bool
    let chartHeight: CGFloat
    let fetchedAt: Date
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
                    accentColor: Color.brainBar(nsColor: lane.accentColor)
                )
            }

            BrainBarHeroSparkline(
                label: lane.sparklineLabel,
                values: lane.values,
                secondaryValues: lane.secondaryValues,
                primarySeriesLabel: lane.primarySeriesLabel,
                secondarySeriesLabel: lane.secondarySeriesLabel,
                latestBucketName: lane.latestBucketName,
                accentColor: lane.accentColor,
                secondaryAccentColor: lane.secondaryAccentColor,
                activityWindowMinutes: lane.activityWindowMinutes,
                fetchedAt: fetchedAt,
                pulseRevision: pulseRevision
            )
            .frame(height: chartHeight)

            if let primarySeriesLabel = lane.primarySeriesLabel,
               let secondarySeriesLabel = lane.secondarySeriesLabel,
               let secondaryAccentColor = lane.secondaryAccentColor {
                BrainBarSeriesLegend(
                    primaryLabel: primarySeriesLabel,
                    primaryColor: lane.accentColor,
                    secondaryLabel: secondarySeriesLabel,
                    secondaryColor: secondaryAccentColor
                )
            }

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
    let watcherText: String
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
                BrainBarLaneMetric(label: "Watcher", value: watcherText)
            }

            VStack(alignment: .leading, spacing: 8) {
                BrainBarLaneMetric(label: "Backlog", value: "\(summary.backlogCount)")
                BrainBarLaneMetric(label: "Coverage", value: coverageText)
                BrainBarLaneMetric(label: "Watcher", value: watcherText)
            }
        }
    }

    private var queueColor: Color {
        switch summary.status {
        case .empty, .stable:
            return BrainBarStateTheme.loading.theme.swiftUIColor
        case .draining:
            return BrainBarStateTheme.active.theme.swiftUIColor
        case .growing, .backlogged:
            return BrainBarStateTheme.degraded.theme.swiftUIColor
        case .unavailable:
            return BrainBarStateTheme.error.theme.swiftUIColor
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

private struct BrainBarSeriesLegend: View {
    let primaryLabel: String
    let primaryColor: NSColor
    let secondaryLabel: String
    let secondaryColor: NSColor

    var body: some View {
        HStack(spacing: 12) {
            legendItem(label: primaryLabel, color: primaryColor, dashed: false)
            legendItem(label: secondaryLabel, color: secondaryColor, dashed: true)
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(primaryLabel) and \(secondaryLabel) write series")
    }

    private func legendItem(label: String, color: NSColor, dashed: Bool) -> some View {
        HStack(spacing: 6) {
            Capsule()
                .fill(Color.brainBar(nsColor: color).opacity(dashed ? 0.55 : 0.9))
                .frame(width: dashed ? 14 : 18, height: 3)
                .overlay {
                    if dashed {
                        HStack(spacing: 2) {
                            ForEach(0..<3, id: \.self) { _ in
                                Capsule()
                                    .fill(Color.brainBarGlassSecondary)
                                    .frame(width: 2, height: 3)
                            }
                        }
                    }
                }
            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Color.brainBarTextMuted)
                .lineLimit(1)
        }
        .fixedSize(horizontal: true, vertical: false)
    }
}

struct BrainBarDashboardLayout {
    let chartColumns: Int
    let overviewMetricColumns: Int
    let diagnosticColumns: Int
    let diagnosticItemColumns: Int
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
    let panelCornerRadius: CGFloat

    init(containerSize: CGSize) {
        let compactHeight = containerSize.height < 620
        let compactWidth = containerSize.width < 920

        chartColumns = containerSize.width >= 1_200 ? 2 : 1
        overviewMetricColumns = containerSize.width >= 980 ? 4 : 2
        diagnosticColumns = containerSize.width >= 960 ? 2 : 1
        diagnosticItemColumns = containerSize.width >= 820 ? 2 : 1

        compactCards = compactWidth || compactHeight
        outerPadding = compactCards ? 28 : 48
        sectionSpacing = compactCards ? 22 : 34
        gridSpacing = compactCards ? 18 : 24
        cardPadding = compactCards ? 28 : 44
        overviewTitleFontSize = compactCards ? BrainBarDesignTokens.TypeScale.title : BrainBarDesignTokens.TypeScale.display
        overviewSubtitleFontSize = BrainBarDesignTokens.TypeScale.body
        overviewStatsWidth = compactCards ? 310 : 420
        metricCardMinHeight = compactCards ? 96 : 128
        metricValueFontSize = compactCards ? 48 : BrainBarDesignTokens.TypeScale.hero
        sparklineHeight = compactCards ? 118 : 150
        panelCornerRadius = BrainBarDesignTokens.Radius.xl
    }
}

private struct BrainBarInjectionTab: View {
    let store: InjectionStore
    let isActive: Bool
    @State private var filterText = ""

    var body: some View {
        InjectionFeedView(store: store, filterText: $filterText)
            .padding(16)
            .onAppear { store.start(active: isActive) }
            .onChange(of: isActive) { _, active in
                store.setActive(active)
            }
            .onDisappear { store.setActive(false) }
    }
}

private struct BrainBarGraphTab: View {
    let isActive: Bool
    @StateObject private var viewModel: KGViewModel

    init(database: BrainDatabase, isActive: Bool) {
        self.isActive = isActive
        _viewModel = StateObject(wrappedValue: KGViewModel(database: database))
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            NonWindowDraggableHostingView {
                KGCanvasView(viewModel: viewModel, isActive: isActive)
            }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if viewModel.degradationState.isDegraded {
                DegradationBadge(reason: viewModel.degradationState.reason)
                    .padding(.top, 12)
                    .padding(.trailing, 12)
            }
        }
    }
}

// AIDEV-NOTE: User-facing indicator that a BrainBar surface is reading from a
// degraded source (transient ReadOnly / busy / locked errors from the writer-
// pidfile contention introduced by PR #309 + amplified post-PR #312). Shown as
// an unobtrusive amber pill so the user sees "data may be stale" rather than
// "blank screen" or "warming memory" lingering — per Etan-mandate 2026-05-22:
// "WITHOUT DEGRATION!" (no blank states, but visible when degraded).
struct DegradationBadge: View {
    let reason: String?

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 10, weight: .semibold))
            Text("Degraded")
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .frame(maxWidth: 180, alignment: .leading)
        .background(
            Capsule().fill(BrainBarStateTheme.degraded.theme.swiftUIColor.opacity(0.85))
        )
        .help(reason ?? "Data source temporarily degraded.")
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
                .fill(Color.brainBarGlassSecondary)
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
            .lineLimit(1)
            .truncationMode(.tail)
            .minimumScaleFactor(0.72)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .fixedSize(horizontal: true, vertical: false)
            .background(.white.opacity(0.18), in: Capsule())
    }
}

private struct BrainBarOverviewStat: View {
    let label: String
    let value: String
    let isHero: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: isHero ? 8 : 4) {
            Text(label)
                .font(.system(size: BrainBarDesignTokens.TypeScale.label, weight: .semibold))
                .foregroundStyle(Color.brainBarTextMuted)
            Text(value)
                .font(.system(size: isHero ? BrainBarDesignTokens.TypeScale.hero : BrainBarDesignTokens.TypeScale.title, weight: .semibold, design: .rounded))
                .lineLimit(1)
                .minimumScaleFactor(0.42)
                .monospacedDigit()
                .foregroundStyle(isHero ? Color.brainBarTextPrimary : Color.brainBarTextSecondary)
        }
        .frame(maxWidth: .infinity, minHeight: isHero ? 128 : 62, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .background(
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .fill(Color.brainBarGlassSecondary)
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
                WrappingPillLayout(spacing: 8, lineSpacing: 8) {
                    ForEach(activity.presences, id: \.family) { presence in
                        BrainBarAgentPresencePill(presence: presence)
                    }
                }

                WrappingPillLayout(spacing: 8, lineSpacing: 8) {
                    ForEach(activity.presences, id: \.family) { presence in
                        BrainBarAgentPresencePill(presence: presence)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
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
                .fill(Color.brainBar(nsColor: presence.family.accentColor))
                .frame(width: 8, height: 8)
                .opacity(presence.isActive ? 1 : 0.25)
            Text(presence.family.label)
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
                .truncationMode(.tail)
                .minimumScaleFactor(0.75)
            Text("\(presence.count)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(
                    Capsule()
                        .fill(Color.brainBar(nsColor: presence.family.accentColor).opacity(presence.isActive ? 0.18 : 0.08))
                )
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            Capsule()
                .fill(Color.brainBarGlassSecondary)
        )
        .overlay(
            Capsule()
                .stroke(Color.brainBar(nsColor: presence.family.accentColor).opacity(presence.isActive ? 0.28 : 0.1), lineWidth: 1)
        )
        .fixedSize(horizontal: true, vertical: false)
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
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .fill(Color.brainBarGlassSecondary)
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
                            .brainBarGlassPrimary,
                            .brainBarGlassSecondary,
                        ]
                        : [
                            .brainBarGlassSecondary,
                            .brainBarGlassTertiary,
                        ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.brainBarBorderSoft, lineWidth: 1)
            )
    }
}

private struct BrainBarFlowStatusPill: View {
    let text: String
    let accentColor: Color

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .lineLimit(1)
            .truncationMode(.tail)
            .minimumScaleFactor(0.72)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .fixedSize(horizontal: true, vertical: false)
            .background(accentColor.opacity(0.16), in: Capsule())
            .overlay(
                Capsule()
                    .stroke(accentColor.opacity(0.28), lineWidth: 1)
            )
    }
}

private struct BrainBarHeroSparkline: View {
    let label: String
    let values: [Int]
    let secondaryValues: [Int]
    let primarySeriesLabel: String?
    let secondarySeriesLabel: String?
    let latestBucketName: String
    let accentColor: NSColor
    let secondaryAccentColor: NSColor?
    let activityWindowMinutes: Int
    let fetchedAt: Date
    let pulseRevision: Int

    var body: some View {
        GeometryReader { proxy in
            let renderSize = NSSize(
                width: max(proxy.size.width.rounded(.up), 1),
                height: max(proxy.size.height.rounded(.up), 1)
            )

            SparklineChart(
                presentation: SparklineChartPresentation(
                    label: label,
                    values: values,
                    secondaryValues: secondaryValues,
                    primarySeriesLabel: primarySeriesLabel,
                    secondarySeriesLabel: secondarySeriesLabel,
                    activityWindowMinutes: activityWindowMinutes,
                    latestBucketName: latestBucketName,
                    fetchedAt: fetchedAt
                ),
                accentColor: accentColor,
                secondaryAccentColor: secondaryAccentColor,
                compact: SparklineRenderer.isCompact(size: renderSize)
            )
            .id(pulseRevision)
            .frame(width: proxy.size.width, height: proxy.size.height)
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

private struct BrainBarAppBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [.brainBarBackgroundBase, .brainBarBackgroundAbyss],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            RadialGradient(
                colors: [.brainBarAccent.opacity(0.16), .clear],
                center: UnitPoint(x: 0.16, y: -0.10),
                startRadius: 0,
                endRadius: 760
            )
            RadialGradient(
                colors: [.brainBarAccentViolet.opacity(0.12), .clear],
                center: UnitPoint(x: 0.94, y: 0.06),
                startRadius: 0,
                endRadius: 680
            )
            RadialGradient(
                colors: [BrainBarStateTheme.active.theme.swiftUIColor.opacity(0.07), .clear],
                center: UnitPoint(x: 0.60, y: 1.16),
                startRadius: 0,
                endRadius: 720
            )
        }
        .ignoresSafeArea()
    }
}

private struct BrainBarGlassPanel: View {
    let cornerRadius: CGFloat
    var tint: Color = .brainBarAccent
    var emphasized = false

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(BrainBarDesignTokens.Glass.primaryMaterial)
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                .brainBarGlassPrimary,
                                tint.opacity(emphasized ? 0.18 : 0.08),
                                .brainBarGlassSecondary,
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.brainBarBorderSoft, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .shadow(color: .brainBarBlack.opacity(emphasized ? 0.32 : 0.24), radius: emphasized ? 36 : 24, y: emphasized ? 18 : 12)
    }
}

private struct BrainBarSectionLabel: View {
    let title: String

    init(_ title: String) {
        self.title = title
    }

    var body: some View {
        HStack(spacing: 14) {
            Text(title.uppercased())
                .font(.system(size: BrainBarDesignTokens.TypeScale.label, weight: .bold))
                .tracking(1.6)
                .foregroundStyle(Color.brainBarTextMuted)
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.brainBarBorderSoft, .clear],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .frame(height: 1)
        }
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

    override func mouseDown(with event: NSEvent) {
        window?.performDrag(with: event)
    }
}

private struct NonWindowDraggableHostingView<Content: View>: NSViewRepresentable {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    func makeNSView(context: Context) -> NoWindowDragHostingView<Content> {
        NoWindowDragHostingView(rootView: content)
    }

    func updateNSView(_ nsView: NoWindowDragHostingView<Content>, context: Context) {
        nsView.rootView = content
    }
}

private final class NoWindowDragHostingView<Content: View>: NSHostingView<Content> {
    override var mouseDownCanMoveWindow: Bool {
        false
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
    @Published private(set) var isWindowVisible = true

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
        isWindowVisible = Self.isWindowActuallyVisible(window)

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
            center.addObserver(
                forName: NSWindow.didChangeOcclusionStateNotification,
                object: window,
                queue: .main
            ) { [weak self, weak window] _ in
                Task { @MainActor [weak self, weak window] in
                    self?.isWindowVisible = Self.isWindowActuallyVisible(window)
                }
            },
            center.addObserver(
                forName: NSWindow.willCloseNotification,
                object: window,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.isWindowVisible = false
                }
            },
        ]
    }

    private static func isWindowActuallyVisible(_ window: NSWindow?) -> Bool {
        guard let window else { return false }
        return window.isVisible && window.occlusionState.contains(.visible)
    }

    private func configure(window: NSWindow) {
        window.title = "BrainBar"
        window.minSize = NSSize(width: 760, height: 560)
        window.maxSize = NSSize(width: 1_600, height: 1_200)
        window.isMovable = false
        window.isMovableByWindowBackground = false
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
