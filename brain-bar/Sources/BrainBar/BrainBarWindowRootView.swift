import AppKit
import Combine
import SwiftUI

struct BrainBarWindowRootView: View {
    static let defaultTab: BrainBarTab = .dashboard

    @ObservedObject var runtime: BrainBarRuntime
    private let managesWindowFrame: Bool
    @ObservedObject private var panelState: BrainBarDashboardPanelState

    @State private var selectedTab = BrainBarWindowRootView.defaultTab
    @State private var hasActivatedGraphTab = false
    @State private var commandBarProvider = BrainBarCommandBarViewModelProvider()
    @StateObject private var windowObserver: BrainBarWindowObserver
    @ObservedObject private var retrievalTools = BrainBarRetrievalToolsSettings.shared

    init(runtime: BrainBarRuntime, managesWindowFrame: Bool = true,
         panelState: BrainBarDashboardPanelState = BrainBarDashboardPanelState()) {
        self.runtime = runtime
        self.managesWindowFrame = managesWindowFrame
        self.panelState = panelState
        _windowObserver = StateObject(
            wrappedValue: BrainBarWindowObserver(coordinator: runtime.windowCoordinator)
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            BrainBarWindowHeader(
                collector: runtime.collector,
                hotkeyStatus: runtime.hotkeyStatus.statusLine,
                commandBarViewModel: commandBarViewModel,
                databasePath: runtime.databasePath,
                showRetrievalTools: retrievalTools.isEnabled,
                isShowingGraph: selectedTab == .graph,
                toggleGraph: {
                    selectedTab = selectedTab == .graph ? .dashboard : .graph
                }
            )
            .background(GeometryReader { proxy in
                Color.clear.preference(key: BrainBarHeaderHeightKey.self, value: proxy.size.height)
            })

            ZStack {
                dashboardContent
                    .brainBarTabVisibility(selectedTab == .dashboard)

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
                    isOnActiveTab: selectedTab == .dashboard,
                    panelState: panelState
                )
            }
        }
        .frame(
            minWidth: 760,
            idealWidth: 900,
            maxWidth: .infinity
        )
        .opacity(managesWindowFrame ? (windowObserver.isContentReady ? 1 : 0) : 1)
        .background(BrainBarAppBackground())
        .environment(\.colorScheme, .dark)
        .background(windowAttachment)
        .onPreferenceChange(BrainBarHeaderHeightKey.self) { panelState.headerHeight = $0 }
        .onAppear {
            activate(tab: selectedTab)
            if let action = runtime.requestedQuickAction {
                handleRequestedQuickAction(action)
            }
        }
        .onChange(of: selectedTab) { _, newTab in
            panelState.graphPresented = newTab == .graph
            activate(tab: newTab)
        }
        .onChange(of: retrievalTools.isEnabled) { _, enabled in
            selectedTab = BrainBarRetrievalToolsPolicy.selectedTab(
                selectedTab,
                showRetrievalTools: enabled
            )
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
            BrainBarDashboardContent(
                collector: collector,
                hotkeyStatus: runtime.hotkeyStatus.statusLine,
                dbPath: runtime.databasePath,
                panelState: panelState
            )
        } else {
            BrainBarLoadingView(title: "BrainBar", subtitle: "Opening database and warming the dashboard...")
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
        guard BrainBarRetrievalToolsPolicy.showsCommandBar(showRetrievalTools: retrievalTools.isEnabled) else {
            return nil
        }
        return commandBarProvider.viewModel(database: runtime.database)
    }

    private func handleRequestedQuickAction(_ action: BrainBarQuickAction) {
        guard BrainBarRetrievalToolsPolicy.allowsQuickActions(showRetrievalTools: retrievalTools.isEnabled) else {
            runtime.clearQuickActionRequest()
            return
        }
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
        case .graph:
            hasActivatedGraphTab = true
        }
    }
}

private struct BrainBarHeaderHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

private struct BrainBarDashboardContent: View {
    @ObservedObject var collector: StatsCollector
    @StateObject private var standalonePanelState = BrainBarDashboardPanelState()
    let hotkeyStatus: String
    var dbPath: String? = nil
    var observabilityResult: ObservabilityReadResult? = nil
    var referenceNow: Date? = nil
    var calendar: Calendar = .current
    var locale: Locale = .current
    var panelState: BrainBarDashboardPanelState? = nil

    var body: some View {
        if collector.snapshotFreshnessState.isLoading {
            BrainBarLoadingView(title: "BrainBar", subtitle: "Connecting to daemon and loading dashboard data...")
        } else {
            BrainBarDashboardView(
                collector: collector,
                hotkeyStatus: hotkeyStatus,
                dbPath: dbPath,
                observabilityResult: observabilityResult,
                referenceNow: referenceNow,
                calendar: calendar,
                locale: locale,
                panelState: panelState ?? standalonePanelState
            )
        }
    }
}

enum BrainBarHeroBackupTruth: Sendable, Equatable {
    case measured(ObservabilityBackupStatus)
    case stale(ObservabilityBackupStatus, ageText: String)
    case unavailable(String)
    case checking(String)

    static func derive(
        from result: ObservabilityReadResult,
        now: Date,
        cadence: ObservabilityCadence
    ) -> Self {
        switch result {
        case let .readable(document):
            let snapshot = ObservabilityPresentation.snapshot(
                document: document,
                now: now,
                cadence: cadence
            )
            guard document.backups.state == "measured" else {
                let detail = snapshot.cards.first(where: { $0.title == "Backups" })?.detail
                let reason = document.backups.reason.isEmpty
                    ? "unmeasurable"
                    : "unmeasurable — \(document.backups.reason)"
                return .unavailable(detail ?? reason)
            }
            let status = ObservabilityPresentation.backupStatus(for: document.backups)
            return snapshot.isStale ? .stale(status, ageText: snapshot.ageText) : .measured(status)
        case let .unreadable(reason):
            return reason == "Loading observability data." ? .checking(reason) : .unavailable(reason)
        }
    }
}

enum BrainBarHeroHealthTone: Sendable, Equatable {
    case green
    case amber
    case red
}

struct BrainBarHeroPresentation: Sendable, Equatable {
    let healthTitle = "Health"
    let backupsTitle = "Backups"
    let indexedTitle = "Indexed"
    let healthVerdict: String
    let healthReason: String
    let healthTone: BrainBarHeroHealthTone
    let backupFailureReason: String?
    let dbBackup: ObservabilityStatusLine
    let transcriptBackup: ObservabilityStatusLine
    let indexedInWindow: String
    let totalIndexed: String

    static func derive(
        flow: DashboardFlowSummary,
        stats: DashboardStats,
        backupTruth: BrainBarHeroBackupTruth,
        locale: Locale = .current
    ) -> Self {
        let dbBackup: ObservabilityStatusLine
        let transcriptBackup: ObservabilityStatusLine
        let backupFailure: String?
        let staleBackupAge: String?
        let checkingReason: String?
        switch backupTruth {
        case let .measured(status):
            dbBackup = status.snapshot
            transcriptBackup = status.upload
            backupFailure = status.lines.first(where: { $0.tone == .red })?.text
            staleBackupAge = nil
            checkingReason = nil
        case let .stale(status, ageText):
            dbBackup = status.snapshot
            transcriptBackup = status.upload
            backupFailure = status.lines.first(where: { $0.tone == .red })?.text
            staleBackupAge = ageText
            checkingReason = nil
        case let .unavailable(reason):
            dbBackup = .init(text: reason, tone: .neutral)
            transcriptBackup = .init(text: reason, tone: .neutral)
            backupFailure = reason
            staleBackupAge = nil
            checkingReason = nil
        case let .checking(reason):
            dbBackup = .init(text: "Checking DB snapshot status", tone: .neutral)
            transcriptBackup = .init(text: "Checking transcript backup status", tone: .neutral)
            backupFailure = nil
            staleBackupAge = nil
            checkingReason = reason
        }

        let health: (String, String, BrainBarHeroHealthTone)
        switch flow.watcherFlowState {
        case .offline:
            health = ("Needs attention", "Watcher is offline.", .red)
        case .stalled:
            health = ("Needs attention", "Watcher is running, but pending work is not moving.", .red)
        default:
            if flow.ingress.status == .unavailable {
                health = ("Needs attention", "Ingest health is unavailable.", .red)
            } else if let backupFailure {
                health = ("Needs attention", backupFailure, .red)
            } else if let checkingReason {
                health = ("Checking health", checkingReason, .amber)
            } else if let staleBackupAge {
                health = (
                    "Check health",
                    "Backup status is \(staleBackupAge); waiting for a fresh observability check.",
                    .amber
                )
            } else {
                switch flow.watcherFlowState {
                case .unknown:
                    health = ("Check health", "Watcher health is unknown.", .amber)
                case .runningFlowUnverified:
                    health = ("Check health", "Watcher is running, but flow could not be verified.", .amber)
                case .flowing:
                    health = ("Healthy", "Watcher is flowing; DB and transcript backups are verified.", .green)
                case .runningNoRecentFlow:
                    health = ("Healthy", "Watcher is running with no recent work; DB and transcript backups are verified.", .green)
                case .offline, .stalled:
                    health = ("Needs attention", "Watcher health needs attention.", .red)
                }
            }
        }

        let window = flow.windowLabel.lowercased()
        return Self(
            healthVerdict: health.0,
            healthReason: health.1,
            healthTone: health.2,
            backupFailureReason: backupFailure,
            dbBackup: dbBackup,
            transcriptBackup: transcriptBackup,
            indexedInWindow: "\(DashboardMetricFormatter.integerString(stats.recentWriteCount, locale: locale)) chunk rows indexed in \(window)",
            totalIndexed: "\(DashboardMetricFormatter.integerString(stats.chunkCount, locale: locale)) chunk rows total"
        )
    }
}

enum BrainBarOnePageComposition {
    static let visibleSectionIDs = ["status", "backups", "memory", "ingest", "details"]
    static let primaryTileCount = 2
    static let primaryTilesHaveEqualHeight = false
    static let detailsExpandedByDefault = false
}

enum BrainBarIngestBandLayout {
    static let plotHeight: CGFloat = 72

    static func chartSizes(containerWidth: CGFloat) -> [NSSize] {
        let compact = containerWidth < 920
        let outerPadding: CGFloat = compact ? 16 : 24
        let bandPadding: CGFloat = 32
        let available = max(containerWidth - (outerPadding * 2) - bandPadding, 1)
        let chartWidth = compact ? available : max((available - 32) / 3, 240)
        return Array(repeating: NSSize(width: chartWidth, height: plotHeight), count: 3)
    }
}

private struct BrainBarIngestBarChart: View {
    let values: [Int]
    let isAvailable: Bool
    let timeframe: PipelineTimeframe
    let accentColor: Color

    private var presentation: SparklineChartPresentation {
        SparklineChartPresentation(
            label: "Ingest",
            values: values,
            activityWindowMinutes: timeframe.windowMinutes,
            latestBucketName: "Current",
            fetchedAt: .distantPast
        )
    }

    private var xLabels: [String] {
        switch timeframe {
        case .live: ["−60m", "−30m", "now"]
        case .threeHour: ["−3h", "−90m", "now"]
        case .day: ["−24h", "−12h", "now"]
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 5) {
            VStack(alignment: .trailing, spacing: 0) {
                Text(DashboardMetricFormatter.axisTickString(presentation.axisMax))
                Spacer(minLength: 0)
                Text("0")
            }
            .font(.system(size: 9))
            .monospacedDigit()
            .foregroundStyle(Color.brainBarTextSecondary.opacity(0.8))
            .frame(width: 24, height: BrainBarIngestBandLayout.plotHeight, alignment: .trailing)

            VStack(spacing: 3) {
                ZStack {
                    VStack(spacing: 0) {
                        Rectangle().fill(Color.brainBarBorderSoft).frame(height: 0.5)
                        Spacer(minLength: 0)
                        Rectangle().fill(Color.brainBarBorderSoft).frame(height: 0.5)
                    }
                    if isAvailable {
                        GeometryReader { proxy in
                            let maxValue = max(presentation.axisMax, 1)
                            HStack(alignment: .bottom, spacing: 1) {
                                ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                                    let isPartial = index == values.indices.last
                                    let barHeight = max(
                                        value == 0 ? 0 : 1,
                                        CGFloat(value) / CGFloat(maxValue) * proxy.size.height
                                    )
                                    Rectangle()
                                        .fill(accentColor.opacity(isPartial ? 0.4 : 0.85))
                                        .frame(maxWidth: .infinity, minHeight: barHeight, maxHeight: barHeight)
                                        .overlay(alignment: .top) {
                                            if index == values.index(before: values.endIndex), values.count > 1 {
                                                Circle()
                                                    .fill(accentColor)
                                                    .frame(width: 5, height: 5)
                                                    .offset(y: -3)
                                            }
                                        }
                                        .help(isPartial ? "partial" : "\(value) chunk rows")
                                }
                            }
                        }
                    } else {
                        Text("Evidence unavailable")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Color.orange)
                    }
                }
                .frame(height: BrainBarIngestBandLayout.plotHeight)

                HStack {
                    Text(xLabels[0])
                    Spacer(minLength: 0)
                    Text(xLabels[1])
                    Spacer(minLength: 0)
                    Text(xLabels[2])
                }
                .font(.system(size: 9))
                .monospacedDigit()
                .foregroundStyle(Color.brainBarTextSecondary.opacity(0.8))
            }
        }
        .frame(minWidth: 240)
    }
}

enum BrainBarOnePageStatusTone: Sendable, Equatable {
    case green
    case amber
    case neutral
}

struct BrainBarOnePageStatus: Sendable, Equatable {
    let headline: String
    let reason: String?
    let tone: BrainBarOnePageStatusTone
}

struct BrainBarOnePagePresentation: Sendable, Equatable {
    let status: BrainBarOnePageStatus
    let backupLines: [ObservabilityStatusLine]
    let totalIndexedChunks: Int?
    let indexedToday: Int?
    let indexedTodayUnavailableText: String?
    let agentWritesCount: Int?
    let agentWritesWindowHours: Int?
    let agentWritesText: String

    static func derive(
        snapshotFreshness: SnapshotFreshnessState,
        hero: BrainBarHeroPresentation,
        observability: ObservabilityReadResult,
        stats: DashboardStats,
        agentActivity: AgentActivitySnapshot,
        now: Date,
        calendar: Calendar = .current,
        locale: Locale = .current,
        observabilityCadence: ObservabilityCadence = ObservabilityReader.installedHealthCheckCadence
    ) -> Self {
        let snapshotAttentionCount: Int = switch snapshotFreshness {
        case .stale, .error: 1
        case .loading, .live: 0
        }
        let snapshotIsLive: Bool = if case .live = snapshotFreshness { true } else { false }
        let otherLiveHeroAttention = snapshotIsLive
            && hero.healthTone != .green
            && hero.healthReason != hero.backupFailureReason
        let attentionCount =
            (hero.backupFailureReason == nil ? 0 : 1)
            + (otherLiveHeroAttention ? 1 : 0)
            + (agentActivity.isMeasured ? 0 : 1)
            + snapshotAttentionCount
        let attentionHeadline = attentionCount == 1
            ? "1 thing needs you"
            : "\(attentionCount) things need you"

        let status: BrainBarOnePageStatus
        if case .unreadable("Loading observability data.") = observability {
            status = .init(headline: "Checking…", reason: nil, tone: .neutral)
        } else if let backupFailureReason = hero.backupFailureReason {
            status = .init(
                headline: attentionHeadline,
                reason: backupFailureReason,
                tone: .amber
            )
        } else {
            switch snapshotFreshness {
            case .loading:
                status = .init(headline: "Checking…", reason: nil, tone: .neutral)
            case let .stale(ageSeconds):
                status = .init(
                    headline: attentionHeadline,
                    reason: "Dashboard data is \(ageText(ageSeconds)) old.",
                    tone: .amber
                )
            case .error:
                status = .init(
                    headline: attentionHeadline,
                    reason: "Dashboard data could not refresh.",
                    tone: .amber
                )
            case .live:
                if hero.healthTone != .green {
                    status = .init(
                        headline: attentionHeadline,
                        reason: hero.healthReason,
                        tone: .amber
                    )
                } else if !agentActivity.isMeasured {
                    status = .init(
                        headline: attentionHeadline,
                        reason: "Agent activity could not be measured.",
                        tone: .amber
                    )
                } else {
                    status = .init(headline: "All good", reason: nil, tone: .green)
                }
            }
        }

        let backupLines: [ObservabilityStatusLine]
        let totalIndexedChunks: Int?
        let indexedToday: Int?
        let indexedTodayUnavailableText: String?
        if case let .readable(document) = observability, document.backups.state == "measured" {
            let snapshot = document.backups.dbSnapshot
            let upload = document.backups.lastVerifiedUpload
            backupLines = [
                .init(
                    text: "Database · Drive · \(snapshot.flatMap { $0.verified ? backupMoment($0.lastAt, now: now, calendar: calendar, locale: locale) : nil } ?? "no verified copy")",
                    tone: snapshot?.verified == true ? .green : .red
                ),
                .init(
                    text: "Transcripts · Drive · \(upload.flatMap { $0.verified ? backupMoment($0.at, now: now, calendar: calendar, locale: locale) : nil } ?? "no verified copy")",
                    tone: upload?.verified == true ? .green : .red
                ),
            ]
        } else {
            backupLines = [
                .init(text: "Database · Drive · status unavailable", tone: .red),
                .init(text: "Transcripts · Drive · status unavailable", tone: .red),
            ]
        }

        if case let .readable(document) = observability, document.stores.state == "measured" {
            totalIndexedChunks = document.stores.totalChunks ?? stats.chunkCount
            let midnight = calendar.startOfDay(for: now)
            let trust = generatedAtTrust(
                document,
                now: now,
                cadence: observabilityCadence,
                sameDayBoundary: midnight,
                calendar: calendar,
                locale: locale
            )
            if case let .untrustworthy(reason) = trust {
                indexedToday = nil
                indexedTodayUnavailableText = "Indexed today unavailable: \(reason)"
            } else if let buckets = document.stores.inWindow?.byHour {
                let todayCount = buckets
                    .filter { $0.hour >= midnight && $0.hour <= now }
                    .reduce(Int?.some(0)) { total, bucket in
                        guard let total else { return nil }
                        let (sum, overflow) = total.addingReportingOverflow(bucket.count)
                        return overflow ? nil : sum
                    }
                if let todayCount {
                    indexedToday = todayCount
                    indexedTodayUnavailableText = nil
                } else {
                    indexedToday = nil
                    indexedTodayUnavailableText = "Indexed today unavailable: hourly observability count overflow"
                }
            } else {
                indexedToday = nil
                indexedTodayUnavailableText = "Indexed today unavailable: hourly observability missing"
            }
        } else {
            totalIndexedChunks = stats.chunkCount
            indexedToday = nil
            if case let .unreadable(reason) = observability {
                indexedTodayUnavailableText = "Indexed today unavailable: \(reason)"
            } else {
                indexedTodayUnavailableText = "Indexed today unavailable: observability unmeasurable"
            }
        }

        let agentWritesCount: Int?
        let agentWritesWindowHours: Int?
        let agentWritesText: String
        if case let .readable(document) = observability {
            let trust = generatedAtTrust(
                document,
                now: now,
                cadence: observabilityCadence,
                sameDayBoundary: nil,
                calendar: calendar,
                locale: locale
            )
            if case let .untrustworthy(reason) = trust {
                agentWritesCount = nil
                agentWritesWindowHours = nil
                agentWritesText = "brain_store writes unavailable: \(reason)"
            } else if document.emitters.state == "measured" {
                if let mcp = document.emitters.byEmitter?.first(where: { $0.emitter == "mcp" }) {
                    agentWritesCount = mcp.countInWindow
                    agentWritesWindowHours = document.windowHours
                    agentWritesText = "\(DashboardMetricFormatter.integerString(mcp.countInWindow, locale: locale)) writes via brain_store in \(document.windowHours) h"
                } else {
                    agentWritesCount = nil
                    agentWritesWindowHours = nil
                    agentWritesText = "brain_store writes unavailable: no MCP count in observability"
                }
            } else {
                let reason = document.emitters.reason.isEmpty ? "emitter measurement unavailable" : document.emitters.reason
                agentWritesCount = nil
                agentWritesWindowHours = nil
                agentWritesText = "brain_store writes unavailable: \(reason)"
            }
        } else if case let .unreadable(reason) = observability {
            agentWritesCount = nil
            agentWritesWindowHours = nil
            agentWritesText = "brain_store writes unavailable: \(reason)"
        } else {
            agentWritesCount = nil
            agentWritesWindowHours = nil
            agentWritesText = "brain_store writes unavailable: observability document unreadable"
        }

        return Self(
            status: status,
            backupLines: backupLines,
            totalIndexedChunks: totalIndexedChunks,
            indexedToday: indexedToday,
            indexedTodayUnavailableText: indexedTodayUnavailableText,
            agentWritesCount: agentWritesCount,
            agentWritesWindowHours: agentWritesWindowHours,
            agentWritesText: agentWritesText
        )
    }

    private enum GeneratedAtTrust: Equatable {
        case trustworthy
        case untrustworthy(String)
    }

    private static func generatedAtTrust(
        _ document: ObservabilityDocument,
        now: Date,
        cadence: ObservabilityCadence,
        sameDayBoundary: Date?,
        calendar: Calendar,
        locale: Locale
    ) -> GeneratedAtTrust {
        let generatedAt = document.generatedAt
        if generatedAt <= Date(timeIntervalSince1970: 0) {
            return .untrustworthy("observability generated_at is zero or epoch sentinel")
        }
        if generatedAt > now {
            return .untrustworthy("observability generated_at is in the future")
        }
        if now.timeIntervalSince(generatedAt) > cadence.interval * 2 {
            return .untrustworthy("observability as of \(shortTime(generatedAt, calendar: calendar, locale: locale))")
        }
        if let sameDayBoundary, generatedAt < sameDayBoundary {
            return .untrustworthy("observability as of \(shortTime(generatedAt, calendar: calendar, locale: locale))")
        }
        return .trustworthy
    }

    private static func shortTime(_ date: Date, calendar: Calendar, locale: Locale) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = locale
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
    }

    private static func ageText(_ seconds: Int) -> String {
        if seconds < 60 { return "\(seconds) s" }
        if seconds < 3_600 { return "\(seconds / 60) min" }
        return "\(seconds / 3_600) h"
    }

    private static func backupMoment(_ date: Date, now: Date, calendar: Calendar, locale: Locale) -> String {
        if calendar.isDate(date, inSameDayAs: now) {
            return "last good today \(shortTime(date, calendar: calendar, locale: locale))"
        }
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = locale
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "MMM d 'at' HH:mm"
        return "last good \(formatter.string(from: date))"
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
    let collector: StatsCollector?
    let hotkeyStatus: String
    let commandBarViewModel: QuickCaptureViewModel?
    let databasePath: String?
    let showRetrievalTools: Bool
    let isShowingGraph: Bool
    let toggleGraph: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            HStack(alignment: .center, spacing: 12) {
                brand
                Spacer(minLength: 12)
                refreshControls
                if showRetrievalTools {
                    Button(action: toggleGraph) {
                        Label(isShowingGraph ? "Dashboard" : "Knowledge Graph", systemImage: isShowingGraph ? "gauge" : "point.3.connected.trianglepath.dotted")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                BrainBarAppControlMenu(databasePath: databasePath)
            }

            if !isShowingGraph,
               BrainBarRetrievalToolsPolicy.showsCommandBar(showRetrievalTools: showRetrievalTools) {
                BrainBarCommandBar(viewModel: commandBarViewModel)
            }
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
            .accessibilityIdentifier("brainbar.shell.brand")
    }

    @ViewBuilder
    private var refreshControls: some View {
        if let collector {
            BrainBarHeaderRefreshControls(collector: collector)
        }
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
    let databasePath: String?
    @State private var showRestartConfirmation = false
    @State private var showQuitConfirmation = false

    var body: some View {
        Menu {
            Button("Settings...") {
                BrainBarSettingsActions.openSettingsWindow(databasePath: databasePath)
            }
            Divider()
            Button("Restart BrainBar") {
                showRestartConfirmation = true
            }
            Button("Quit BrainBar") {
                showQuitConfirmation = true
            }
        } label: {
            Image(systemName: "power")
                .frame(width: 18, height: 18)
        }
        .menuStyle(.borderlessButton)
        .controlSize(.small)
        .help("Restart or quit BrainBar")
        .accessibilityIdentifier("brainbar.shell.power")
        .confirmationDialog(
            "Restart BrainBar?",
            isPresented: $showRestartConfirmation,
            titleVisibility: .visible
        ) {
            Button("Restart BrainBar", role: .destructive) {
                BrainBarProcessControl.restart()
            }
        } message: {
            Text("The menu bar app will close and relaunch.")
        }
        .confirmationDialog(
            "Quit BrainBar?",
            isPresented: $showQuitConfirmation,
            titleVisibility: .visible
        ) {
            Button("Quit BrainBar", role: .destructive) {
                BrainBarProcessControl.quit()
            }
        } message: {
            Text("Quick capture and dashboard access will be unavailable until BrainBar is opened again.")
        }
    }
}

private struct BrainBarDashboardView: View {
    @ObservedObject var collector: StatsCollector
    let hotkeyStatus: String
    var dbPath: String? = nil
    var observabilityResult: ObservabilityReadResult? = nil
    var referenceNow: Date? = nil
    var calendar: Calendar = .current
    var locale: Locale = .current
    @ObservedObject var panelState: BrainBarDashboardPanelState

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var previousAllCommitBuckets: [Int] = []
    @State private var previousWriteBuckets: [Int] = []
    @State private var previousWatcherBuckets: [Int] = []
    @State private var allCommitPulseRevision = 0
    @State private var writePulseRevision = 0
    @State private var watcherPulseRevision = 0
    @State private var selectedTimeframe: PipelineTimeframe = .live
    @State private var vectorSignalDetailExpanded = false
    @State private var vectorSignalRootFrame: CGRect = .zero
    @State private var liveObservabilityResult: ObservabilityReadResult = .unreadable("Loading observability data.")
    private let observabilityCadence = ObservabilityReader.installedHealthCheckCadence
    @State private var vectorDetailHeight: CGFloat = 0
    @State private var ingestHelpPresented = false

    private var pipelineStats: BrainDatabase.DashboardStats {
        guard selectedTimeframe != .live,
              let buckets = collector.windowedBuckets,
              collector.windowedBucketsWindowMinutes == selectedTimeframe.windowMinutes else {
            return collector.stats
        }
        return collector.stats.withWindowedPipelineBuckets(buckets)
    }

    private var displayedTimeframe: PipelineTimeframe {
        PipelineTimeframe.truthfulDisplay(
            selected: selectedTimeframe,
            loadedWindowMinutes: collector.windowedBucketsWindowMinutes
        )
    }

    private var pipelineFlowSummary: DashboardFlowSummary {
        DashboardFlowSummary.derive(daemon: collector.daemon, stats: pipelineStats, now: currentNow)
    }

    private var flowSummary: DashboardFlowSummary {
        DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: currentNow)
    }

    private var vectorSignal: BrainBarSignalCoverage {
        BrainBarSignalCoverage(
            name: "Vector",
            indexedCount: collector.stats.vectorIndexedChunkCount,
            totalCount: collector.stats.signalEligibleChunkCount,
            backlogCount: collector.stats.vectorBacklogCount,
            coveragePercent: collector.stats.vectorCoveragePercent,
            isAvailable: collector.stats.signalCoverageIsAvailable,
            accentColor: .brainBarSignalVector,
            showsDetail: true,
            vectorNetDrainRatePerHour: collector.stats.vectorNetDrainRatePerHour,
            vectorBacklogETAHours: collector.stats.vectorBacklogETAHours
        )
    }

    private var effectiveObservabilityResult: ObservabilityReadResult {
        if let observabilityResult { return observabilityResult }
        guard dbPath != nil else { return .unreadable("Database path unavailable.") }
        return liveObservabilityResult
    }

    private var heroPresentation: BrainBarHeroPresentation {
        let backupTruth = BrainBarHeroBackupTruth.derive(
            from: effectiveObservabilityResult,
            now: currentNow,
            cadence: observabilityCadence
        )
        return BrainBarHeroPresentation.derive(
            flow: flowSummary,
            stats: collector.stats,
            backupTruth: backupTruth,
            locale: locale
        )
    }

    private var currentNow: Date { referenceNow ?? Date() }

    private var onePagePresentation: BrainBarOnePagePresentation {
        BrainBarOnePagePresentation.derive(
            snapshotFreshness: collector.snapshotFreshnessState,
            hero: heroPresentation,
            observability: effectiveObservabilityResult,
            stats: collector.stats,
            agentActivity: collector.agentActivity,
            now: currentNow,
            calendar: calendar,
            locale: locale
        )
    }

    var body: some View {
        GeometryReader { proxy in
            let layout = BrainBarDashboardLayout(containerSize: proxy.size)

            ZStack(alignment: .topLeading) {
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: layout.sectionSpacing) {
                        statusStrip
                        summaryTiles(layout: layout)
                        ingestBand(layout: layout)
                        diagnostics(layout: layout)
                    }
                    .padding(layout.outerPadding)
                    .frame(maxWidth: layout.maxContentWidth, alignment: .topLeading)
                    .frame(maxWidth: .infinity, alignment: .top)
                    .focusSection()
                    .background(GeometryReader { proxy in
                        Color.clear.preference(key: BrainBarDashboardHeightKey.self, value: proxy.size.height)
                    })
                    .background(
                        BrainBarDashboardScrollResetter(
                            disclosureState: BrainBarDashboardDisclosureState(
                                detailsExpanded: panelState.detailsExpanded,
                                signalCoverageExpanded: panelState.signalCoverageExpanded
                            )
                        )
                            .frame(width: 0, height: 0)
                    )
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .accessibilityIdentifier("brainbar.dashboard.scroll")
            }
            .coordinateSpace(name: BrainBarVectorSignalCoordinateSpace.root)
            .onPreferenceChange(BrainBarVectorSignalRootFrameKey.self) { frame in
                MainActor.assumeIsolated {
                    vectorSignalRootFrame = frame
                }
            }
            .overlay(alignment: .topLeading) {
                // Gate on the PARENT too: nothing resets signalCoverageExpanded or
                // vectorSignalDetailExpanded when Details collapses, so without this the
                // backlog popover floats over a closed section.
                if panelState.detailsExpanded, panelState.signalCoverageExpanded, vectorSignalDetailExpanded,
                   vectorSignalRootFrame != .zero {
                    BrainBarVectorSignalDetail(signal: vectorSignal, compact: layout.compactCards)
                        .frame(width: vectorDetailWidth(layout: layout), alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .background(
                            GeometryReader { geo in
                                Color.clear.preference(
                                    key: BrainBarVectorDetailHeightKey.self,
                                    value: geo.size.height
                                )
                            }
                        )
                        .offset(
                            x: vectorDetailXOffset(layout: layout),
                            y: vectorDetailYOffset(layout: layout, containerHeight: proxy.size.height)
                        )
                        .shadow(color: .brainBarBlack.opacity(0.55), radius: 22, y: 12)
                        .shadow(color: .brainBarBlack.opacity(0.30), radius: 6, y: 2)
                        .transition(vectorDetailTransition)
                        .zIndex(vectorSignalDetailExpanded ? 30 : 0)
                }
            }
            .onPreferenceChange(BrainBarVectorDetailHeightKey.self) { height in
                MainActor.assumeIsolated {
                    if height > 0 {
                        vectorDetailHeight = height
                    }
                }
            }
        }
        .onPreferenceChange(BrainBarDashboardHeightKey.self) { height in
            panelState.dashboardHeight = height
        }
        .onAppear {
            previousAllCommitBuckets = collector.stats.recentActivityBuckets
            previousWriteBuckets = collector.stats.recentAgentWriteBuckets
            previousWatcherBuckets = collector.stats.recentWatcherWriteBuckets
        }
        .onChange(of: collector.stats.recentActivityBuckets) { _, newBuckets in
            if BrainBarPipelinePulseGate.shouldPulse(
                previous: previousAllCommitBuckets,
                current: newBuckets,
                timeframe: selectedTimeframe
            ) {
                allCommitPulseRevision += 1
            }
            previousAllCommitBuckets = newBuckets
        }
        .onChange(of: collector.stats.recentAgentWriteBuckets) { _, newBuckets in
            if BrainBarPipelinePulseGate.shouldPulse(
                previous: previousWriteBuckets,
                current: newBuckets,
                timeframe: selectedTimeframe
            ) {
                writePulseRevision += 1
            }
            previousWriteBuckets = newBuckets
        }
        .onChange(of: collector.stats.recentWatcherWriteBuckets) { _, newBuckets in
            if BrainBarPipelinePulseGate.shouldPulse(
                previous: previousWatcherBuckets,
                current: newBuckets,
                timeframe: selectedTimeframe
            ) {
                watcherPulseRevision += 1
            }
            previousWatcherBuckets = newBuckets
        }
        .onChange(of: selectedTimeframe) { _, timeframe in
            collector.selectTimeframe(
                windowMinutes: timeframe.windowMinutes,
                isLive: timeframe == .live
            )
        }
        .task(id: dbPath) {
            guard observabilityResult == nil, let dbPath else { return }
            let url = ObservabilityReader.url(dbPath: dbPath)
            for await next in ObservabilityLiveView.Reader.watch(url: url) {
                liveObservabilityResult = next
            }
        }
    }

    private var statusStrip: some View {
        let status = onePagePresentation.status
        let statusColor: Color = switch status.tone {
        case .green: .green
        case .amber: .orange
        case .neutral: .brainBarTextSecondary
        }
        return HStack(spacing: 9) {
            Circle()
                .fill(statusColor)
                .frame(width: 9, height: 9)
            if let reason = status.reason {
                Text(reason)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Spacer(minLength: 8)
                Text(status.headline)
                    .font(.system(size: 11, weight: .semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(statusColor.opacity(0.16)))
            } else {
                Text(status.headline)
                    .font(.system(size: 13, weight: .semibold))
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            BrainBarGlassPanel(
                cornerRadius: 12,
                tint: statusColor
            )
        )
        .accessibilityIdentifier("brainbar.dashboard.status")
    }

    @ViewBuilder
    private func summaryTiles(layout: BrainBarDashboardLayout) -> some View {
        Group {
            if layout.compactCards {
                VStack(spacing: layout.gridSpacing) {
                    backupTile
                    todayTile
                }
            } else {
                HStack(alignment: .top, spacing: layout.gridSpacing) {
                    backupTile
                    todayTile
                }
            }
        }
    }

    private var backupTile: some View {
        summaryTile(title: "Backups", identifier: "backups") {
            ObservabilityStatusRows(lines: onePageBackupLines, textColor: .brainBarTextPrimary)
                .font(.system(size: 13, weight: .regular))
        }
    }

    private var todayTile: some View {
        let counts = onePagePresentation
        return summaryTile(title: "Today", identifier: "memory") {
            VStack(alignment: .leading, spacing: 5) {
                if let indexedToday = counts.indexedToday {
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        Text(DashboardMetricFormatter.integerString(indexedToday, locale: locale))
                            .font(.system(size: 28, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                        Text("indexed today")
                            .font(.system(size: 13))
                    }
                } else {
                    Text("—")
                        .font(.system(size: 28, weight: .semibold, design: .rounded))
                    Text(todayUnavailableSummary(counts.indexedTodayUnavailableText))
                        .font(.system(size: 11))
                        .foregroundStyle(Color.orange)
                        .lineLimit(1)
                }
                // Gated on agentWritesCount alone: a missing indexedToday must not hide a
                // MEASURED brain_store count. One unknown never erases a known.
                if let writes = counts.agentWritesCount {
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        Text(DashboardMetricFormatter.integerString(writes, locale: locale))
                            .font(.system(size: 20, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                        Text("brain_store writes (\(counts.agentWritesWindowHours ?? 24) h)")
                            .font(.system(size: 13))
                    }
                } else {
                    Text(counts.agentWritesText)
                        .font(.system(size: 11))
                        .foregroundStyle(Color.orange)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 3)
            if let total = counts.totalIndexedChunks {
                Text("\(DashboardMetricFormatter.integerString(total, locale: locale)) total")
                    .font(.system(size: 11))
                    .monospacedDigit()
                    .foregroundStyle(Color.brainBarTextSecondary)
            }
        }
    }

    private func todayUnavailableSummary(_ text: String?) -> String {
        guard let text else { return "Unavailable" }
        return text.replacingOccurrences(of: "Indexed today unavailable: ", with: "")
    }

    private func ingestBand(layout: BrainBarDashboardLayout) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Ingest")
                        .font(.system(size: 13, weight: .semibold))
                    Text("Each chart has its own scale")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.brainBarTextSecondary)
                }
                Spacer(minLength: 8)
                BrainBarSharedTimeframeSelector(
                selection: Binding(
                    get: { displayedTimeframe },
                    set: {
                        if selectedTimeframe == $0 {
                            collector.selectTimeframe(
                                windowMinutes: $0.windowMinutes,
                                isLive: $0 == .live
                            )
                        } else {
                            selectedTimeframe = $0
                        }
                    }
                ),
                isLoading: collector.isWindowedBucketsLoading,
                loadError: collector.windowedBucketsError
            )
                BrainBarKeyboardFocusButton {
                    ingestHelpPresented.toggle()
                } label: {
                    Image(systemName: "questionmark.circle")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 24, height: 24)
                }
                .help("How ingest charts are measured")
                .popover(isPresented: $ingestHelpPresented) {
                    Text("Source-time charts count chunk rows. Watcher counts unique chunk IDs by ingest time. The final bucket is partial.")
                        .font(.system(size: 11))
                        .padding(12)
                        .frame(width: 260)
                }
            }

            if layout.compactCards {
                VStack(spacing: 14) {
                    ingestSeriesChart(.allCommits)
                    ingestSeriesChart(.agentStores)
                    ingestSeriesChart(.jsonlWatcher)
                }
            } else {
                HStack(alignment: .top, spacing: 16) {
                    ingestSeriesChart(.allCommits)
                    ingestSeriesChart(.agentStores)
                    ingestSeriesChart(.jsonlWatcher)
                }
            }
        }
        .padding(16)
        .background(BrainBarDashboardCardStyle(emphasized: true))
        .accessibilityIdentifier("brainbar.dashboard.tile.ingest")
    }

    private func ingestSeriesChart(_ series: PipelineSeries) -> some View {
        let lane = pipelineFlowSummary.lane(for: series)
        let disclosure = BrainBarDashboardChartDisclosure(
            series: series,
            lane: lane,
            timeframe: displayedTimeframe
        )
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(ingestSeriesTitle(series))
                    .font(.system(size: 11))
                    .foregroundStyle(Color.brainBarTextSecondary)
                Spacer(minLength: 4)
                Text(DashboardMetricFormatter.integerString(lane.values.reduce(0, +), locale: locale))
                    .font(.system(size: 13, weight: .semibold))
                    .monospacedDigit()
            }
            BrainBarIngestBarChart(
                values: lane.values,
                isAvailable: lane.status != .unavailable,
                timeframe: displayedTimeframe,
                accentColor: Color.brainBar(nsColor: lane.accentColor)
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityIdentifier(disclosure.accessibilityIdentifier)
        .accessibilityLabel(disclosure.accessibilitySummary)
    }

    private func ingestSeriesTitle(_ series: PipelineSeries) -> String {
        switch series {
        case .allCommits: "All chunks · chunk rows"
        case .agentStores: "Agent · chunk rows"
        case .jsonlWatcher: "Watcher · unique chunk IDs"
        case .enrichment: "Enriched · chunk rows"
        }
    }

    private func summaryTile<Content: View>(
        title: String,
        identifier: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
            content()
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .padding(16)
        .background(BrainBarDashboardCardStyle(emphasized: true))
        .accessibilityIdentifier("brainbar.dashboard.tile.\(identifier)")
    }

    private var onePageBackupLines: [ObservabilityStatusLine] {
        onePagePresentation.backupLines
    }

    private func signalCoveragePanel(layout: BrainBarDashboardLayout) -> some View {
        BrainBarSignalCoveragePanel(
            stats: collector.stats,
            compact: layout.compactCards,
            isExpanded: $panelState.signalCoverageExpanded,
            isVectorDetailExpanded: $vectorSignalDetailExpanded,
            onAnimationCompleted: panelState.disclosureAnimationDidComplete
        )
    }

    private func vectorDetailXOffset(layout: BrainBarDashboardLayout) -> CGFloat {
        max(layout.outerPadding, vectorSignalRootFrame.minX)
    }

    private func vectorDetailYOffset(layout: BrainBarDashboardLayout, containerHeight: CGFloat) -> CGFloat {
        BrainBarVectorDetailLayout.yOffset(
            anchorMaxY: vectorSignalRootFrame.maxY,
            gap: layout.compactCards ? 8 : 10,
            detailHeight: vectorDetailHeight,
            containerHeight: containerHeight,
            padding: layout.outerPadding
        )
    }

    private func vectorDetailWidth(layout: BrainBarDashboardLayout) -> CGFloat {
        max(layout.compactCards ? 150 : 170, vectorSignalRootFrame.width)
    }

    private var vectorDetailTransition: AnyTransition {
        // NOTE: deliberately NOT using RevealClip here. RevealClip masks the popover
        // to height*progress; the height-measurement re-render (BrainBarVectorDetailHeightKey)
        // interrupts the insertion animation, leaving progress < 1 → the popover renders
        // CLIPPED to a partial height with empty space below it (Etan live-QA 2026-06-20).
        // A plain opacity (+ subtle slide on removal) can never clip the content height.
        if reduceMotion {
            .opacity
        } else {
            .asymmetric(
                insertion: .opacity,
                removal: .opacity.combined(with: .offset(y: -6))
            )
        }
    }

    @ViewBuilder
    private func diagnostics(layout: BrainBarDashboardLayout) -> some View {
        let activityRows = [
            ("Indexed in window", flowSummary.allCommits.volumeText),
            ("Agent writes (24 h)", onePagePresentation.agentWritesCount.map { DashboardMetricFormatter.integerString($0, locale: locale) } ?? onePagePresentation.agentWritesText),
            ("DB size", ByteCountFormatter.string(
                fromByteCount: collector.stats.databaseSizeBytes,
                countStyle: .file
            )),
        ]
        let runtimeRows = [
            ("Daemon", daemonSummary),
            ("Agents", collector.agentActivity.summaryText),
            ("State", collector.state.label),
            ("Hotkey", hotkeyStatus.replacingOccurrences(of: "Hotkey ", with: "")),
            ("Last seen", daemonLastSeenSummary),
        ]

        VStack(alignment: .leading, spacing: 12) {
            BrainBarDisclosureRow(
                isExpanded: $panelState.detailsExpanded,
                accessibilityIdentifier: "brainbar.dashboard.runtime-disclosure",
                accessibilityLabel: "Details",
                onAnimationCompleted: panelState.disclosureAnimationDidComplete
            ) {
                VStack(alignment: .leading, spacing: 14) {
                    signalCoveragePanel(layout: layout)
                    if layout.diagnosticColumns == 2 {
                        HStack(alignment: .top, spacing: 24) {
                            BrainBarDefinitionList(title: "Activity", rows: activityRows)
                            BrainBarDefinitionList(title: "Runtime", rows: runtimeRows)
                        }
                    } else {
                        VStack(spacing: 16) {
                            BrainBarDefinitionList(title: "Activity", rows: activityRows)
                            BrainBarDefinitionList(title: "Runtime", rows: runtimeRows)
                        }
                    }
                }
                .padding(.top, 8)
            } label: {
                Text("Details")
                    .font(.system(size: 14, weight: .semibold))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            BrainBarGlassPanel(cornerRadius: layout.panelCornerRadius, tint: .brainBarAccentViolet)
        )
    }

    private var daemonSummary: String {
        guard let daemon = collector.daemon else { return "Unavailable" }
        return "PID \(daemon.pid) · \(daemon.openConnections) sockets"
    }

    private var daemonLastSeenSummary: String {
        guard let daemon = collector.daemon else { return "Unavailable" }
        return DashboardMetricFormatter.relativeEventString(lastEventAt: daemon.lastSeenAt, now: currentNow)
    }
}

private struct BrainBarDefinitionList: View {
    let title: String
    let rows: [(String, String)]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .padding(.bottom, 5)
            ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(row.0)
                        .font(.system(size: 11))
                        .foregroundStyle(Color.brainBarTextSecondary)
                    Spacer(minLength: 8)
                    Text(row.1)
                        .font(.system(size: 13, weight: .semibold))
                        .monospacedDigit()
                        .foregroundStyle(row.1.localizedCaseInsensitiveContains("unavailable") ? Color.orange : Color.brainBarTextPrimary)
                        .lineLimit(1)
                }
                .frame(height: 24)
                if index < rows.count - 1 {
                    Rectangle().fill(Color.brainBarBorderSoft).frame(height: 0.5)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct BrainBarKeyboardFocusButton<Label: View>: View {
    let action: () -> Void
    @ViewBuilder let label: () -> Label
    @FocusState private var isFocused: Bool
    @State private var interaction = BrainBarDisclosureInteractionState()

    var body: some View {
        Button {
            _ = interaction.activate(isExpanded: false, source: .current())
            action()
        } label: {
            label().contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .focusEffectDisabled()
        .focused($isFocused)
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color.accentColor, lineWidth: 2)
                .opacity(interaction.showsKeyboardFocusRing && isFocused ? 1 : 0)
        }
        .onChange(of: isFocused) { _, focused in
            interaction.registerFocusChange(isFocused: focused, source: .current())
        }
    }
}

enum BrainBarDisclosureActivationSource: Equatable {
    case pointer
    case keyboard

    @MainActor
    static func current(event: NSEvent? = NSApp.currentEvent) -> Self {
        switch event?.type {
        case .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp,
             .otherMouseDown, .otherMouseUp:
            return .pointer
        default:
            // Keyboard focus changes and accessibility actions do not always
            // retain a key event, so the non-pointer fallback must stay visible.
            return .keyboard
        }
    }
}

struct BrainBarDisclosureInteractionState {
    private(set) var showsKeyboardFocusRing = false

    mutating func activate(
        isExpanded: Bool,
        source: BrainBarDisclosureActivationSource
    ) -> Bool {
        showsKeyboardFocusRing = source == .keyboard
        return !isExpanded
    }

    mutating func registerFocusChange(
        isFocused: Bool,
        source: BrainBarDisclosureActivationSource
    ) {
        showsKeyboardFocusRing = isFocused && source == .keyboard
    }
}

enum BrainBarDashboardScrollPosition {
    static func topOrigin(
        documentBounds: CGRect,
        viewportHeight: CGFloat,
        documentIsFlipped: Bool,
        currentX: CGFloat
    ) -> CGPoint {
        let y = documentIsFlipped
            ? documentBounds.minY
            : max(documentBounds.maxY - viewportHeight, documentBounds.minY)
        return CGPoint(x: currentX, y: y)
    }
}

private struct BrainBarDashboardDisclosureState: Equatable {
    let detailsExpanded: Bool
    let signalCoverageExpanded: Bool
}

private struct BrainBarDashboardScrollResetter: NSViewRepresentable {
    let disclosureState: BrainBarDashboardDisclosureState

    final class Coordinator {
        var previousState: BrainBarDashboardDisclosureState?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> NSView { NSView(frame: .zero) }

    func updateNSView(_ nsView: NSView, context: Context) {
        let previousState = context.coordinator.previousState
        context.coordinator.previousState = disclosureState
        guard previousState != nil, previousState != disclosureState else { return }

        DispatchQueue.main.async {
            guard let scrollView = nsView.enclosingScrollView,
                  let documentView = scrollView.documentView else { return }
            let clipView = scrollView.contentView
            let origin = BrainBarDashboardScrollPosition.topOrigin(
                documentBounds: documentView.bounds,
                viewportHeight: clipView.bounds.height,
                documentIsFlipped: documentView.isFlipped,
                currentX: clipView.bounds.origin.x
            )
            clipView.scroll(to: origin)
            scrollView.reflectScrolledClipView(clipView)
        }
    }
}

private struct BrainBarDisclosureRow<Label: View, Content: View>: View {
    @Binding var isExpanded: Bool
    let accessibilityIdentifier: String
    let accessibilityLabel: String
    let chevronPlacement: BrainBarDisclosureChevronPlacement
    let focusStateOverride: Bool?
    let onAnimationCompleted: () -> Void
    @ViewBuilder let content: () -> Content
    @ViewBuilder let label: () -> Label

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var isFocused: Bool
    @State private var interaction: BrainBarDisclosureInteractionState
    @State private var expansionProgress: CGFloat
    @State private var isAnimatingExpansion = false

    init(
        isExpanded: Binding<Bool>,
        accessibilityIdentifier: String,
        accessibilityLabel: String,
        chevronPlacement: BrainBarDisclosureChevronPlacement = .leading,
        focusStateOverride: Bool? = nil,
        initialInteraction: BrainBarDisclosureInteractionState = .init(),
        onAnimationCompleted: @escaping () -> Void = {},
        @ViewBuilder content: @escaping () -> Content,
        @ViewBuilder label: @escaping () -> Label
    ) {
        _isExpanded = isExpanded
        self.accessibilityIdentifier = accessibilityIdentifier
        self.accessibilityLabel = accessibilityLabel
        self.chevronPlacement = chevronPlacement
        self.focusStateOverride = focusStateOverride
        self.onAnimationCompleted = onAnimationCompleted
        _interaction = State(initialValue: initialInteraction)
        _expansionProgress = State(initialValue: isExpanded.wrappedValue ? 1 : 0)
        self.content = content
        self.label = label
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                let nextExpansion = interaction.activate(
                    isExpanded: isExpanded,
                    source: .current()
                )
                beginExpansionTransition(to: nextExpansion)
            } label: {
                HStack(spacing: 8) {
                    if chevronPlacement == .leading { disclosureChevron }
                    label()
                    if chevronPlacement == .trailing { disclosureChevron }
                }
                .padding(.vertical, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .focusEffectDisabled()
            .focused($isFocused)
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color.accentColor, lineWidth: 2)
                    .opacity(focusRingIsVisible ? 1 : 0)
            }
            .onChange(of: isFocused) { _, focused in
                interaction.registerFocusChange(isFocused: focused, source: .current())
            }
            .accessibilityLabel(accessibilityLabel)
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
            .accessibilityHint(isExpanded ? "Collapse" : "Expand")
            .accessibilityIdentifier(accessibilityIdentifier)

            if isExpanded, !isAnimatingExpansion {
                content()
            } else if isAnimatingExpansion {
                BrainBarDisclosureContentLayout(progress: expansionProgress) {
                    content()
                }
                .clipped()
                .allowsHitTesting(isExpanded)
                .accessibilityHidden(!isExpanded)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onChange(of: isExpanded) { _, expanded in
            guard !isAnimatingExpansion else { return }
            expansionProgress = expanded ? 1 : 0
        }
    }

    private var focusRingIsVisible: Bool {
        interaction.showsKeyboardFocusRing && (focusStateOverride ?? isFocused)
    }

    private var disclosureChevron: some View {
        Image(systemName: "chevron.right")
            .font(.system(size: 9, weight: .bold))
            .frame(width: 12, height: 12)
            .rotationEffect(.degrees(isExpanded ? 90 : 0))
    }

    private func beginExpansionTransition(to expanded: Bool) {
        guard !isAnimatingExpansion else { return }

        let direction: BrainBarDisclosureAnimation.Direction = expanded ? .open : .close
        let animation = BrainBarDisclosureAnimation.animation(for: direction, reduceMotion: reduceMotion)
        let targetProgress: CGFloat = expanded ? 1 : 0

        guard animation != nil else {
            isExpanded = expanded
            expansionProgress = targetProgress
            onAnimationCompleted()
            return
        }

        isAnimatingExpansion = true
        expansionProgress = expanded ? 0 : 1
        DispatchQueue.main.async {
            withAnimation(animation, completionCriteria: .logicallyComplete) {
                isExpanded = expanded
                expansionProgress = targetProgress
            } completion: {
                isAnimatingExpansion = false
                onAnimationCompleted()
            }
        }
    }
}

enum BrainBarDisclosureChevronPlacement {
    case leading
    case trailing
}

private struct BrainBarDisclosureContentLayout: Layout {
    var progress: CGFloat
    nonisolated var animatableData: CGFloat {
        get { progress }
        set { progress = newValue }
    }

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        guard let subview = subviews.first else { return .zero }
        let expandedSize = subview.sizeThatFits(ProposedViewSize(width: proposal.width, height: nil))
        return CGSize(
            width: proposal.width ?? expandedSize.width,
            height: BrainBarDisclosureAnimation.containerHeight(
                progress: progress,
                collapsedContainerHeight: 0,
                expandedContainerHeight: expandedSize.height
            )
        )
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard let subview = subviews.first else { return }
        let expandedSize = subview.sizeThatFits(ProposedViewSize(width: bounds.width, height: nil))
        subview.place(
            at: bounds.origin,
            anchor: .topLeading,
            proposal: ProposedViewSize(width: bounds.width, height: expandedSize.height)
        )
    }
}

enum BrainBarDisclosureRowPreview {
    @MainActor
    static func make(focusSource: BrainBarDisclosureActivationSource) -> some View {
        var interaction = BrainBarDisclosureInteractionState()
        interaction.registerFocusChange(isFocused: true, source: focusSource)
        return BrainBarDisclosureRow(
            isExpanded: .constant(false),
            accessibilityIdentifier: "brainbar.preview.disclosure",
            accessibilityLabel: "Details",
            focusStateOverride: true,
            initialInteraction: interaction
        ) {
            EmptyView()
        } label: {
            Text("Details")
                .font(.system(size: 14, weight: .semibold))
        }
        .padding(12)
        .frame(width: 320, height: 64)
        .background(Color(red: 0.08, green: 0.10, blue: 0.15))
        .environment(\.colorScheme, .dark)
    }
}

private struct BrainBarDashboardHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

private struct BrainBarSnapshotFreshnessBanner: View {
    let state: SnapshotFreshnessState
    let lastGoodFetchedAt: Date?
    let heartbeatText: String?
    let isHeartbeatAheadOfStats: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    stateLabel
                    Text(dataAgeText)
                    Text(lastGoodText)
                    Spacer(minLength: 8)
                    heartbeatLabel
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 10) {
                        stateLabel
                        Text(dataAgeText)
                        Spacer(minLength: 0)
                    }
                    Text(lastGoodText)
                    heartbeatLabel
                }
            }

            if let errorMessage {
                Text("Current error: \(errorMessage)")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(stateColor)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .font(.system(size: 11, weight: .semibold))
        .monospacedDigit()
        .foregroundStyle(Color.brainBarTextSecondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .fill(stateColor.opacity(isAttentionState ? 0.14 : 0.07))
        )
        .overlay(
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .stroke(stateColor.opacity(isAttentionState ? 0.48 : 0.22), lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilitySummary)
        .accessibilityIdentifier("brainbar.dashboard.freshness")
        .help(accessibilitySummary)
    }

    private var stateLabel: some View {
        Label(state.label, systemImage: stateSymbol)
            .font(.system(size: 11, weight: .bold))
            .foregroundStyle(stateColor)
    }

    @ViewBuilder
    private var heartbeatLabel: some View {
        if let heartbeatText {
            Text("Collector heartbeat: \(heartbeatText)\(isHeartbeatAheadOfStats ? " · refreshing" : "")")
                .foregroundStyle(Color.brainBarTextSecondary.opacity(0.78))
                .lineLimit(1)
        }
    }

    private var dataAgeText: String {
        guard let ageSeconds = state.ageSeconds else { return "Data age: not yet available" }
        return "Data age: \(ageLabel(ageSeconds))"
    }

    private var lastGoodText: String {
        guard let lastGoodFetchedAt else { return "Last good: none yet" }
        return "Last good: \(DashboardMetricFormatter.absoluteTimeString(lastGoodFetchedAt))"
    }

    private var errorMessage: String? {
        guard case .error(let message, _) = state else { return nil }
        return message
    }

    private var isAttentionState: Bool {
        switch state {
        case .stale, .error:
            return true
        case .loading, .live:
            return false
        }
    }

    private var stateColor: Color {
        switch state {
        case .loading:
            return BrainBarStateTheme.loading.theme.swiftUIColor
        case .live:
            return BrainBarStateTheme.active.theme.swiftUIColor
        case .stale:
            return BrainBarStateTheme.degraded.theme.swiftUIColor
        case .error:
            return BrainBarStateTheme.error.theme.swiftUIColor
        }
    }

    private var stateSymbol: String {
        switch state {
        case .loading: return "arrow.clockwise"
        case .live: return "checkmark.circle.fill"
        case .stale: return "clock.badge.exclamationmark"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var accessibilitySummary: String {
        var parts = [state.label, dataAgeText, lastGoodText]
        if let errorMessage {
            parts.append("Current error: \(errorMessage)")
        }
        if let heartbeatText {
            parts.append("Collector heartbeat: \(heartbeatText)")
        }
        return parts.joined(separator: ". ")
    }

    private func ageLabel(_ seconds: Int) -> String {
        if seconds < 60 { return "\(seconds)s" }
        let minutes = seconds / 60
        let remainder = seconds % 60
        return remainder == 0 ? "\(minutes)m" : "\(minutes)m \(remainder)s"
    }
}

private struct RevealClip: ViewModifier, Animatable {
    var progress: CGFloat          // 0 = collapsed, 1 = full height
    nonisolated var animatableData: CGFloat { get { progress } set { progress = newValue } }
    func body(content: Content) -> some View {
        content
            .frame(maxWidth: .infinity, alignment: .top)
            .mask(alignment: .top) {
                GeometryReader { geo in
                    Color.black.frame(height: geo.size.height * max(0, min(progress, 1)), alignment: .top)
                }
            }
    }
}

private enum BrainBarVectorSignalCoordinateSpace {
    static let root = "BrainBarRoot"
}

private struct BrainBarVectorSignalRootFrameKey: PreferenceKey {
    static let defaultValue: CGRect = .zero

    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        let next = nextValue()
        if next != .zero {
            value = next
        }
    }
}

private struct BrainBarVectorDetailHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        let next = nextValue()
        if next > 0 {
            value = next
        }
    }
}

/// Positioning math for the Vector "see under the hood" popover. Kept as a pure,
/// testable function so the clamp logic can't silently regress to clipping.
enum BrainBarVectorDetailLayout {
    /// Y offset for the popover within the dashboard container. Prefers sitting just
    /// below the anchor (`anchorMaxY + gap`), but clamps upward so the full popover
    /// height stays within `containerHeight - padding` — it pops OVER the content
    /// above rather than overflowing and clipping at the bottom edge. Falls back to
    /// the preferred offset until the popover height has been measured.
    static func yOffset(
        anchorMaxY: CGFloat,
        gap: CGFloat,
        detailHeight: CGFloat,
        containerHeight: CGFloat,
        padding: CGFloat
    ) -> CGFloat {
        let preferred = anchorMaxY + gap
        guard detailHeight > 0, containerHeight > 0 else { return preferred }
        let highestTop = max(padding, containerHeight - padding - detailHeight)
        return min(preferred, highestTop)
    }
}

private struct BrainBarSignalCoveragePanel: View {
    let stats: BrainDatabase.DashboardStats
    let compact: Bool
    @Binding var isExpanded: Bool
    @Binding var isVectorDetailExpanded: Bool
    var onAnimationCompleted: () -> Void = {}
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealedSignalIDs: Set<String> = []

    private var signals: [BrainBarSignalCoverage] {
        [
            BrainBarSignalCoverage(
                name: "Vector",
                indexedCount: stats.vectorIndexedChunkCount,
                totalCount: stats.signalEligibleChunkCount,
                backlogCount: stats.vectorBacklogCount,
                coveragePercent: stats.vectorCoveragePercent,
                isAvailable: stats.signalCoverageIsAvailable,
                accentColor: .brainBarSignalVector,
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
                isAvailable: stats.signalCoverageIsAvailable,
                accentColor: .brainBarSignalFTS5,
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
                isAvailable: stats.signalCoverageIsAvailable,
                accentColor: .brainBarSignalTrigram,
                showsDetail: false,
                vectorNetDrainRatePerHour: nil,
                vectorBacklogETAHours: nil
            ),
        ]
    }

    private func setVectorDetail(_ open: Bool) {
        withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.25)) {
            isVectorDetailExpanded = open
        }
    }

    var body: some View {
        BrainBarDisclosureRow(
            isExpanded: $isExpanded,
            accessibilityIdentifier: "brainbar.dashboard.signal-coverage-disclosure",
            accessibilityLabel: "Signal coverage",
            chevronPlacement: .trailing,
            onAnimationCompleted: onAnimationCompleted
        ) {
            signalBars
                .frame(maxWidth: .infinity, alignment: .top)
                .padding(.top, compact ? 10 : 12)
        } label: {
            HStack(spacing: 10) {
                Text("Signal coverage")
                    .font(.system(size: 13, weight: .semibold))
                if !isExpanded {
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: compact ? 10 : 14) {
                            ForEach(signals) { signal in signalChip(for: signal) }
                        }
                        Text(signals.map { "\($0.name) \($0.percentText)" }.joined(separator: " · "))
                            .font(.system(size: 11))
                            .foregroundStyle(Color.brainBarTextSecondary)
                    }
                }
                Spacer(minLength: 8)
                Text(isExpanded ? "Hide" : "Under the hood")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onExitCommand {
            if isVectorDetailExpanded { setVectorDetail(false) }
        }
        .onAppear { updateRevealedSignals(animated: false) }
        .onChange(of: isExpanded) { _, expanded in
            if !expanded { setVectorDetail(false) }
            updateRevealedSignals(animated: true)
        }
    }

    private func signalChip(for signal: BrainBarSignalCoverage) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(signal.accentColor)
                .frame(width: 7, height: 7)
            Text(signal.name)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextSecondary)
            Text(signal.percentText)
                .font(.system(size: 12, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(Color.brainBarTextPrimary)
        }
        .fixedSize(horizontal: true, vertical: false)
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
                BrainBarKeyboardFocusButton {
                    setVectorDetail(!isVectorDetailExpanded)
                } label: {
                    BrainBarSignalCoverageRow(
                        signal: signal,
                        compact: compact,
                        isSelected: isVectorDetailExpanded
                    )
                    .frame(maxWidth: .infinity)
                }
                .help("Show Vector backlog details")
            } else {
                BrainBarSignalCoverageRow(signal: signal, compact: compact, isSelected: false)
            }
        }
        .frame(minWidth: compact ? 150 : 170, maxWidth: .infinity, alignment: .topLeading)
        .opacity(isExpanded ? (revealedSignalIDs.contains(signal.id) ? 1 : 0) : 1)
        .background {
            if signal.showsDetail {
                GeometryReader { proxy in
                    Color.clear
                        .preference(
                            key: BrainBarVectorSignalRootFrameKey.self,
                            value: proxy.frame(in: .named(BrainBarVectorSignalCoordinateSpace.root))
                        )
                }
            }
        }
    }

    private func updateRevealedSignals(animated: Bool) {
        guard isExpanded else {
            revealedSignalIDs.removeAll()
            return
        }
        if reduceMotion || !animated {
            revealedSignalIDs = Set(signals.map(\.id))
            return
        }
        revealedSignalIDs.removeAll()
        for (index, signal) in signals.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.06) {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                    _ = revealedSignalIDs.insert(signal.id)
                }
            }
        }
    }
}

private struct BrainBarSignalCoverage: Identifiable {
    let name: String
    let indexedCount: Int
    let totalCount: Int
    let backlogCount: Int
    let coveragePercent: Double
    let isAvailable: Bool
    let accentColor: Color
    let showsDetail: Bool
    let vectorNetDrainRatePerHour: Double?
    let vectorBacklogETAHours: Double?

    var id: String { name }

    var percentText: String {
        guard isAvailable else { return "computing…" }
        return String(format: "%.0f%%", coveragePercent)
    }

    var clampedCoveragePercent: Double {
        min(max(coveragePercent, 0), 100)
    }

    var backlogText: String {
        guard isAvailable else { return "computing…" }
        return NumberFormatter.localizedString(from: NSNumber(value: backlogCount), number: .decimal)
    }
}

private struct BrainBarSignalCoverageRow: View {
    let signal: BrainBarSignalCoverage
    let compact: Bool
    let isSelected: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 7 : 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(signal.name)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Color.brainBarTextPrimary)
                Spacer(minLength: 8)
                Text(signal.percentText)
                    .font(.system(size: compact ? 18 : 20, weight: .bold, design: .rounded))
                    .foregroundStyle(signal.accentColor)
                    .monospacedDigit()
            }

            if signal.isAvailable {
                BrainBarAnimatedCoverageBar(percent: signal.clampedCoveragePercent, accentColor: signal.accentColor)
                    .frame(height: 6)
            } else {
                Capsule()
                    .fill(signal.accentColor.opacity(0.16))
                    .frame(height: 6)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(BrainBarDashboardCardStyle(emphasized: isSelected, cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(signal.accentColor.opacity(isSelected ? 0.70 : 0.2), lineWidth: isSelected ? 1.5 : 1)
        }
        .shadow(color: isSelected ? signal.accentColor.opacity(0.18) : .clear, radius: 12, y: 2)
        .scaleEffect(isSelected ? 0.98 : 1)
        .animation(reduceMotion ? nil : .spring(response: 0.4, dampingFraction: 0.85), value: signal.clampedCoveragePercent)
    }
}

private struct BrainBarAnimatedCoverageBar: View {
    let percent: Double
    let accentColor: Color
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var displayedPercent: Double = 0

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(accentColor.opacity(0.16))
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [accentColor, accentColor.opacity(0.70)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: proxy.size.width * min(max(displayedPercent, 0), 100) / 100)
            }
        }
        .onAppear {
            if reduceMotion {
                displayedPercent = percent
            } else {
                displayedPercent = 0
                withAnimation(.easeOut(duration: 0.45)) {
                    displayedPercent = percent
                }
            }
        }
        .onChange(of: percent) { _, newValue in
            if reduceMotion {
                displayedPercent = newValue
            } else {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.85)) {
                    displayedPercent = newValue
                }
            }
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
            ZStack {
                RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                    .fill(Color.brainBarBackgroundRaised)
                RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                    .fill(signal.accentColor.opacity(0.10))
                RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                    .stroke(
                        LinearGradient(colors: [Color.brainBarWhite.opacity(0.08), .clear],
                                       startPoint: .top, endPoint: .bottom),
                        lineWidth: 1
                    )
            }
            .overlay(
                RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                    .stroke(signal.accentColor.opacity(0.45), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous))
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
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.60)
                .foregroundStyle(Color.brainBarTextSecondary.opacity(0.50))
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
                VStack(alignment: .leading, spacing: 3) {
                    Text(lane.name.uppercased())
                        .font(.system(size: 10, weight: .semibold))
                        .tracking(0.80)
                        .foregroundStyle(Color.brainBarTextSecondary.opacity(0.60))
                    Text(lane.rateText)
                        .font(.system(size: compact ? 22 : 26, weight: .bold))
                        .monospacedDigit()
                        .foregroundStyle(Color.brainBarTextPrimary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.78)
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
                tertiaryValues: lane.tertiaryValues,
                tertiarySeriesLabel: lane.tertiarySeriesLabel,
                latestBucketName: lane.latestBucketName,
                accentColor: lane.accentColor,
                secondaryAccentColor: lane.secondaryAccentColor,
                tertiaryAccentColor: lane.tertiaryAccentColor,
                activityWindowMinutes: lane.activityWindowMinutes,
                fetchedAt: fetchedAt,
                pulseRevision: pulseRevision,
                referenceValue: sparklineReferenceValue,
                metricDisclosure: nil,
                accessibilitySummary: nil
            )
            .frame(height: chartHeight)

            if let primarySeriesLabel = lane.primarySeriesLabel,
               let secondarySeriesLabel = lane.secondarySeriesLabel,
               let secondaryAccentColor = lane.secondaryAccentColor {
                BrainBarSeriesLegend(
                    primaryLabel: primarySeriesLabel,
                    primaryColor: lane.accentColor,
                    primaryIsActive: lane.values.reduce(0, +) > 0,
                    secondaryLabel: secondarySeriesLabel,
                    secondaryColor: secondaryAccentColor,
                    secondaryIsActive: lane.secondaryValues.reduce(0, +) > 0,
                    tertiaryLabel: lane.tertiarySeriesLabel,
                    tertiaryColor: lane.tertiaryAccentColor,
                    tertiaryIsActive: lane.tertiaryValues.reduce(0, +) > 0
                )
            }

            Text("\(lane.volumeText) · \(lane.lastEventText)")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextSecondary.opacity(0.70))
                .lineLimit(1)
                .truncationMode(.tail)

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

    private var sparklineReferenceValue: Int? {
        guard lane.name.localizedCaseInsensitiveContains("enrichment") else { return nil }
        let peak = max(lane.values.max() ?? 0, lane.secondaryValues.max() ?? 0, lane.tertiaryValues.max() ?? 0)
        // No benchmark line on a completely empty chart — it would imply a phantom target.
        return peak > 0 ? peak : nil
    }
}

/// The ONE shared window for every pipeline graph: Live(1h) / 3h / 24h.
/// `.live` is the resting 1h window the graphs always have; `.threeHour` and
/// `.day` trigger a REAL windowed re-fetch from `BrainDatabase` (180 / 1440
/// minutes) so the charts show genuine historical data, not a relabel. Selecting
/// a window switches ALL graphs at once — there is no per-card timeframe control.
enum PipelineTimeframe: String, CaseIterable, Identifiable {
    case live
    case threeHour
    case day

    var id: String { rawValue }

    /// The chip label shown in the shared selector.
    var label: String {
        switch self {
        case .live: return "Live"
        case .threeHour: return "3h"
        case .day: return "24h"
        }
    }

    /// Sublabel for the live chip so "Live" still surfaces the 1h window.
    var subLabel: String? {
        switch self {
        case .live: return "1h"
        case .threeHour, .day: return nil
        }
    }

    var windowMinutes: Int {
        switch self {
        // Live is the resting 1h window (matches StatsCollector.defaultActivityWindowMinutes
        // and the hero's "in last 1h"); the chip previously mislabeled it 30m.
        case .live: return 60
        case .threeHour: return 180
        case .day: return 1_440
        }
    }

    static func truthfulDisplay(
        selected: PipelineTimeframe,
        loadedWindowMinutes: Int?
    ) -> PipelineTimeframe {
        guard selected != .live,
              loadedWindowMinutes == selected.windowMinutes else {
            return .live
        }
        return selected
    }
}

enum BrainBarPipelinePulseGate {
    static func shouldPulse(
        previous: [Int],
        current: [Int],
        timeframe: PipelineTimeframe
    ) -> Bool {
        guard timeframe == .live else { return false }
        return BrainBarLivePulse.shouldPulse(previous: previous, current: current)
    }
}

/// The single shared Live/3h/24h selector that drives every pipeline graph.
/// Replaces the removed per-card 1h/3h/24h picker. A small spinner appears
/// while a wider window is being fetched from the DB.
struct BrainBarSharedTimeframeSelector: View {
    @Binding var selection: PipelineTimeframe
    var isLoading: Bool = false
    var loadError: String?

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack(spacing: 6) {
                if isLoading {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.7)
                        .frame(width: 14, height: 14)
                }
                ForEach(PipelineTimeframe.allCases) { frame in
                    BrainBarTimeframeButton(frame: frame, isSelected: selection == frame) {
                        selection = frame
                    }
                }
            }
            if let loadError {
                Text(loadError)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(BrainBarStateTheme.degraded.theme.swiftUIColor)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("brainbar.dashboard.timeframe.error")
            }
        }
        .accessibilityIdentifier("brainbar.dashboard.timeframe")
        .accessibilityLabel("Dashboard chart window")
    }
}

private struct BrainBarTimeframeButton: View {
    let frame: PipelineTimeframe
    let isSelected: Bool
    let action: () -> Void
    @FocusState private var isFocused: Bool
    @State private var interaction = BrainBarDisclosureInteractionState()

    var body: some View {
        Button {
            _ = interaction.activate(isExpanded: isSelected, source: .current())
            action()
        } label: {
            HStack(spacing: 4) {
                Text(frame.label)
                    .font(.system(size: 11, weight: .semibold))
                if let sub = frame.subLabel {
                    Text(sub)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            .monospacedDigit()
            .padding(.horizontal, 11)
            .padding(.vertical, 5)
            .foregroundStyle(isSelected ? Color.brainBarAccent : Color.brainBarTextSecondary.opacity(0.85))
            .background(Capsule().fill(isSelected ? Color.brainBarAccent.opacity(0.16) : .clear))
            .overlay(
                Capsule().stroke(
                    isSelected ? Color.brainBarAccent.opacity(0.32) : Color.brainBarBorderSoft,
                    lineWidth: 1
                )
            )
        }
        .buttonStyle(.plain)
        .focusEffectDisabled()
        .focused($isFocused)
        .overlay {
            Capsule().stroke(Color.accentColor, lineWidth: 2)
                .opacity(interaction.showsKeyboardFocusRing && isFocused ? 1 : 0)
        }
        .onChange(of: isFocused) { _, focused in
            interaction.registerFocusChange(isFocused: focused, source: .current())
        }
        .help("Show \(frame.label) window")
        .accessibilityIdentifier("brainbar.dashboard.timeframe.\(frame.rawValue)")
    }
}

/// A single-series truth card for chunk rows, agent-origin chunks,
/// watcher-ingested chunks, or successful enrichments.
///
/// Unlike `BrainBarFlowLaneCard` (kept for legacy `ingress`/diagnostics callers),
/// this card plots exactly ONE series so `SparklineChartPresentation.maxValue`
/// auto-fits to that series alone — fixing the scale disconnect for free. It
/// preserves the Aldante aesthetic by reusing `BrainBarHeroSparkline` (gradient
/// density fill + point-anchored hover) and the number-first hero header.
///
/// REDESIGN: there is no longer any expand/collapse/peek state. Every graph is
/// ALWAYS visible at its resting height; clicking a graph does nothing to its
/// layout. The window is driven externally by the ONE shared timeframe selector
/// (`timeframe`), which feeds REAL windowed data into `lane` so the chart shows
/// genuine history for the selected window.
private struct BrainBarPipelineSeriesCard: View {
    let series: PipelineSeries
    let lane: DashboardFlowLane
    let pulseRevision: Int
    let compact: Bool
    let chartHeight: CGFloat
    let fetchedAt: Date
    let timeframe: PipelineTimeframe

    private var presentation: SparklineChartPresentation {
        let disclosure = chartDisclosure
        return SparklineChartPresentation(
            label: lane.sparklineLabel,
            values: lane.values,
            activityWindowMinutes: lane.activityWindowMinutes,
            latestBucketName: lane.latestBucketName,
            fetchedAt: fetchedAt,
            metricDisclosure: disclosure.tooltipDisclosure,
            accessibilitySummary: disclosure.accessibilitySummary
        )
    }

    /// "scale · peak N" so two similar-height waveforms are not misread as equal
    /// volume — the magnitude gap stays honest even at auto-fit y-scales. When a
    /// dashed reference line is drawn (enrichment lanes), the caption doubles as
    /// its legend — "dashed = peak N" — so the line is never an unlabeled mystery.
    private var scaleCaptionText: String {
        if sparklineReferenceValue != nil {
            return "scale · dashed = peak \(presentation.axisMax)"
        }
        return "scale · peak \(presentation.axisMax)"
    }

    /// The window currently selected by the shared control, e.g. "Last 1h" /
    /// "Last 3h" / "Last 24h" — kept in the caption so the chart always names the
    /// window it is plotting.
    private var timeframeWindowText: String {
        DashboardMetricFormatter.windowLabel(minutes: timeframe.windowMinutes)
    }

    private var latestBucketCaptionText: String? {
        let latest = lane.values.last ?? 0
        guard latest > 0 else { return nil }
        return "latest: \(DashboardMetricFormatter.axisTickString(latest))"
    }

    private var accent: Color { Color.brainBar(nsColor: lane.accentColor) }

    private var chartDisclosure: BrainBarDashboardChartDisclosure {
        BrainBarDashboardChartDisclosure(series: series, lane: lane, timeframe: timeframe)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 10) {
            header

            ZStack(alignment: .bottomTrailing) {
                BrainBarHeroSparkline(
                    label: lane.sparklineLabel,
                    values: lane.values,
                    secondaryValues: [],
                    primarySeriesLabel: nil,
                    secondarySeriesLabel: nil,
                    tertiaryValues: [],
                    tertiarySeriesLabel: nil,
                    latestBucketName: lane.latestBucketName,
                    accentColor: lane.accentColor,
                    secondaryAccentColor: nil,
                    tertiaryAccentColor: nil,
                    activityWindowMinutes: lane.activityWindowMinutes,
                fetchedAt: fetchedAt,
                pulseRevision: pulseRevision,
                referenceValue: sparklineReferenceValue,
                metricDisclosure: chartDisclosure.tooltipDisclosure,
                accessibilitySummary: chartDisclosure.accessibilitySummary
                )
                .frame(height: chartHeight)

                Text(scaleCaptionText)
                    .font(.system(size: 10, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.62))
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(
                        Capsule().fill(Color.brainBarGlassSecondary.opacity(0.85))
                    )
                    .padding(6)
                    .allowsHitTesting(false)
            }

            // ONE caption line: volume for the selected window, plus the window
            // label so the shared selector visibly relabels every card.
            HStack(spacing: 8) {
                Text(lane.volumeText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.70))
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text("·")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.45))
                if let latestBucketCaptionText {
                    Text(latestBucketCaptionText)
                        .font(.system(size: 11, weight: .semibold))
                        .monospacedDigit()
                        .foregroundStyle(accent.opacity(0.88))
                        .lineLimit(1)
                    Text("·")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(Color.brainBarTextSecondary.opacity(0.45))
                }
                Text("window: \(timeframeWindowText)")
                    .font(.system(size: 11, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(accent.opacity(0.85))
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(compact ? 12 : 14)
        .background(BrainBarDashboardCardStyle(emphasized: true))
        .accessibilityIdentifier(chartDisclosure.accessibilityIdentifier)
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text(lane.name.uppercased())
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(0.80)
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.60))
                Text(lane.rateText)
                    .font(.system(size: compact ? 22 : 26, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(Color.brainBarTextPrimary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.78)
                Text(chartDisclosure.subtitle)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.76))
                    .fixedSize(horizontal: false, vertical: true)
                if let detail = chartDisclosure.visibleDetail {
                    Text(detail)
                        .font(.system(size: 9.5, weight: .medium))
                        .foregroundStyle(Color.brainBarTextSecondary.opacity(0.64))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 12)
            BrainBarFlowStatusPill(
                text: chartDisclosure.statusLabel,
                accentColor: lane.status.stateTheme.theme.swiftUIColor
            )
        }
    }

    private var sparklineReferenceValue: Int? {
        guard series == .enrichment else { return nil }
        let peak = lane.values.max() ?? 0
        return peak > 0 ? peak : nil
    }
}

struct BrainBarIngestSeriesPresentation: Equatable {
    let metricText: String
    let showsSparkline: Bool

    init(lane: DashboardFlowLane, locale: Locale = .current) {
        guard lane.status != .unavailable else {
            metricText = "Unavailable"
            showsSparkline = false
            return
        }

        metricText = "\(DashboardMetricFormatter.integerString(lane.values.reduce(0, +), locale: locale)) · peak \(DashboardMetricFormatter.axisTickString(lane.values.max() ?? 0, locale: locale))"
        showsSparkline = true
    }
}

struct BrainBarDashboardChartDisclosure: Equatable {
    let subtitle: String
    let visibleDetail: String?
    let accessibilityIdentifier: String
    let statusLabel: String
    let accessibilitySummary: String
    let tooltipDisclosure: String

    init(series: PipelineSeries, lane: DashboardFlowLane, timeframe: PipelineTimeframe) {
        let totalCount = lane.values.reduce(0, +)
        let windowLabel = DashboardMetricFormatter.windowLabel(minutes: timeframe.windowMinutes)
        let clockLabel: String
        let unitLabel: String

        switch series {
        case .allCommits:
            subtitle = "Source time · chunk rows"
            visibleDetail = nil
            accessibilityIdentifier = "brainbar.dashboard.chart.chunk-rows"
            statusLabel = lane.status.label
            clockLabel = "source time"
            unitLabel = "chunk rows"
        case .agentStores:
            subtitle = "Source time · chunk rows · documented agent origins"
            visibleDetail = "Sources: MCP, manual, digest, precompact-hook, brain_store, pending, fallback, fallback-replay."
            accessibilityIdentifier = "brainbar.dashboard.chart.agent-origin-chunks"
            statusLabel = lane.status.label
            clockLabel = "source time"
            unitLabel = "agent-origin chunk rows"
        case .jsonlWatcher:
            subtitle = "Ingest time · unique chunk IDs first seen in window · not additive with source-time charts"
            visibleDetail = "Live state uses a 60s process-and-flow truth window; this chart uses the selected context window."
            accessibilityIdentifier = "brainbar.dashboard.chart.watcher-ingested-chunks"
            statusLabel = lane.statusText
            clockLabel = "ingest time"
            unitLabel = "unique chunk IDs first seen in this window"
        case .enrichment:
            subtitle = "Completion time · successful chunk rows"
            visibleDetail = "Success status only; failed, skipped, and pending rows are disclosed above."
            accessibilityIdentifier = "brainbar.dashboard.chart.enriched-successfully"
            statusLabel = lane.status.label
            clockLabel = "successful enrichment completion time"
            unitLabel = "success-status chunk rows"
        }

        if lane.status == .unavailable {
            accessibilitySummary = "\(lane.name). Evidence unavailable. Window: \(windowLabel). Unit: \(unitLabel). Clock: \(clockLabel)."
            tooltipDisclosure = "Evidence unavailable"
        } else {
            accessibilitySummary = "\(lane.name). Window: \(windowLabel). Count: \(DashboardMetricFormatter.integerString(totalCount)). Unit: \(unitLabel). Clock: \(clockLabel)."
            tooltipDisclosure = "Window: \(windowLabel) · Count: hovered value below · Unit: \(unitLabel) · Clock: \(clockLabel)"
        }
    }
}

#if DEBUG
/// Debug-only seam: render the (private) full dashboard view for deterministic
/// visual QA. Never compiled into a release build. Pair with a fixture collector
/// (`BrainBarDashboardFixture.makeCollector()`) and `accessibilityReduceMotion`
/// so the render is byte-stable. `width` (set on the host frame by the caller)
/// selects the layout breakpoint: compact < 920 ≤ default < 1040 ≤ wide.
@MainActor
enum BrainBarDashboardPreview {
    static var goldenCalendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        calendar.locale = goldenLocale
        return calendar
    }

    static let goldenLocale = Locale(identifier: "en_US")

    static func make(
        collector: StatsCollector,
        hotkeyStatus: String = "Hotkey ⌃⌥Space ready",
        observabilityResult: ObservabilityReadResult? = nil,
        now: Date? = nil,
        calendar: Calendar = goldenCalendar,
        locale: Locale = goldenLocale,
        panelState: BrainBarDashboardPanelState? = nil
    ) -> AnyView {
        AnyView(
            ZStack {
                BrainBarAppBackground()
                BrainBarDashboardContent(
                    collector: collector,
                    hotkeyStatus: hotkeyStatus,
                    observabilityResult: observabilityResult,
                    referenceNow: now,
                    calendar: calendar,
                    locale: locale,
                    panelState: panelState
                )
            }
            .environment(\.colorScheme, .dark)
            // Suppress SwiftUI animations so every value lands at its final state
            // immediately — no mid-animation capture. (The read-only
            // `accessibilityReduceMotion` env key can't be injected directly.)
            .transaction { $0.disablesAnimations = true }
        )
    }
}

/// Debug-only seam: render the (private) number-first flow lane card for visual QA.
/// Never compiled into a release build.
@MainActor
enum BrainBarFlowLaneCardPreview {
    static func make(
        lane: DashboardFlowLane,
        compact: Bool = false,
        chartHeight: CGFloat = 170,
        fetchedAt: Date = Date()
    ) -> AnyView {
        AnyView(
            BrainBarFlowLaneCard(
                lane: lane,
                pulseRevision: 0,
                compact: compact,
                chartHeight: chartHeight,
                fetchedAt: fetchedAt,
                emphasize: true
            )
        )
    }
}

/// Debug-only seam: render the (private) PIPELINE + SIGNAL-COVERAGE composition
/// (writes card -> signal coverage panel -> enrichments card + queue rail ->
/// agent presence) exactly as `BrainBarDashboardView.pipelinePanel` lays it out,
/// but driven by an injected mock `DashboardStats` instead of a live
/// `StatsCollector`. This is the slice currently being redesigned; visual QA
/// snapshots render this seam so the layout is faithful to production without
/// needing a real DB. Never compiled into a release build.
@MainActor
enum BrainBarPipelinePanelPreview {
    static func make(
        stats: BrainDatabase.DashboardStats,
        containerSize: CGSize = CGSize(width: 980, height: 780),
        fetchedAt: Date = Date(),
        watcherText: String = "12 files",
        signalCoverageExpanded: Bool = true,
        replayDebtExpanded: Bool = false,
        selectedTimeframe: PipelineTimeframe = .live
    ) -> AnyView {
        AnyView(
            BrainBarPipelinePanelPreviewView(
                stats: stats,
                containerSize: containerSize,
                fetchedAt: fetchedAt,
                watcherText: watcherText,
                signalCoverageExpanded: signalCoverageExpanded,
                replayDebtExpanded: replayDebtExpanded,
                selectedTimeframe: selectedTimeframe
            )
            .environment(\.colorScheme, .dark)
            .transaction { $0.disablesAnimations = true }
        )
    }
}

/// Wrapper that owns the @State bindings the real signal-coverage panel needs
/// (`isExpanded` / `isVectorDetailExpanded`) so the production views can be
/// rendered as-is. Mirrors `BrainBarDashboardView.pipelinePanel(layout:)` — now
/// with the ONE shared timeframe selector in the Ingest header and ALL graphs
/// always visible at resting height (no expand/collapse).
private struct BrainBarPipelinePanelPreviewView: View {
    let stats: BrainDatabase.DashboardStats
    let containerSize: CGSize
    let fetchedAt: Date
    let watcherText: String
    let replayDebtExpanded: Bool
    @State private var signalCoverageExpanded: Bool
    @State private var vectorSignalDetailExpanded = false
    @State private var selectedTimeframe: PipelineTimeframe

    init(
        stats: BrainDatabase.DashboardStats,
        containerSize: CGSize,
        fetchedAt: Date,
        watcherText: String,
        signalCoverageExpanded: Bool,
        replayDebtExpanded: Bool,
        selectedTimeframe: PipelineTimeframe
    ) {
        self.stats = stats
        self.containerSize = containerSize
        self.fetchedAt = fetchedAt
        self.watcherText = watcherText
        self.replayDebtExpanded = replayDebtExpanded
        _signalCoverageExpanded = State(initialValue: signalCoverageExpanded)
        _selectedTimeframe = State(initialValue: selectedTimeframe)
    }

    private var flowSummary: DashboardFlowSummary {
        DashboardFlowSummary.derive(daemon: nil, stats: stats, now: fetchedAt)
    }

    var body: some View {
        let layout = BrainBarDashboardLayout(containerSize: containerSize)
        let restingHeight = layout.sparklineHeight

        VStack(alignment: .leading, spacing: layout.sectionSpacing) {
            // Band 1 + Band 2 — the INGEST panel, with the ONE shared selector.
            VStack(alignment: .leading, spacing: layout.gridSpacing) {
                HStack(alignment: .center, spacing: 12) {
                    BrainBarSectionLabel(
                        "Ingest",
                        caption: "Three independently scaled ingest series. Source-time charts count chunk rows; the watcher chart counts unique chunk IDs by ingest time."
                    )
                    Spacer(minLength: 8)
                    BrainBarSharedTimeframeSelector(selection: $selectedTimeframe)
                }

                ViewThatFits(in: .horizontal) {
                    VStack(alignment: .leading, spacing: 12) {
                        seriesCard(.allCommits, layout: layout, restingHeight: restingHeight)
                        seriesCard(.agentStores, layout: layout, restingHeight: restingHeight)
                        seriesCard(.jsonlWatcher, layout: layout, restingHeight: restingHeight)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        seriesCard(.allCommits, layout: layout, restingHeight: restingHeight)
                        seriesCard(.agentStores, layout: layout, restingHeight: restingHeight)
                        seriesCard(.jsonlWatcher, layout: layout, restingHeight: restingHeight)
                    }
                }

                BrainBarSignalCoveragePanel(
                    stats: stats,
                    compact: layout.compactCards,
                    isExpanded: $signalCoverageExpanded,
                    isVectorDetailExpanded: $vectorSignalDetailExpanded
                )
            }
            .padding(layout.cardPadding)
            .background(
                BrainBarGlassPanel(cornerRadius: layout.panelCornerRadius, tint: .brainBarAccent)
            )
        }
    }

    private func seriesCard(
        _ series: PipelineSeries,
        layout: BrainBarDashboardLayout,
        restingHeight: CGFloat
    ) -> some View {
        BrainBarPipelineSeriesCard(
            series: series,
            lane: flowSummary.lane(for: series),
            pulseRevision: 0,
            compact: layout.compactCards,
            chartHeight: restingHeight,
            fetchedAt: fetchedAt,
            timeframe: selectedTimeframe
        )
    }
}
#endif

private struct BrainBarQueueRail: View {
    let summary: DashboardQueueSummary
    let replayDebtBreakdown: BrainDatabase.ReplayDebtBreakdown
    let censusText: String
    let coverageText: String
    let watcherText: String
    let compact: Bool
    @State private var replayDebtExpanded: Bool

    init(
        summary: DashboardQueueSummary,
        replayDebtBreakdown: BrainDatabase.ReplayDebtBreakdown,
        censusText: String,
        coverageText: String,
        watcherText: String,
        compact: Bool,
        replayDebtExpanded: Bool = false
    ) {
        self.summary = summary
        self.replayDebtBreakdown = replayDebtBreakdown
        self.censusText = censusText
        self.coverageText = coverageText
        self.watcherText = watcherText
        self.compact = compact
        _replayDebtExpanded = State(initialValue: replayDebtExpanded)
    }

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

            if replayDebtBreakdown.deduplicatedTotal > 0 || replayDebtBreakdown.isPartial {
                replayDebtDisclosure
            } else {
                Text(summary.detail)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(compact ? 10 : 12)
        .background(BrainBarDashboardCardStyle())
    }

    private var replayDebtDisclosure: some View {
        BrainBarDisclosureRow(
            isExpanded: $replayDebtExpanded,
            accessibilityIdentifier: "brainbar.dashboard.replay-debt-disclosure",
            accessibilityLabel: "Replay debt"
        ) {
            VStack(alignment: .leading, spacing: 7) {
                replayDebtRow("Pending stores", component: replayDebtBreakdown.pendingStores)
                replayDebtRow("Queue entries", component: replayDebtBreakdown.durableQueue)
                replayDebtRow("Fallback entries", component: replayDebtBreakdown.repositoryFallback)
                replayDebtValueRow("Deduplicated total", value: DashboardMetricFormatter.integerString(replayDebtBreakdown.deduplicatedTotal))
                replayDebtValueRow("Unreadable inputs", value: unreadableInputsText)
                replayDebtValueRow("Census time", value: censusText)
                Text("Path identities are de-duplicated before the aggregate is calculated.")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.72))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 7)
        } label: {
            HStack(spacing: 8) {
                Text("Replay debt")
                    .font(.system(size: 11, weight: .bold))
                Text(DashboardMetricFormatter.integerString(replayDebtBreakdown.deduplicatedTotal))
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .monospacedDigit()
                if replayDebtBreakdown.isPartial {
                    Text("PARTIAL")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(BrainBarStateTheme.degraded.theme.swiftUIColor)
                }
                Spacer(minLength: 0)
                Text("Pending stores · queue entries · fallback entries")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }

    private func replayDebtRow(
        _ label: String,
        component: BrainDatabase.ReplayDebtBreakdown.Component
    ) -> some View {
        let readability: String
        switch component.readability {
        case .readable:
            readability = "readable"
        case .unreadable(let reason):
            readability = "unreadable · \(reason)"
        }
        return replayDebtValueRow(
            label,
            value: "\(DashboardMetricFormatter.integerString(component.snapshot.depth)) · \(readability)"
        )
    }

    private func replayDebtValueRow(_ label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Color.brainBarTextSecondary)
            Spacer(minLength: 8)
            Text(value)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(Color.brainBarTextPrimary)
                .multilineTextAlignment(.trailing)
        }
    }

    private var unreadableInputsText: String {
        guard replayDebtBreakdown.isPartial else { return "None" }
        return replayDebtBreakdown.unreadableSources.map { source in
            switch source {
            case .pendingStores: return "pending stores"
            case .durableQueue: return "queue entries"
            case .repositoryFallback: return "fallback entries"
            }
        }.joined(separator: ", ")
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
                BrainBarLaneMetric(label: "Coverage", value: coverageText)
                BrainBarLaneMetric(label: "Last watcher heartbeat", value: watcherText)
            }

            VStack(alignment: .leading, spacing: 8) {
                BrainBarLaneMetric(label: "Coverage", value: coverageText)
                BrainBarLaneMetric(label: "Last watcher heartbeat", value: watcherText)
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
    let primaryIsActive: Bool
    let secondaryLabel: String
    let secondaryColor: NSColor
    let secondaryIsActive: Bool
    let tertiaryLabel: String?
    let tertiaryColor: NSColor?
    let tertiaryIsActive: Bool

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 16) {
                legendItems
                Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 6) {
                legendItems
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
    }

    @ViewBuilder
    private var legendItems: some View {
        legendItem(label: primaryLabel, color: primaryColor, isActive: primaryIsActive)
        legendItem(label: secondaryLabel, color: secondaryColor, isActive: secondaryIsActive)
        if let tertiaryLabel, let tertiaryColor {
            legendItem(label: tertiaryLabel, color: tertiaryColor, isActive: tertiaryIsActive)
        }
    }

    private var accessibilityLabel: String {
        var labels = [primaryLabel, secondaryLabel]
        if let tertiaryLabel {
            labels.append(tertiaryLabel)
        }
        return "\(labels.joined(separator: ", ")) write series"
    }

    private func legendItem(label: String, color: NSColor, isActive: Bool) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(Color.brainBar(nsColor: color).opacity(isActive ? 1 : 0.35))
                .frame(width: 10, height: 10)
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextSecondary.opacity(isActive ? 0.70 : 0.35))
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
    let maxContentWidth: CGFloat

    init(containerSize: CGSize) {
        let compactWidth = containerSize.width < 920

        chartColumns = containerSize.width >= 1_040 ? 2 : 1
        overviewMetricColumns = containerSize.width >= 900 ? 4 : 2
        diagnosticColumns = containerSize.width >= 880 ? 2 : 1
        diagnosticItemColumns = containerSize.width >= 760 ? 2 : 1

        compactCards = compactWidth
        outerPadding = compactCards ? 16 : 24
        sectionSpacing = compactCards ? 14 : 20
        gridSpacing = compactCards ? 12 : 16
        cardPadding = compactCards ? 16 : 22
        overviewTitleFontSize = compactCards ? BrainBarDesignTokens.TypeScale.title : BrainBarDesignTokens.TypeScale.display
        overviewSubtitleFontSize = BrainBarDesignTokens.TypeScale.body
        overviewStatsWidth = compactCards ? 330 : 430
        metricCardMinHeight = compactCards ? 72 : 88
        metricValueFontSize = compactCards ? 32 : 40
        sparklineHeight = compactCards ? 112 : 140
        panelCornerRadius = BrainBarDesignTokens.Radius.xl
        maxContentWidth = 1_280
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
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .monospacedDigit()
            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.60)
                .foregroundStyle(Color.brainBarTextSecondary.opacity(0.50))
                .textCase(.uppercase)
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
                    // Cap tile width so diagnostics don't sprawl across a wide
                    // window — pairs of label/value stay compact and scannable
                    // (Part B: "diagnostics too wide").
                    repeating: GridItem(.flexible(minimum: 150, maximum: 260), spacing: 10, alignment: .leading),
                    count: columns
                ),
                alignment: .leading,
                spacing: 10
            ) {
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    BrainBarDiagnosticTile(label: row.0, value: row.1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(BrainBarDashboardCardStyle())
    }
}

private struct BrainBarHeroSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: BrainBarDesignTokens.TypeScale.label, weight: .bold))
                .tracking(0.7)
                .foregroundStyle(Color.brainBarTextSecondary)
                .textCase(.uppercase)
            content
        }
        .frame(maxWidth: .infinity, minHeight: 104, alignment: .topLeading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.md, style: .continuous)
                .fill(Color.brainBarGlassSecondary)
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("brainbar.dashboard.hero.\(title.lowercased())")
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
    var cornerRadius: CGFloat = 18

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
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
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.brainBarBorderSoft, lineWidth: 1)
            )
            .overlay(alignment: .top) {
                Capsule(style: .continuous)
                    .fill(Color.brainBarWhite.opacity(0.06))
                    .frame(height: 1)
                    .padding(.horizontal, max(10, cornerRadius * 0.70))
            }
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .shadow(color: Color.brainBarBlack.opacity(0.30), radius: 8, y: 2)
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
    let tertiaryValues: [Int]
    let tertiarySeriesLabel: String?
    let latestBucketName: String
    let accentColor: NSColor
    let secondaryAccentColor: NSColor?
    let tertiaryAccentColor: NSColor?
    let activityWindowMinutes: Int
    let fetchedAt: Date
    let pulseRevision: Int
    let referenceValue: Int?
    let metricDisclosure: String?
    let accessibilitySummary: String?

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
                    tertiaryValues: tertiaryValues,
                    primarySeriesLabel: primarySeriesLabel,
                    secondarySeriesLabel: secondarySeriesLabel,
                    tertiarySeriesLabel: tertiarySeriesLabel,
                    activityWindowMinutes: activityWindowMinutes,
                    latestBucketName: latestBucketName,
                    fetchedAt: fetchedAt,
                    metricDisclosure: metricDisclosure,
                    accessibilitySummary: accessibilitySummary
                ),
                accentColor: accentColor,
                secondaryAccentColor: secondaryAccentColor,
                tertiaryAccentColor: tertiaryAccentColor,
                compact: SparklineRenderer.isCompact(size: renderSize),
                referenceValue: referenceValue
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
    let caption: String?

    init(_ title: String, caption: String? = nil) {
        self.title = title
        self.caption = caption
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
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
            if let caption {
                Text(caption)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary.opacity(0.70))
                    .fixedSize(horizontal: false, vertical: true)
            }
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
