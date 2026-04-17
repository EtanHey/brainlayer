import AppKit
import Combine
import SwiftUI

struct BrainBarWindowRootView: View {
    @ObservedObject var runtime: BrainBarRuntime

    @State private var selectedTab: BrainBarTab = .dashboard
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

            Group {
                switch selectedTab {
                case .dashboard:
                    dashboardContent
                case .injections:
                    injectionsContent
                case .graph:
                    graphContent
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .top) {
                BrainBarCommandBarResultsOverlay(viewModel: commandBarViewModel)
                    .padding(.horizontal, 20)
                    .padding(.top, 10)
                    .animation(.easeInOut(duration: 0.18), value: commandBarViewModel?.results.count)
                    .animation(.easeInOut(duration: 0.18), value: commandBarViewModel?.mode)
                    .animation(.easeInOut(duration: 0.18), value: commandBarViewModel?.inputText.isEmpty)
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
            if let action = runtime.requestedQuickAction {
                handleRequestedQuickAction(action)
            }
        }
        .onReceive(runtime.$database) { _ in
            ensureCommandBarViewModel()
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
        selectedTab = .dashboard
        commandBarViewModel?.setMode(action == .capture ? .capture : .search)
        commandBarViewModel?.panelDidAppear()
        runtime.clearQuickActionRequest()
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

    @State private var previousBuckets: [Int] = []
    @State private var pulseRevision = 0

    private var livePresentation: BrainBarLivePresentation {
        BrainBarLivePresentation.derive(stats: collector.stats)
    }

    private var heroSparkline: NSImage {
        SparklineRenderer.render(
            state: collector.state,
            values: collector.stats.recentEnrichmentBuckets,
            size: NSSize(width: 360, height: 160),
            accentColor: livePresentation.accentColor
        )
    }

    var body: some View {
        GeometryReader { proxy in
            let layout = BrainBarDashboardLayout(containerSize: proxy.size)

            VStack(spacing: 0) {
                hero(layout: layout)

                VStack(spacing: layout.sectionSpacing) {
                    HStack(alignment: .top, spacing: layout.gridSpacing) {
                        BrainBarMetricCard(
                            title: "Chunks",
                            value: "\(collector.stats.chunkCount)",
                            valueFontSize: layout.metricValueFontSize,
                            minHeight: layout.metricCardMinHeight,
                            cardPadding: layout.metricCardPadding
                        )
                        BrainBarMetricCard(
                            title: "Enriched",
                            value: "\(collector.stats.enrichedChunkCount)",
                            valueFontSize: layout.metricValueFontSize,
                            minHeight: layout.metricCardMinHeight,
                            cardPadding: layout.metricCardPadding
                        )
                        BrainBarMetricCard(
                            title: "Backlog",
                            value: "\(collector.stats.pendingEnrichmentCount)",
                            valueFontSize: layout.metricValueFontSize,
                            minHeight: layout.metricCardMinHeight,
                            cardPadding: layout.metricCardPadding
                        )
                        BrainBarMetricCard(
                            title: "Last enriched",
                            value: DashboardMetricFormatter.lastCompletionString(
                                recentEnrichmentBuckets: collector.stats.recentEnrichmentBuckets
                            ),
                            valueFontSize: layout.metricValueFontSize,
                            minHeight: layout.metricCardMinHeight,
                            cardPadding: layout.metricCardPadding
                        )
                    }

                    HStack(alignment: .top, spacing: layout.infoSpacing) {
                        BrainBarInfoCard(
                            title: "Pipeline",
                            rows: [
                                ("State", collector.state.label),
                                (
                                    "Recent writes",
                                    DashboardMetricFormatter.activitySummaryString(
                                        recentActivityBuckets: collector.stats.recentActivityBuckets
                                    )
                                ),
                                ("Coverage", "\(Int(collector.stats.enrichmentPercent.rounded()))% enriched"),
                                ("DB", ByteCountFormatter.string(
                                    fromByteCount: collector.stats.databaseSizeBytes,
                                    countStyle: .file
                                )),
                            ]
                        )

                        BrainBarInfoCard(
                            title: "Runtime",
                            rows: [
                                ("Hotkey", hotkeyStatus),
                                ("Daemon", daemonSummary),
                                ("Window", runtimeSummary),
                            ]
                        )
                    }
                }
                .padding(.horizontal, layout.outerPadding)
                .padding(.top, layout.outerPadding)
                .padding(.bottom, layout.outerPadding)
            }
            .frame(width: proxy.size.width, height: layout.baseHeight, alignment: .top)
            .scaleEffect(layout.scale, anchor: .top)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .onAppear {
            previousBuckets = collector.stats.recentEnrichmentBuckets
        }
        .onChange(of: collector.stats.recentEnrichmentBuckets) { _, newBuckets in
            if BrainBarLivePulse.shouldPulse(previous: previousBuckets, current: newBuckets) {
                pulseRevision += 1
            }
            previousBuckets = newBuckets
        }
    }

    private func hero(layout: BrainBarDashboardLayout) -> some View {
        HStack(alignment: .bottom, spacing: 24) {
            VStack(alignment: .leading, spacing: 14) {
                Text("BrainLayer is \(collector.state.label.lowercased())")
                    .font(.system(size: layout.heroTitleFontSize, weight: .bold))

                Text(heroSubtitle)
                    .font(.system(size: layout.heroSubtitleFontSize, weight: .medium))
                    .foregroundStyle(.white.opacity(0.82))

                HStack(spacing: 10) {
                    BrainBarHeroBadge(text: livePresentation.badgeText)
                    BrainBarHeroBadge(text: "\(Int(collector.stats.enrichmentPercent.rounded()))% enriched")
                }

                Text(enrichmentSummary)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.72))
            }

            Spacer(minLength: 24)

            VStack(alignment: .trailing, spacing: 10) {
                BrainBarHeroSparkline(
                    image: heroSparkline,
                    values: collector.stats.recentEnrichmentBuckets,
                    accentColor: Color(nsColor: livePresentation.accentColor),
                    pulseRevision: pulseRevision
                )
                .frame(width: layout.sparklineWidth, height: layout.sparklineHeight)

                Text(livePresentation.statusText)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.8))
            }
        }
        .padding(.horizontal, layout.heroHorizontalPadding)
        .padding(.vertical, layout.heroVerticalPadding)
        .frame(maxWidth: .infinity, minHeight: layout.heroMinHeight, alignment: .leading)
        .background(
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: livePresentation.accentColor).opacity(0.88),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Color.white.opacity(0.12))
                .frame(height: 1)
        }
    }

    private var heroSubtitle: String {
        switch collector.state {
        case .indexing:
            return "Incoming writes are landing and enrichment is keeping pace."
        case .enriching:
            return "Backlog is draining through the current enrichment stream."
        case .idle:
            return "No fresh completions in the last minute. Window stays ready in the menu bar."
        case .degraded:
            return "The pipeline needs attention before live stats are trustworthy."
        }
    }

    private var enrichmentSummary: String {
        let recentCompletions = collector.stats.recentEnrichmentBuckets.reduce(0, +)
        if recentCompletions == 0 {
            return "No completions in the last 30 minutes."
        }
        return "\(recentCompletions) completions in the last 30 minutes."
    }

    private var daemonSummary: String {
        guard let daemon = collector.daemon else { return "Unavailable" }
        return "PID \(daemon.pid) · \(daemon.openConnections) sockets"
    }

    private var runtimeSummary: String {
        livePresentation.badgeText == "idle" ? "Hidden until reopened" : "Active live-state"
    }
}

struct BrainBarDashboardLayout {
    let baseHeight: CGFloat
    let scale: CGFloat
    let outerPadding: CGFloat
    let sectionSpacing: CGFloat
    let gridSpacing: CGFloat
    let infoSpacing: CGFloat
    let heroMinHeight: CGFloat
    let heroHorizontalPadding: CGFloat
    let heroVerticalPadding: CGFloat
    let heroTitleFontSize: CGFloat
    let heroSubtitleFontSize: CGFloat
    let metricCardMinHeight: CGFloat
    let metricCardPadding: CGFloat
    let metricValueFontSize: CGFloat
    let sparklineWidth: CGFloat
    let sparklineHeight: CGFloat

    init(containerSize: CGSize) {
        let compact = containerSize.height < 620 || containerSize.width < 900

        baseHeight = compact ? 520 : 584
        scale = min(1, max(0.84, containerSize.height / baseHeight))
        outerPadding = compact ? 14 : 20
        sectionSpacing = compact ? 14 : 20
        gridSpacing = compact ? 10 : 14
        infoSpacing = compact ? 10 : 14
        heroMinHeight = compact ? 188 : 220
        heroHorizontalPadding = compact ? 20 : 24
        heroVerticalPadding = compact ? 20 : 26
        heroTitleFontSize = compact ? 24 : 28
        heroSubtitleFontSize = compact ? 13 : 14
        metricCardMinHeight = compact ? 74 : 86
        metricCardPadding = compact ? 12 : 16
        metricValueFontSize = compact ? 20 : 24
        sparklineWidth = compact ? 280 : 320
        sparklineHeight = compact ? 128 : 150
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

private struct BrainBarInfoCard: View {
    let title: String
    let rows: [(String, String)]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))

            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(alignment: .top) {
                    Text(row.0)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 16)
                    Text(row.1)
                        .font(.system(size: 12, weight: .medium))
                        .multilineTextAlignment(.trailing)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
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

private struct BrainBarHeroSparkline: View {
    let image: NSImage
    let values: [Int]
    let accentColor: Color
    let pulseRevision: Int

    @State private var ringScale: CGFloat = 0.8
    @State private var ringOpacity = 0.0

    var body: some View {
        GeometryReader { proxy in
            let endpoint = SparklineRenderer.endpoint(
                values: values,
                size: NSSize(width: proxy.size.width, height: proxy.size.height)
            )

            ZStack(alignment: .topLeading) {
                Image(nsImage: image)
                    .interpolation(.high)
                    .resizable()
                    .scaledToFit()

                if let endpoint {
                    let point = CGPoint(
                        x: endpoint.x,
                        y: proxy.size.height - endpoint.y
                    )

                    Circle()
                        .stroke(accentColor.opacity(0.45), lineWidth: 2)
                        .frame(width: 26, height: 26)
                        .scaleEffect(ringScale)
                        .opacity(ringOpacity)
                        .position(point)

                    Circle()
                        .fill(accentColor)
                        .frame(width: 9, height: 9)
                        .shadow(color: accentColor.opacity(0.65), radius: 6)
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
