import AppKit
import Combine
import Foundation
import SwiftUI

// MARK: - Tab Enum

enum PopoverTab: Int, CaseIterable, Sendable {
    case dashboard = 0
    case injections = 1
    case graph = 2

    var label: String {
        switch self {
        case .dashboard: "Dashboard"
        case .injections: "Injections"
        case .graph: "Graph"
        }
    }

    var contentSize: NSSize {
        // Keep a single utility-panel size across tabs. AppKit popovers do not
        // handle repeated tab-triggered resizing cleanly; the earlier per-tab
        // sizes caused content to squash and overflow when switching around.
        NSSize(width: 560, height: 520)
    }
}

// MARK: - Popover View Controller

@MainActor
final class StatusPopoverView: NSViewController {
    private let collector: StatsCollector
    private let hotkeyStatus: HotkeyRouteStatus?
    private let injectionStore: InjectionStore?
    private let database: BrainDatabase?
    private var cancellables: Set<AnyCancellable> = []

    private let segmentedControl = NSSegmentedControl()
    private let containerView = NSView()
    private(set) var currentTab: PopoverTab = .dashboard
    var onPreferredSizeChange: (@MainActor (NSSize) -> Void)?

    // Dashboard labels
    private let titleLabel = NSTextField(labelWithString: "BrainBar")
    private let statusLabel = NSTextField(labelWithString: "")
    private let pipelineValueLabel = NSTextField(labelWithString: "")
    private let databaseSizeLabel = NSTextField(labelWithString: "")
    private let chunkMetric = MetricTileView(title: "Chunks")
    private let enrichedMetric = MetricTileView(title: "Enriched")
    private let pendingMetric = MetricTileView(title: "Backlog")
    private let rateMetric = MetricTileView(title: "Speed")
    private let indexingIndicator = PipelineIndicatorBadgeView(name: "Indexing")
    private let enrichingIndicator = PipelineIndicatorBadgeView(name: "Enriching")
    private let activityLabel = NSTextField(labelWithString: "Pipeline Throughput")
    private let enrichmentLabel = NSTextField(labelWithString: "")
    private let indexingActivityView = PipelineActivityRowView(
        title: "Indexing",
        symbolName: "server.rack",
        accentColor: .systemBlue
    )
    private let enrichingActivityView = PipelineActivityRowView(
        title: "Enriching",
        symbolName: "sparkles",
        accentColor: .systemPurple
    )
    private let daemonLabel = NSTextField(labelWithString: "")
    private let hotkeyLabel = NSTextField(labelWithString: "")
    private let headerPanel = SurfacePanelView(cornerRadius: 14)
    private let activityPanel = SurfacePanelView(cornerRadius: 14)

    // Lazily created tab content
    private var dashboardContent: NSView?
    private var injectionHosting: NSHostingController<PopoverInjectionTab>?
    private var graphHosting: NSHostingController<PopoverGraphTab>?

    init(
        collector: StatsCollector,
        hotkeyStatus: HotkeyRouteStatus? = nil,
        injectionStore: InjectionStore? = nil,
        database: BrainDatabase? = nil
    ) {
        self.collector = collector
        self.hotkeyStatus = hotkeyStatus
        self.injectionStore = injectionStore
        self.database = database
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        let size = PopoverTab.dashboard.contentSize
        view = NSView(frame: NSRect(origin: .zero, size: size))
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        configureDashboardLabels()
        configureSegmentedControl()
        configureLayout()
        showTab(.dashboard)
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        bind()
        render()
    }

    // MARK: - Tab Switching

    func showTab(_ tab: PopoverTab) {
        containerView.subviews.forEach { $0.removeFromSuperview() }
        currentTab = tab

        let content: NSView
        switch tab {
        case .dashboard:
            content = makeDashboardContent()
        case .injections:
            content = makeInjectionContent()
        case .graph:
            content = makeGraphContent()
        }

        content.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(content)
        NSLayoutConstraint.activate([
            content.topAnchor.constraint(equalTo: containerView.topAnchor),
            content.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            content.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            content.bottomAnchor.constraint(equalTo: containerView.bottomAnchor),
        ])

        if preferredContentSize != tab.contentSize {
            preferredContentSize = tab.contentSize
            onPreferredSizeChange?(tab.contentSize)
        }
    }

    // MARK: - Segmented Control

    private func configureSegmentedControl() {
        segmentedControl.segmentCount = PopoverTab.allCases.count
        for tab in PopoverTab.allCases {
            segmentedControl.setLabel(tab.label, forSegment: tab.rawValue)
            segmentedControl.setWidth(0, forSegment: tab.rawValue)
        }
        segmentedControl.selectedSegment = 0
        segmentedControl.segmentStyle = .automatic
        segmentedControl.target = self
        segmentedControl.action = #selector(tabChanged(_:))

        if injectionStore == nil {
            segmentedControl.setEnabled(false, forSegment: PopoverTab.injections.rawValue)
        }
        if database == nil {
            segmentedControl.setEnabled(false, forSegment: PopoverTab.graph.rawValue)
        }
    }

    @objc private func tabChanged(_ sender: NSSegmentedControl) {
        guard let tab = PopoverTab(rawValue: sender.selectedSegment) else { return }
        showTab(tab)
    }

    // MARK: - Layout

    private func configureLayout() {
        segmentedControl.translatesAutoresizingMaskIntoConstraints = false
        containerView.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(segmentedControl)
        view.addSubview(containerView)

        NSLayoutConstraint.activate([
            segmentedControl.topAnchor.constraint(equalTo: view.topAnchor, constant: 10),
            segmentedControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            containerView.topAnchor.constraint(equalTo: segmentedControl.bottomAnchor, constant: 8),
            containerView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            containerView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            containerView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    // MARK: - Dashboard Content

    private func configureDashboardLabels() {
        titleLabel.font = .systemFont(ofSize: 18, weight: .bold)
        statusLabel.font = .systemFont(ofSize: 12, weight: .medium)
        statusLabel.textColor = .secondaryLabelColor
        pipelineValueLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        databaseSizeLabel.font = .monospacedDigitSystemFont(ofSize: 11, weight: .medium)
        databaseSizeLabel.alignment = .right
        databaseSizeLabel.textColor = .tertiaryLabelColor
        activityLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        enrichmentLabel.font = .monospacedDigitSystemFont(ofSize: 11, weight: .medium)
        enrichmentLabel.alignment = .right
        enrichmentLabel.textColor = .tertiaryLabelColor
        daemonLabel.font = .monospacedDigitSystemFont(ofSize: 11, weight: .medium)
        daemonLabel.textColor = .secondaryLabelColor
        daemonLabel.maximumNumberOfLines = 2
        daemonLabel.lineBreakMode = .byTruncatingTail
        hotkeyLabel.font = .systemFont(ofSize: 11, weight: .medium)
        hotkeyLabel.textColor = .secondaryLabelColor
        hotkeyLabel.maximumNumberOfLines = 2
        hotkeyLabel.lineBreakMode = .byWordWrapping
    }

    private func makeDashboardContent() -> NSView {
        if let existing = dashboardContent { return existing }

        let titleRow = NSStackView(views: [titleLabel, NSView(), pipelineValueLabel])
        titleRow.orientation = .horizontal
        titleRow.alignment = .firstBaseline
        titleRow.spacing = 12

        let metaRow = NSStackView(views: [statusLabel, NSView(), enrichmentLabel, databaseSizeLabel])
        metaRow.orientation = .horizontal
        metaRow.alignment = .centerY
        metaRow.spacing = 12

        let indicatorsRow = NSStackView(views: [indexingIndicator, enrichingIndicator, NSView()])
        indicatorsRow.orientation = .horizontal
        indicatorsRow.alignment = .centerY
        indicatorsRow.spacing = 8

        let metricsRow = NSStackView(views: [chunkMetric, enrichedMetric, pendingMetric, rateMetric])
        metricsRow.orientation = .horizontal
        metricsRow.distribution = .fillEqually
        metricsRow.spacing = 10

        let quitButton = NSButton(title: "Quit", target: self, action: #selector(quitBrainBar))
        quitButton.bezelStyle = .rounded
        quitButton.controlSize = .small

        let footerRow = NSStackView(views: [daemonLabel, NSView(), hotkeyLabel, quitButton])
        footerRow.orientation = .horizontal
        footerRow.alignment = .centerY
        footerRow.spacing = 10
        hotkeyLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        daemonLabel.setContentCompressionResistancePriority(.defaultHigh, for: .horizontal)

        let headerStack = NSStackView(views: [titleRow, metaRow, indicatorsRow, metricsRow, footerRow])
        headerStack.orientation = .vertical
        headerStack.spacing = 10
        let headerCard = headerPanel.wrap(
            headerStack,
            edgeInsets: NSEdgeInsets(top: 14, left: 14, bottom: 14, right: 14)
        )

        let activityStack = NSStackView(views: [activityLabel, indexingActivityView, enrichingActivityView])
        activityStack.orientation = .vertical
        activityStack.spacing = 10
        let activityCard = activityPanel.wrap(
            activityStack,
            edgeInsets: NSEdgeInsets(top: 14, left: 14, bottom: 14, right: 14)
        )
        hotkeyLabel.isHidden = hotkeyStatus == nil

        for row in [headerCard, activityCard] as [NSView] {
            row.translatesAutoresizingMaskIntoConstraints = false
            row.setContentHuggingPriority(.defaultLow, for: .horizontal)
        }

        let content = NSStackView(views: [headerCard, activityCard])
        content.orientation = .vertical
        content.alignment = .width
        content.spacing = 10
        content.edgeInsets = NSEdgeInsets(top: 8, left: 14, bottom: 14, right: 14)
        dashboardContent = content
        return content
    }

    // MARK: - Injection Content

    private func makeInjectionContent() -> NSView {
        if let hosting = injectionHosting { return hosting.view }

        guard let store = injectionStore else {
            return makePlaceholder("Injection store unavailable")
        }

        store.start()
        let hosting = NSHostingController(rootView: PopoverInjectionTab(store: store))
        injectionHosting = hosting
        addChild(hosting)
        return hosting.view
    }

    // MARK: - Graph Content

    private func makeGraphContent() -> NSView {
        if let hosting = graphHosting { return hosting.view }

        guard let db = database else {
            return makePlaceholder("Database unavailable")
        }

        let viewModel = KGViewModel(database: db)
        let hosting = NSHostingController(rootView: PopoverGraphTab(viewModel: viewModel))
        graphHosting = hosting
        addChild(hosting)
        return hosting.view
    }

    // MARK: - Data Binding

    private func bind() {
        collector.$stats
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        collector.$state
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        collector.$daemon
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        hotkeyStatus?.$statusLine
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)
    }

    private func render() {
        titleLabel.stringValue = "BrainBar"
        statusLabel.stringValue = collector.state == .indexing ? "Incoming writes active" :
            (collector.state == .enriching ? "Backlog is being processed" :
            (collector.state == .idle ? "Pipeline settled" : "Pipeline degraded"))
        pipelineValueLabel.stringValue = collector.state.label
        pipelineValueLabel.textColor = collector.state.color
        headerPanel.backgroundColor = NSColor.underPageBackgroundColor.withAlphaComponent(0.92)
        headerPanel.borderColor = NSColor.separatorColor.withAlphaComponent(0.45)
        activityPanel.backgroundColor = NSColor.underPageBackgroundColor.withAlphaComponent(0.92)
        activityPanel.borderColor = NSColor.separatorColor.withAlphaComponent(0.45)
        databaseSizeLabel.stringValue = "\(byteString(collector.stats.databaseSizeBytes)) db"

        let indicators = PipelineIndicators.derive(daemon: collector.daemon, stats: collector.stats)
        let enrichmentDisplayRatePerMinute = PipelineActivityTracks.displayedRatePerMinute(
            primaryRatePerMinute: collector.stats.enrichmentRatePerMinute,
            values: collector.stats.recentEnrichmentBuckets
        )
        let indexingRatePerMinute = PipelineActivityTracks.recentRatePerMinute(
            values: collector.stats.recentActivityBuckets,
            activityWindowMinutes: 30,
            trailingBucketCount: 2
        )
        indexingIndicator.setStatus(indicators.indexing.status)
        enrichingIndicator.setStatus(indicators.enriching.status)

        chunkMetric.value = integerString(collector.stats.chunkCount)
        enrichedMetric.value = integerString(collector.stats.enrichedChunkCount)
        pendingMetric.value = integerString(collector.stats.pendingEnrichmentCount)
        rateMetric.value = DashboardMetricFormatter.speedString(ratePerMinute: enrichmentDisplayRatePerMinute)

        enrichmentLabel.stringValue = "\(Int(collector.stats.enrichmentPercent.rounded()))% enriched"
        let activityTracks = PipelineActivityTracks.derive(daemon: collector.daemon, stats: collector.stats)
        indexingActivityView.update(
            track: activityTracks.indexing,
            detailRateText: DashboardMetricFormatter.rateDetailString(
                ratePerMinute: indexingRatePerMinute
            )
        )
        enrichingActivityView.update(
            track: activityTracks.enriching,
            detailRateText: DashboardMetricFormatter.rateDetailString(
                ratePerMinute: enrichmentDisplayRatePerMinute
            )
        )

        if let daemon = collector.daemon {
            daemonLabel.stringValue = "PID \(daemon.pid)   RSS \(byteString(Int64(daemon.rssBytes)))   Sockets \(daemon.openConnections)"
        } else {
            daemonLabel.stringValue = "Daemon metrics unavailable"
        }

        if let hotkeyStatus {
            hotkeyLabel.isHidden = false
            hotkeyLabel.stringValue = hotkeyStatus.statusLine
        } else {
            hotkeyLabel.isHidden = true
        }
    }

    // MARK: - Actions

    @objc
    private func quitBrainBar() {
        NSApplication.shared.terminate(nil)
    }

    // MARK: - Helpers

    private func byteString(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }

    private func integerString(_ value: Int) -> String {
        NumberFormatter.localizedString(from: NSNumber(value: value), number: .decimal)
    }

    private func enrichmentActivitySummary(_ stats: DashboardStats) -> String {
        let recentCompletions = stats.recentEnrichmentBuckets.reduce(0, +)
        if recentCompletions == 0 {
            return "No completions in the last 30m"
        }
        return "\(recentCompletions) completions in the last 30m"
    }

    private func makePlaceholder(_ text: String) -> NSView {
        let label = NSTextField(labelWithString: text)
        label.font = .systemFont(ofSize: 13)
        label.textColor = .secondaryLabelColor
        label.alignment = .center
        let wrapper = NSView()
        label.translatesAutoresizingMaskIntoConstraints = false
        wrapper.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: wrapper.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: wrapper.centerYAnchor),
        ])
        return wrapper
    }
}

// MARK: - SwiftUI Tab Wrappers

struct PopoverInjectionTab: View {
    @ObservedObject var store: InjectionStore
    @State private var filterText = ""

    var body: some View {
        InjectionFeedView(store: store, filterText: $filterText)
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
    }
}

struct PopoverGraphTab: View {
    @ObservedObject var viewModel: KGViewModel

    var body: some View {
        KGCanvasView(viewModel: viewModel)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .clipped()
    }
}

// MARK: - Metric Tile

private final class MetricTileView: NSView {
    private let titleLabel = NSTextField(labelWithString: "")
    private let valueLabel = NSTextField(labelWithString: "")
    var value: String {
        get { valueLabel.stringValue }
        set { valueLabel.stringValue = newValue }
    }

    override var wantsUpdateLayer: Bool { true }

    override func updateLayer() {
        layer?.backgroundColor = NSColor.clear.cgColor
        layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor
        layer?.borderWidth = 0.8
    }

    init(title: String) {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = 6

        titleLabel.stringValue = title
        titleLabel.font = .systemFont(ofSize: 10, weight: .medium)
        titleLabel.textColor = .secondaryLabelColor
        valueLabel.font = .monospacedDigitSystemFont(ofSize: 20, weight: .semibold)
        valueLabel.alignment = .right
        valueLabel.textColor = .labelColor
        valueLabel.lineBreakMode = .byClipping
        valueLabel.maximumNumberOfLines = 1

        let stack = NSStackView(views: [titleLabel, valueLabel])
        stack.orientation = .vertical
        stack.alignment = .width
        stack.spacing = 4
        stack.translatesAutoresizingMaskIntoConstraints = false

        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10),
            heightAnchor.constraint(equalToConstant: 60),
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

private final class PipelineActivityRowView: NSView {
    private let iconView = NSImageView(frame: .zero)
    private let titleLabel = NSTextField(labelWithString: "")
    private let rateLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(labelWithString: "")
    private let sparklineImageView = NSImageView(frame: .zero)
    private let accentColor: NSColor

    init(title: String, symbolName: String, accentColor: NSColor) {
        self.accentColor = accentColor
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = 10
        layer?.backgroundColor = NSColor.separatorColor.withAlphaComponent(0.05).cgColor

        titleLabel.stringValue = title
        titleLabel.font = .systemFont(ofSize: 11, weight: .semibold)
        titleLabel.textColor = .secondaryLabelColor

        if let image = NSImage(systemSymbolName: symbolName, accessibilityDescription: title) {
            iconView.image = image
        }
        iconView.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 12, weight: .medium)
        iconView.contentTintColor = accentColor

        rateLabel.font = .monospacedDigitSystemFont(ofSize: 16, weight: .semibold)
        rateLabel.alignment = .right
        rateLabel.textColor = .labelColor

        detailLabel.font = .monospacedDigitSystemFont(ofSize: 11, weight: .medium)
        detailLabel.alignment = .right
        detailLabel.textColor = .tertiaryLabelColor
        detailLabel.lineBreakMode = .byTruncatingTail
        detailLabel.setContentCompressionResistancePriority(.required, for: .vertical)

        sparklineImageView.imageScaling = .scaleAxesIndependently
        sparklineImageView.wantsLayer = true
        sparklineImageView.layer?.cornerRadius = 4
        sparklineImageView.layer?.backgroundColor = NSColor.clear.cgColor

        let titleRow = NSStackView(views: [iconView, titleLabel, NSView(), rateLabel])
        titleRow.orientation = .horizontal
        titleRow.alignment = .centerY
        titleRow.spacing = 6

        let stack = NSStackView(views: [titleRow, sparklineImageView, detailLabel])
        stack.orientation = .vertical
        stack.spacing = 6
        stack.translatesAutoresizingMaskIntoConstraints = false

        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10),
            iconView.widthAnchor.constraint(equalToConstant: 14),
            iconView.heightAnchor.constraint(equalToConstant: 14),
            sparklineImageView.heightAnchor.constraint(equalToConstant: 56),
            heightAnchor.constraint(greaterThanOrEqualToConstant: 120),
        ])
    }

    func update(track: PipelineActivityTrack, detailRateText: String) {
        rateLabel.stringValue = track.rateText

        if track.rateText == "idle" || track.rateText == "queued" {
            detailLabel.stringValue = track.detailText
        } else if track.rateText.contains("/s") {
            detailLabel.stringValue = track.detailText
        } else {
            detailLabel.stringValue = "\(detailRateText) · \(track.detailText)"
        }

        let color = statusColor(for: track.status)
        iconView.contentTintColor = color
        sparklineImageView.image = SparklineRenderer.render(
            color: color,
            values: track.values,
            size: NSSize(width: 500, height: 56)
        )
    }

    private func statusColor(for status: PipelineIndicatorStatus) -> NSColor {
        switch status {
        case .live:
            return accentColor
        case .queued:
            return .systemOrange
        case .idle:
            return .secondaryLabelColor
        case .unavailable:
            return .systemRed
        }
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

private final class PipelineIndicatorBadgeView: NSView {
    private let dotView = NSView(frame: NSRect(x: 0, y: 0, width: 8, height: 8))
    private let label = NSTextField(labelWithString: "")
    private let name: String

    init(name: String) {
        self.name = name
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = 8

        dotView.wantsLayer = true
        dotView.layer?.cornerRadius = 4

        label.font = .systemFont(ofSize: 11, weight: .medium)
        label.textColor = .secondaryLabelColor

        let stack = NSStackView(views: [dotView, label])
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 6
        stack.translatesAutoresizingMaskIntoConstraints = false

        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
            dotView.widthAnchor.constraint(equalToConstant: 8),
            dotView.heightAnchor.constraint(equalToConstant: 8),
        ])

        layer?.backgroundColor = NSColor.separatorColor.withAlphaComponent(0.08).cgColor
        setStatus(.idle)
    }

    func setStatus(_ status: PipelineIndicatorStatus) {
        dotView.layer?.backgroundColor = status.color.cgColor
        label.stringValue = "\(name) \(status.label)"
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

private final class SurfacePanelView: NSView {
    var backgroundColor: NSColor = .controlBackgroundColor {
        didSet { needsDisplay = true }
    }

    var borderColor: NSColor = .separatorColor {
        didSet { needsDisplay = true }
    }

    override var wantsUpdateLayer: Bool { true }

    init(cornerRadius: CGFloat) {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = cornerRadius
    }

    override func updateLayer() {
        layer?.backgroundColor = backgroundColor.cgColor
        layer?.borderColor = borderColor.cgColor
        layer?.borderWidth = 1
    }

    func wrap(_ content: NSView, edgeInsets: NSEdgeInsets) -> NSView {
        content.translatesAutoresizingMaskIntoConstraints = false
        addSubview(content)
        NSLayoutConstraint.activate([
            content.topAnchor.constraint(equalTo: topAnchor, constant: edgeInsets.top),
            content.leadingAnchor.constraint(equalTo: leadingAnchor, constant: edgeInsets.left),
            content.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -edgeInsets.right),
            content.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -edgeInsets.bottom),
        ])
        return self
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
