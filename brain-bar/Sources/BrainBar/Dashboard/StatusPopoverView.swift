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
        switch self {
        case .dashboard: NSSize(width: 360, height: 350)
        case .injections: NSSize(width: 360, height: 440)
        case .graph: NSSize(width: 620, height: 500)
        }
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
    private let databaseSizeLabel = NSTextField(labelWithString: "")
    private let chunkMetric = MetricTileView(title: "Chunks")
    private let enrichedMetric = MetricTileView(title: "Enriched")
    private let pendingMetric = MetricTileView(title: "Pending")
    private let activityLabel = NSTextField(labelWithString: "Recent Activity")
    private let enrichmentLabel = NSTextField(labelWithString: "")
    private let sparklineImageView = NSImageView(frame: .zero)
    private let daemonLabel = NSTextField(labelWithString: "")
    private let hotkeyLabel = NSTextField(labelWithString: "")

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

        preferredContentSize = tab.contentSize
        onPreferredSizeChange?(tab.contentSize)
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
        titleLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        statusLabel.font = .systemFont(ofSize: 12, weight: .medium)
        databaseSizeLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        databaseSizeLabel.textColor = .secondaryLabelColor
        activityLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        enrichmentLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        enrichmentLabel.textColor = .secondaryLabelColor
        daemonLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        daemonLabel.textColor = .secondaryLabelColor
        hotkeyLabel.font = .systemFont(ofSize: 11, weight: .medium)
        hotkeyLabel.textColor = .secondaryLabelColor
        hotkeyLabel.maximumNumberOfLines = 2
        hotkeyLabel.lineBreakMode = .byWordWrapping

        sparklineImageView.imageScaling = .scaleAxesIndependently
        sparklineImageView.wantsLayer = true
        sparklineImageView.layer?.cornerRadius = 10
        sparklineImageView.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
    }

    private func makeDashboardContent() -> NSView {
        if let existing = dashboardContent { return existing }

        let headerStack = NSStackView(views: [
            verticalStack([titleLabel, statusLabel]),
            NSView(),
            databaseSizeLabel,
        ])
        headerStack.orientation = .horizontal
        headerStack.alignment = .centerY

        let metricsStack = NSStackView(views: [chunkMetric, enrichedMetric, pendingMetric])
        metricsStack.orientation = .horizontal
        metricsStack.distribution = .fillEqually
        metricsStack.spacing = 8

        let activityHeader = NSStackView(views: [activityLabel, NSView(), enrichmentLabel])
        activityHeader.orientation = .horizontal
        activityHeader.alignment = .centerY

        let refreshButton = NSButton(title: "Refresh", target: self, action: #selector(refreshDashboard))
        let quitButton = NSButton(title: "Quit BrainBar", target: self, action: #selector(quitBrainBar))
        let buttonRow = NSStackView(views: [refreshButton, quitButton, NSView()])
        buttonRow.orientation = .horizontal
        buttonRow.alignment = .centerY
        buttonRow.spacing = 8

        sparklineImageView.translatesAutoresizingMaskIntoConstraints = false
        hotkeyLabel.isHidden = hotkeyStatus == nil

        let content = NSStackView(views: [
            headerStack,
            metricsStack,
            activityHeader,
            sparklineImageView,
            daemonLabel,
            hotkeyLabel,
            buttonRow,
        ])
        content.orientation = .vertical
        content.alignment = .leading
        content.spacing = 12
        content.edgeInsets = NSEdgeInsets(top: 4, left: 14, bottom: 14, right: 14)

        NSLayoutConstraint.activate([
            sparklineImageView.widthAnchor.constraint(equalToConstant: 320),
            sparklineImageView.heightAnchor.constraint(equalToConstant: 42),
        ])

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
        statusLabel.stringValue = collector.state.label
        statusLabel.textColor = collector.state.color
        databaseSizeLabel.stringValue = byteString(collector.stats.databaseSizeBytes)

        chunkMetric.value = "\(collector.stats.chunkCount)"
        enrichedMetric.value = "\(collector.stats.enrichedChunkCount)"
        pendingMetric.value = "\(collector.stats.pendingEnrichmentCount)"

        enrichmentLabel.stringValue = "\(Int(collector.stats.enrichmentPercent.rounded()))% enriched"
        sparklineImageView.image = SparklineRenderer.render(
            state: collector.state,
            values: collector.stats.recentActivityBuckets,
            size: NSSize(width: 320, height: 42)
        )

        if let daemon = collector.daemon {
            daemonLabel.stringValue = "PID \(daemon.pid)   RSS \(byteString(Int64(daemon.rssBytes)))   Sockets \(daemon.openConnections)"
        } else {
            daemonLabel.stringValue = "Daemon metrics unavailable"
        }

        if let hotkeyStatus {
            hotkeyLabel.isHidden = false
            hotkeyLabel.stringValue = hotkeyStatus.statusLine
        }
    }

    // MARK: - Actions

    @objc
    private func refreshDashboard() {
        collector.refresh(force: true)
    }

    @objc
    private func quitBrainBar() {
        NSApplication.shared.terminate(nil)
    }

    // MARK: - Helpers

    private func byteString(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }

    private func verticalStack(_ views: [NSView]) -> NSStackView {
        let stack = NSStackView(views: views)
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 2
        return stack
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
        InjectionFeedView(events: store.events, filterText: $filterText)
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
    }
}

struct PopoverGraphTab: View {
    @ObservedObject var viewModel: KGViewModel

    var body: some View {
        KGCanvasView(viewModel: viewModel)
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

    init(title: String) {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = 10
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor

        titleLabel.stringValue = title
        titleLabel.font = .systemFont(ofSize: 11, weight: .medium)
        titleLabel.textColor = .secondaryLabelColor
        valueLabel.font = .systemFont(ofSize: 18, weight: .semibold)

        let stack = NSStackView(views: [titleLabel, valueLabel])
        stack.orientation = .vertical
        stack.alignment = .leading
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
