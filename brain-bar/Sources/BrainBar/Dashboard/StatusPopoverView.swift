import AppKit
import Combine
import Foundation

@MainActor
final class StatusPopoverView: NSViewController {
    private let collector: StatsCollector
    private let hotkeyStatus: HotkeyRouteStatus?
    private var cancellables: Set<AnyCancellable> = []

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

    init(collector: StatsCollector, hotkeyStatus: HotkeyRouteStatus? = nil) {
        self.collector = collector
        self.hotkeyStatus = hotkeyStatus
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = NSView(frame: NSRect(x: 0, y: 0, width: 360, height: 320))
        configureLayout()
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        bind()
        render()
    }

    private func configureLayout() {
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
        content.translatesAutoresizingMaskIntoConstraints = false

        sparklineImageView.translatesAutoresizingMaskIntoConstraints = false
        hotkeyLabel.isHidden = hotkeyStatus == nil

        view.addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 14),
            content.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -14),
            content.topAnchor.constraint(equalTo: view.topAnchor, constant: 14),
            content.bottomAnchor.constraint(lessThanOrEqualTo: view.bottomAnchor, constant: -14),
            sparklineImageView.widthAnchor.constraint(equalToConstant: 320),
            sparklineImageView.heightAnchor.constraint(equalToConstant: 42),
        ])
    }

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

    @objc
    private func refreshDashboard() {
        collector.refresh(force: true)
    }

    @objc
    private func quitBrainBar() {
        NSApplication.shared.terminate(nil)
    }

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
}

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
