import AppKit
import Combine
import SwiftUI

@MainActor
final class BrainBarStatusPopoverController: NSObject {
    static let statusItemEventMask: NSEvent.EventTypeMask = [.leftMouseUp, .rightMouseUp]

    let statusItemForTesting: NSStatusItem
    let contextMenuForTesting: NSMenu

    private let runtime: BrainBarRuntime
    private let dashboardPanelController: BrainBarDashboardPanelController
    private var runtimeCancellables: Set<AnyCancellable> = []
    private var collectorCancellables: Set<AnyCancellable> = []
    private let badgeReadQueue = DispatchQueue(label: "com.brainlayer.brainbar.badge-read", qos: .utility)
    private var badgeReadGeneration = UUID()
    private var badgePresentation = BadgeStatePresentation.failVisible("Badge state has not been read yet.")
    private var latestStats: BrainDatabase.DashboardStats?
    private var latestState: PipelineState?

    init(runtime: BrainBarRuntime, dashboardPanelController: BrainBarDashboardPanelController) {
        self.runtime = runtime
        self.dashboardPanelController = dashboardPanelController
        statusItemForTesting = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        contextMenuForTesting = NSMenu(title: "BrainBar")
        super.init()

        configureContextMenu()
        configureStatusItem()
        bindRuntime()
        dashboardPanelController.statusItemButton = statusItemForTesting.button
    }

    func toggle(_ sender: Any?) {
        dashboardPanelController.toggle(anchoredTo: statusItemForTesting.button)
    }

    func show(_ sender: Any?) {
        dashboardPanelController.show(anchoredTo: statusItemForTesting.button)
    }

    func close(_ sender: Any?) {
        dashboardPanelController.dismiss()
    }

    func stop() {
        collectorCancellables.removeAll()
        badgeReadGeneration = UUID()
        close(nil)
        NSStatusBar.system.removeStatusItem(statusItemForTesting)
    }

    private func configureStatusItem() {
        guard let button = statusItemForTesting.button else { return }
        button.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "BrainBar")
        button.target = self
        button.action = #selector(toggleFromStatusItem(_:))
        button.sendAction(on: Self.statusItemEventMask)
        button.toolTip = "BrainBar"
    }

    private func bindRuntime() {
        runtime.$collector
            .receive(on: RunLoop.main)
            .sink { [weak self] collector in
                self?.bindCollector(collector)
            }
            .store(in: &runtimeCancellables)
    }

    private func bindCollector(_ collector: StatsCollector?) {
        collectorCancellables.removeAll()
        badgeReadGeneration = UUID()
        badgePresentation = .failVisible("Badge state has not been read yet.")
        latestStats = nil
        latestState = nil
        guard let collector else { return }

        Publishers.CombineLatest(collector.$stats, collector.$state)
            .receive(on: RunLoop.main)
            .sink { [weak self] stats, state in
                self?.latestStats = stats
                self?.latestState = state
                self?.renderStatusIcon(stats: stats, state: state)
            }
            .store(in: &collectorCancellables)

        let cadence = ObservabilityReader.installedHealthCheckCadence
        let generation = badgeReadGeneration
        let history = BadgeReadHistory()
        let badgeURL = BadgeStateReader.url(dbPath: collector.databasePathForObservability)
        refreshBadge(url: badgeURL, cadence: cadence, history: history, generation: generation)
        Timer.publish(every: max(cadence.interval, 1), on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.refreshBadge(url: badgeURL, cadence: cadence, history: history, generation: generation)
            }
            .store(in: &collectorCancellables)
    }

    private func refreshBadge(
        url: URL,
        cadence: ObservabilityCadence,
        history: BadgeReadHistory,
        generation: UUID
    ) {
        badgeReadQueue.async { [weak self] in
            let badge = BadgeStateReader.read(url: url, now: Date(), cadence: cadence, history: history)
            Task { @MainActor [weak self] in
                guard let self, self.badgeReadGeneration == generation else { return }
                self.badgePresentation = badge
                if let stats = self.latestStats, let state = self.latestState {
                    self.renderStatusIcon(stats: stats, state: state)
                }
            }
        }
    }

    private func renderStatusIcon(stats: BrainDatabase.DashboardStats, state _: PipelineState) {
        let badge = badgePresentation
        // Three overlapping pipeline lines (Agent stores / JSONL watcher / Enrichment)
        // with an always-visible baseline so the icon stays legible on a dark
        // fullscreen menu bar instead of the old single gray line that vanished.
        statusItemForTesting.button?.image = SparklineRenderer.renderStatusBarIcon(
            agent: stats.recentAgentWriteBuckets,
            watcher: stats.recentWatcherWriteBuckets,
            enrichment: stats.recentEnrichmentBuckets,
            badgeOn: badge.badgeOn,
            size: NSSize(width: 26, height: 14)
        )
        statusItemForTesting.button?.toolTip = badge.badgeOn
            ? "BrainBar — needs attention: \(badge.reason)"
            : "BrainBar"
    }

    @objc private func toggleFromStatusItem(_ sender: Any?) {
        if let event = NSApp.currentEvent, event.type == .rightMouseUp,
           let button = statusItemForTesting.button {
            NSMenu.popUpContextMenu(contextMenuForTesting, with: event, for: button)
            return
        }

        toggle(sender)
    }

    private func configureContextMenu() {
        contextMenuForTesting.addItem(
            NSMenuItem(
                title: "Toggle BrainBar",
                action: #selector(toggleFromContextMenu(_:)),
                keyEquivalent: ""
            )
        )
        contextMenuForTesting.addItem(
            NSMenuItem(
                title: "Settings...",
                action: #selector(openSettings(_:)),
                keyEquivalent: ""
            )
        )
        contextMenuForTesting.addItem(NSMenuItem.separator())
        contextMenuForTesting.addItem(
            NSMenuItem(
                title: "Restart BrainBar",
                action: #selector(restartBrainBar(_:)),
                keyEquivalent: ""
            )
        )
        contextMenuForTesting.addItem(NSMenuItem.separator())
        contextMenuForTesting.addItem(
            NSMenuItem(
                title: "Quit BrainBar",
                action: #selector(quitBrainBar(_:)),
                keyEquivalent: ""
            )
        )

        for item in contextMenuForTesting.items where item.action != nil {
            item.target = self
        }
    }

    @objc private func toggleFromContextMenu(_ sender: Any?) {
        toggle(sender)
    }

    @objc private func restartBrainBar(_ sender: Any?) {
        BrainBarProcessControl.restart()
    }

    @objc private func openSettings(_ sender: Any?) {
        BrainBarSettingsActions.openSettingsWindow(databasePath: runtime.databasePath)
    }

    @objc private func quitBrainBar(_ sender: Any?) {
        BrainBarProcessControl.quit()
    }
}
