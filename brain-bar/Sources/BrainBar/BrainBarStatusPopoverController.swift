import AppKit
import Combine
import SwiftUI

/// The small menu-bar status item (#963, vNext D5), like VoiceBar's: its icon draws the live
/// sparkline and badge, and every click opens a short menu. Dashboard and Settings open in the
/// one real BrainBar window.
@MainActor
final class BrainBarStatusPopoverController: NSObject, NSMenuDelegate {
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
    private let statusLineItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")

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

    /// The menu's first, informational row.
    static func statusLineTitle(for badge: BadgeStatePresentation) -> String {
        badge.badgeOn ? "Needs attention: \(badge.reason)" : "Nothing needs attention"
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
        statusItemForTesting.menu = contextMenuForTesting
        guard let button = statusItemForTesting.button else { return }
        button.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "BrainBar")
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

    func menuNeedsUpdate(_ menu: NSMenu) {
        statusLineItem.title = Self.statusLineTitle(for: badgePresentation)
    }

    private func configureContextMenu() {
        contextMenuForTesting.delegate = self
        contextMenuForTesting.autoenablesItems = false
        statusLineItem.title = Self.statusLineTitle(for: badgePresentation)
        statusLineItem.isEnabled = false
        contextMenuForTesting.addItem(statusLineItem)
        contextMenuForTesting.addItem(NSMenuItem.separator())
        contextMenuForTesting.addItem(
            NSMenuItem(
                title: "Open Dashboard",
                action: #selector(openDashboard(_:)),
                keyEquivalent: ""
            )
        )
        contextMenuForTesting.addItem(
            NSMenuItem(
                title: "Settings…",
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

    @objc private func openDashboard(_ sender: Any?) {
        dashboardPanelController.showDashboard()
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
