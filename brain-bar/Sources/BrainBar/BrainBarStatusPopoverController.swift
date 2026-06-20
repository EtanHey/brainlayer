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
        guard let collector else { return }

        Publishers.CombineLatest(collector.$stats, collector.$state)
            .receive(on: RunLoop.main)
            .sink { [weak self] stats, state in
                self?.renderStatusIcon(stats: stats, state: state)
            }
            .store(in: &collectorCancellables)
    }

    private func renderStatusIcon(stats: BrainDatabase.DashboardStats, state: PipelineState) {
        // Three overlapping pipeline lines (Agent stores / JSONL watcher / Enrichment)
        // with an always-visible baseline so the icon stays legible on a dark
        // fullscreen menu bar instead of the old single gray line that vanished.
        statusItemForTesting.button?.image = SparklineRenderer.renderStatusBarIcon(
            agent: stats.recentAgentWriteBuckets,
            watcher: stats.recentWatcherWriteBuckets,
            enrichment: stats.recentEnrichmentBuckets,
            size: NSSize(width: 26, height: 14)
        )
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

    @objc private func restartBrainBar(_ sender: Any?) {
        BrainBarProcessControl.restart()
    }

    @objc private func openSettings(_ sender: Any?) {
        BrainBarSettingsActions.openSettingsWindow()
    }

    @objc private func quitBrainBar(_ sender: Any?) {
        BrainBarProcessControl.quit()
    }
}
