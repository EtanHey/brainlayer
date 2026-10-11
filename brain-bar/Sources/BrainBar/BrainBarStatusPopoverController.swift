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
    /// The badge as read, before reconciling with the job-alert state; re-reconciled on menu open.
    private var rawBadgePresentation: BadgeStatePresentation?
    private var jobAlertsURL: URL?
    private var latestStats: BrainDatabase.DashboardStats?
    private var latestState: PipelineState?
    private let statusLineItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    /// Shown under the status line only while a job alert is active.
    private let showLogItem = NSMenuItem(title: BrainBarStatusPopoverController.showLogTitle, action: nil, keyEquivalent: "")

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

    /// The failing job behind an active job-alert badge code (`job_alert_<key>`), or nil.
    static func jobAlertKey(for badge: BadgeStatePresentation) -> String? {
        guard badge.badgeOn else { return nil }
        return badge.activeCodes.first { $0.hasPrefix(jobAlertCodePrefix) }.map { String($0.dropFirst(jobAlertCodePrefix.count)) }
    }

    private static let jobAlertCodePrefix = "job_alert_"
    static let showLogTitle = "Show log"

    /// The menu's rows as titles, in order; "" is a separator. The render harness draws these.
    /// `showLogItem` is the Show log row's title, when it differs from "Show log".
    static func menuRowTitles(for badge: BadgeStatePresentation, showLogItem: String? = nil) -> [String] {
        [statusLineTitle(for: badge)] + (jobAlertKey(for: badge) == nil ? [] : [showLogItem ?? showLogTitle])
            + ["", "Open Dashboard", "Settings…", "", "Restart BrainBar", "", "Quit BrainBar"]
    }

#if BRAINBAR_UI
    /// The Show log row's title: "Show log" when the job's log exists, else the sentence saying
    /// there is none yet (the row is then informational). Nil when no job alert is active.
    static func showLogItemTitle(
        for badge: BadgeStatePresentation,
        paths: BrainBarBackupSources.Paths,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> String? {
        guard let key = jobAlertKey(for: badge) else { return nil }
        if let log = BrainBarJobAlerts.logURL(forKey: key, paths: paths), fileExists(log) { return showLogTitle }
        return BrainBarJobAlerts.missingLogMessage(forKey: key)
    }
#endif

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
        rawBadgePresentation = nil
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
        let alertsURL = BrainBarJobAlerts.producerURL(
            dbPath: collector.databasePathForObservability, producerLabel: BrainBarJobAlerts.badgeProducerLabel
        )
        jobAlertsURL = alertsURL
        refreshBadge(url: badgeURL, alertsURL: alertsURL, cadence: cadence, history: history, generation: generation)
        Timer.publish(every: max(cadence.interval, 1), on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.refreshBadge(url: badgeURL, alertsURL: alertsURL, cadence: cadence, history: history, generation: generation)
            }
            .store(in: &collectorCancellables)
    }

    private func refreshBadge(
        url: URL,
        alertsURL: URL,
        cadence: ObservabilityCadence,
        history: BadgeReadHistory,
        generation: UUID
    ) {
        badgeReadQueue.async { [weak self] in
            let raw = BadgeStateReader.read(url: url, now: Date(), cadence: cadence, history: history)
            // B2: a recovered job alert clears here exactly as on Backups and the Dashboard.
            let badge = raw.reconciled(with: BrainBarJobAlerts.read(url: alertsURL))
            Task { @MainActor [weak self] in
                guard let self, self.badgeReadGeneration == generation else { return }
                self.rawBadgePresentation = raw
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
        statusItemForTesting.button?.image = Self.statusIconImage(stats: stats, badgeOn: badge.badgeOn)
        updateTooltip()
    }

    static func statusIconImage(stats: BrainDatabase.DashboardStats, badgeOn: Bool) -> NSImage {
        return SparklineRenderer.renderStatusBarIcon(
            agent: stats.recentAgentWriteBuckets,
            watcher: stats.recentWatcherWriteBuckets,
            badgeOn: badgeOn,
            size: NSSize(width: 26, height: 14)
        )
    }

    private func updateTooltip() {
        statusItemForTesting.button?.toolTip = badgePresentation.badgeOn
            ? "BrainBar — needs attention: \(badgePresentation.reason)"
            : "BrainBar"
    }

#if DEBUG
    func setBadgeForTesting(_ raw: BadgeStatePresentation, alertsURL: URL) {
        rawBadgePresentation = raw
        badgePresentation = raw
        jobAlertsURL = alertsURL
    }
#endif

    func menuNeedsUpdate(_ menu: NSMenu) {
        // The alert file can clear between badge reads; the open menu reads it now.
        if let raw = rawBadgePresentation, let jobAlertsURL {
            let reconciled = raw.reconciled(with: BrainBarJobAlerts.read(url: jobAlertsURL))
            if reconciled != badgePresentation {
                badgePresentation = reconciled
                // The icon and its tooltip follow the menu (Macroscope #1062), not the next read.
                if let stats = latestStats, let state = latestState { renderStatusIcon(stats: stats, state: state) }
                updateTooltip()
            }
        }
        statusLineItem.title = Self.statusLineTitle(for: badgePresentation)
        showLogItem.isHidden = Self.jobAlertKey(for: badgePresentation) == nil
#if BRAINBAR_UI
        if let title = Self.showLogItemTitle(
            for: badgePresentation, paths: .live(databasePath: runtime.databasePath ?? BrainBarServer.defaultDBPath())
        ) {
            showLogItem.title = title
            showLogItem.isEnabled = title == Self.showLogTitle
        }
#endif
    }

    private func configureContextMenu() {
        contextMenuForTesting.delegate = self
        contextMenuForTesting.autoenablesItems = false
        statusLineItem.title = Self.statusLineTitle(for: badgePresentation)
        statusLineItem.isEnabled = false
        contextMenuForTesting.addItem(statusLineItem)
#if BRAINBAR_UI
        showLogItem.action = #selector(showJobAlertLog(_:))
        showLogItem.isHidden = Self.jobAlertKey(for: badgePresentation) == nil
        contextMenuForTesting.addItem(showLogItem)
#endif
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

#if BRAINBAR_UI
    @objc private func showJobAlertLog(_ sender: Any?) {
        BrainBarJobAlerts.showLog(
            forKey: Self.jobAlertKey(for: badgePresentation),
            paths: .live(databasePath: runtime.databasePath ?? BrainBarServer.defaultDBPath()),
            workspace: BrainBarWorkspace()
        )
    }
#endif

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
