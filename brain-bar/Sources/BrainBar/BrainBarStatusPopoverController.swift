import AppKit
import Combine
import SwiftUI

/// Tests supply an in-memory host: AppKit only supports the system NSStatusBar,
/// not separately allocated bars. Keep that native lifecycle behind this seam.
@MainActor
protocol BrainBarStatusItem: AnyObject {
    var length: CGFloat { get }
    var menu: NSMenu? { get set }
    var button: NSButton? { get }
}

@MainActor
protocol BrainBarStatusItemHosting: AnyObject {
    func statusItem(withLength length: CGFloat) -> any BrainBarStatusItem
    func removeStatusItem(_ item: any BrainBarStatusItem)
}

@MainActor
final class SystemBrainBarStatusItemHost: BrainBarStatusItemHosting {
    private let bar = NSStatusBar.system

    private final class Item: BrainBarStatusItem {
        let native: NSStatusItem
        init(_ native: NSStatusItem) { self.native = native }
        var length: CGFloat { native.length }
        var menu: NSMenu? {
            get { native.menu }
            set { native.menu = newValue }
        }
        var button: NSButton? { native.button }
    }

    func statusItem(withLength length: CGFloat) -> any BrainBarStatusItem {
        Item(bar.statusItem(withLength: length))
    }

    func removeStatusItem(_ item: any BrainBarStatusItem) {
        guard let item = item as? Item else { preconditionFailure("Status item belongs to another host") }
        bar.removeStatusItem(item.native)
    }
}

/// The small menu-bar status item (#963, vNext D5), like VoiceBar's: its icon draws the live
/// sparkline and badge, and every click opens a short menu. Dashboard and Settings open in the
/// one real BrainBar window.
@MainActor
final class BrainBarStatusPopoverController: NSObject, NSMenuDelegate {
    let statusItemForTesting: any BrainBarStatusItem
    let contextMenuForTesting: NSMenu

#if BRAINBAR_UI
    private var logPathsOverride: BrainBarBackupSources.Paths?
    private var logNote: (key: String, reason: String, message: String)?
    private var activeLogKey: String?
    private var activeLogReason = ""
    private let logNoteItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private var backupReceiptItems: [NSMenuItem] = []
    private var logPaths: BrainBarBackupSources.Paths {
        logPathsOverride ?? .live(databasePath: runtime.databasePath ?? BrainBarServer.defaultDBPath())
    }
#endif
    private let workspace: any BrainBarWorkspaceActing
    private let runtime: BrainBarRuntime
    private let dashboardPanelController: BrainBarDashboardPanelController
    private let statusBar: any BrainBarStatusItemHosting
    private var isStopped = false
    private var isMenuTracking = false
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
    /// Shown for a job alert or a structured backup failure.
    private let showLogItem = NSMenuItem(title: BrainBarStatusPopoverController.showLogTitle, action: nil, keyEquivalent: "")

    init(
        workspace: any BrainBarWorkspaceActing,
        runtime: BrainBarRuntime,
        dashboardPanelController: BrainBarDashboardPanelController,
        statusBar: any BrainBarStatusItemHosting
    ) {
        self.workspace = workspace
        self.runtime = runtime
        self.dashboardPanelController = dashboardPanelController
        self.statusBar = statusBar
        statusItemForTesting = statusBar.statusItem(withLength: NSStatusItem.variableLength)
        contextMenuForTesting = NSMenu(title: "BrainBar")
        super.init()

        configureContextMenu()
        configureStatusItem()
        bindRuntime()
        dashboardPanelController.statusItemButton = statusItemForTesting.button
    }

    /// Includes the attention prefix and ellipsis, so long errors cannot widen the menu.
    private static let statusLineTitleMaxCharacters = 60

    /// The menu's first, informational row; its tooltip retains the full reason.
    static func statusLineTitle(for badge: BadgeStatePresentation) -> String {
        guard badge.badgeOn || !badge.activeMessages.isEmpty else { return "Nothing needs attention" }
        let prefix = badge.badgeOn ? "Needs attention: " : "Info: "
        let title = prefix + badge.reason
        guard title.count > statusLineTitleMaxCharacters else { return title }

        let reasonPrefix = badge.reason.prefix(statusLineTitleMaxCharacters - prefix.count - 1)
        let boundary = reasonPrefix.lastIndex(where: { $0.isWhitespace }) ?? reasonPrefix.endIndex
        let shortenedReason = String(reasonPrefix[..<boundary]).trimmingCharacters(in: .whitespacesAndNewlines)
        return prefix + shortenedReason + "…"
    }

    /// The menu's first row from the shared snapshot (S5).
    nonisolated static func statusLineTitle(for verdict: BrainBarStatusVerdict) -> String {
        "Nothing needs attention"
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
    /// An action label, never a claim that a file or folder has already opened.
    static func showLogItemTitle(
        for badge: BadgeStatePresentation,
        paths: BrainBarBackupSources.Paths,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> String? {
        guard let key = jobAlertKey(for: badge) else { return nil }
        return showLogItemTitle(forKey: key, paths: paths, fileExists: fileExists)
    }

    private static func showLogItemTitle(
        forKey key: String, paths: BrainBarBackupSources.Paths,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> String {
        if let log = BrainBarJobAlerts.logURL(forKey: key, paths: paths), fileExists(log) { return showLogTitle }
        return "Open logs folder"
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
        guard !isStopped else { return }
        isStopped = true
        runtimeCancellables.removeAll()
        collectorCancellables.removeAll()
        badgeReadGeneration = UUID()
        NotificationCenter.default.removeObserver(self, name: NSMenu.didBeginTrackingNotification, object: contextMenuForTesting)
        contextMenuForTesting.cancelTracking()
        isMenuTracking = false
        close(nil)
        dashboardPanelController.statusItemButton = nil
        contextMenuForTesting.delegate = nil
        statusItemForTesting.menu = nil
        statusBar.removeStatusItem(statusItemForTesting)
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
        guard !isStopped else { return }
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
        // Setting NSStatusBarButton.image relayouts its variable-length status item.
        // Keep AppKit's tracking anchor stable; collectors still cache the latest state.
        guard !isStopped, !isMenuTracking else { return }
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
        guard !isStopped, !isMenuTracking else { return }
        statusItemForTesting.button?.toolTip = badgePresentation.badgeOn
            ? "BrainBar — needs attention: \(badgePresentation.reason)"
            : "BrainBar"
    }

#if DEBUG
#if BRAINBAR_UI
    func setLogPathsForTesting(_ paths: BrainBarBackupSources.Paths) { logPathsOverride = paths }
#endif
    var statusDotForTesting: Bool { badgePresentation.badgeOn }
    func setBadgeForTesting(_ raw: BadgeStatePresentation, alertsURL: URL) {
        rawBadgePresentation = raw
        badgePresentation = raw
        jobAlertsURL = alertsURL
    }
#endif

    @objc private func menuTrackingDidBegin(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu else { return }
        menuWillOpen(menu)
    }

    func menuWillOpen(_ menu: NSMenu) {
        guard menu === contextMenuForTesting, !isStopped else { return }
        isMenuTracking = true
    }

    func menuDidClose(_ menu: NSMenu) {
        guard menu === contextMenuForTesting, isMenuTracking else { return }
        isMenuTracking = false
        // menuDidClose is still inside AppKit's tracking teardown. Apply only after
        // it returns, and never flush a prior close into an immediately reopened menu.
        DispatchQueue.main.async { [weak self] in
            guard let self, !self.isStopped, !self.isMenuTracking else { return }
            if let stats = self.latestStats, let state = self.latestState {
                self.renderStatusIcon(stats: stats, state: state)
            } else {
                self.updateTooltip()
            }
        }
    }

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
        var menuBadge = badgePresentation
#if BRAINBAR_UI
        activeLogKey = Self.jobAlertKey(for: badgePresentation)
        activeLogReason = badgePresentation.reason
        for item in backupReceiptItems { menu.removeItem(item) }
        backupReceiptItems.removeAll()
        if let model = runtime.jobsModel {
            // The single menu action follows desired-ON attention; named screen rows retain history.
            if activeLogKey == nil, let failure = BrainBarBackupLogTarget.failures(in: model.config).first(where: {
                model.config.launchdJobs[$0.job]?.enabled != false
            }) {
                activeLogKey = failure.key
                activeLogReason = failure.reason
            }
            let drive = DriveAuthPresentation.derive(status: runtime.driveAuth.status,
                isReconnecting: runtime.driveAuth.isReconnecting, lastOutcome: runtime.driveAuth.lastOutcome,
                now: model.observabilityNow, formatDate: BrainBarDriveAuthFormat.date)
            let health = model.backupsHealth(drive: drive)
            if !menuBadge.badgeOn, let reason = health.reason,
               health.badge == .attention || menuBadge.activeMessages.isEmpty {
                menuBadge = health.badge == .attention ? .failVisible(reason)
                    : .init(badgeOn: false, reason: reason, activeCodes: [], activeMessages: [reason])
            }
            // This property derives from the published document AND observabilityNow shared
            // with Dashboard and Backups. Opening the menu never starts another receipt read.
            for (index, line) in (model.backupReceiptLines ?? []).enumerated() {
                let item = NSMenuItem(title: line.text, action: nil, keyEquivalent: "")
                item.identifier = NSUserInterfaceItemIdentifier("brainbar.menu.backup.\(index)")
                item.isEnabled = false
                item.toolTip = "observability.json · database snapshot creation / transcript upload receipt · shared check clock"
                let color: NSColor = switch line.tone {
                case .green: BrainBarDesignTokens.Colors.statusOK
                case .red: BrainBarDesignTokens.Colors.statusError
                case .neutral: BrainBarDesignTokens.Colors.statusUnknown
                }
                item.attributedTitle = NSAttributedString(string: line.text, attributes: [.foregroundColor: color])
                menu.insertItem(item, at: 1 + index)
                backupReceiptItems.append(item)
            }
        }
#endif
        statusLineItem.title = Self.statusLineTitle(for: menuBadge)
        statusLineItem.toolTip = menuBadge.reason.isEmpty ? nil : menuBadge.reason
#if BRAINBAR_UI
        let key = activeLogKey
        showLogItem.isHidden = key == nil
        if let key {
            showLogItem.title = Self.showLogItemTitle(forKey: key, paths: logPaths)
            showLogItem.isEnabled = true
            showLogItem.toolTip = showLogItem.title == Self.showLogTitle ? nil : BrainBarJobAlerts.missingLogMessage(forKey: key)
        }
        if logNote?.key != key || logNote?.reason != activeLogReason { logNote = nil }
        logNoteItem.title = logNote?.message ?? ""
        logNoteItem.toolTip = logNote?.message
        logNoteItem.isHidden = logNote == nil
#endif
    }

    private func configureContextMenu() {
        contextMenuForTesting.delegate = self
        // Native tracking begins before menuNeedsUpdate reconciles badge/job alerts.
        NotificationCenter.default.addObserver(self, selector: #selector(menuTrackingDidBegin(_:)),
            name: NSMenu.didBeginTrackingNotification, object: contextMenuForTesting)
        contextMenuForTesting.autoenablesItems = false
        statusLineItem.title = Self.statusLineTitle(for: badgePresentation)
        statusLineItem.toolTip = badgePresentation.reason.isEmpty ? nil : badgePresentation.reason
        statusLineItem.isEnabled = false
        contextMenuForTesting.addItem(statusLineItem)
#if BRAINBAR_UI
        showLogItem.action = #selector(showJobAlertLog(_:))
        showLogItem.isHidden = Self.jobAlertKey(for: badgePresentation) == nil
        contextMenuForTesting.addItem(showLogItem)
        logNoteItem.isEnabled = false
        logNoteItem.isHidden = true
        contextMenuForTesting.addItem(logNoteItem)
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
        menuNeedsUpdate(contextMenuForTesting)
        guard let key = activeLogKey else { return }
        let result = BrainBarJobAlerts.showLog(forKey: key, paths: logPaths, workspace: workspace)
        logNote = result.message.map { (key, activeLogReason, $0) }
        menuNeedsUpdate(contextMenuForTesting)
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
