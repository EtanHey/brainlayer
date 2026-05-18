import AppKit
import Combine
import SwiftUI

@MainActor
final class BrainBarStatusPopoverController: NSObject {
    static let contentSize = NSSize(width: 900, height: 640)

    let statusItemForTesting: NSStatusItem
    let popoverForTesting: NSPopover

    private let runtime: BrainBarRuntime
    private var runtimeCancellables: Set<AnyCancellable> = []
    private var collectorCancellables: Set<AnyCancellable> = []

    init(runtime: BrainBarRuntime) {
        self.runtime = runtime
        statusItemForTesting = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        popoverForTesting = NSPopover()
        super.init()

        configureStatusItem()
        prewarmPopover()
        bindRuntime()
    }

    func toggle(_ sender: Any?) {
        if popoverForTesting.isShown {
            popoverForTesting.performClose(sender)
        } else {
            show(sender)
        }
    }

    func show(_ sender: Any?) {
        guard let button = statusItemForTesting.button else { return }
        popoverForTesting.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        popoverForTesting.contentViewController?.view.window?.makeKey()
    }

    func close(_ sender: Any?) {
        popoverForTesting.performClose(sender)
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
        button.toolTip = "BrainBar"
    }

    private func prewarmPopover() {
        popoverForTesting.behavior = .transient
        popoverForTesting.contentSize = Self.contentSize

        let hosting = NSHostingController(
            rootView: BrainBarWindowRootView(runtime: runtime, managesWindowFrame: false)
                .frame(width: Self.contentSize.width, height: Self.contentSize.height)
        )
        _ = hosting.view
        popoverForTesting.contentViewController = hosting
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
        let livePresentation = BrainBarLivePresentation.derive(stats: stats)
        statusItemForTesting.button?.image = SparklineRenderer.render(
            state: state,
            values: stats.recentEnrichmentBuckets,
            size: NSSize(width: 22, height: 12),
            accentColor: livePresentation.accentColor
        )
    }

    @objc private func toggleFromStatusItem(_ sender: Any?) {
        toggle(sender)
    }
}
