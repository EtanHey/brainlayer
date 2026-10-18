import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarSettingsEntryTests: XCTestCase {
    func testDashboardToBackupsRefreshesRetainedEvidenceAndPreservesDrafts() async throws {
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath())
            .appendingPathComponent("settings-entry-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("fixture.env"))
        try store.save(.defaultConfig)
        let url = root.appendingPathComponent("observability.json")
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            refreshStatusOnLoad: false, observabilityURL: url,
            observabilityRead: { readURL in
                // Decode this fixture only: readReconciled also resolves installed job-alert inputs.
                XCTAssertEqual(readURL, url)
                guard readURL == url else { return .unreadable("Unexpected fixture URL") }
                return ObservabilityReader.read(url: readURL)
            })
        await settle { model.hasCompletedObservabilityRead }
        XCTAssertNotNil(model.backupStatusReason)
        model.onePasswordReference = "op://synthetic/draft/key"
        model.pendingPlainAPIKey = "synthetic-unsaved-draft"
        let runtime = BrainBarRuntime()
        runtime.jobsModel = model
        let panel = BrainBarDashboardPanelState()
        let host = NSHostingView(rootView: BrainBarWindowRootView(
            workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            managesWindowFrame: false, panelState: panel))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 720),
            styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        // The producer and external configuration recover while Dashboard is selected.
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "observability-main-58849a70",
            withExtension: "json", subdirectory: "Fixtures"))
        try Data(contentsOf: fixture).write(to: url)
        var external = try store.loadDocument().config
        external.systemEnabled = false
        try store.save(external)
        panel.select(.backups)
        host.layoutSubtreeIfNeeded()
        await settle { model.backupStatus != nil && !model.config.systemEnabled }
        XCTAssertNotNil(model.backupStatus, "First insertion must read recovered observability")
        XCTAssertFalse(model.config.systemEnabled, "First insertion must reload external config")
        XCTAssertEqual(panel.settingsActivationRevision, 0, "Sidebar entry does not reshow the window")
        XCTAssertEqual(model.onePasswordReference, "op://synthetic/draft/key")
        XCTAssertEqual(model.pendingPlainAPIKey, "synthetic-unsaved-draft")
        // Re-entry must also discover a new failure, rather than retaining the recovery.
        panel.select(.dashboard)
        await settle { panel.selectedTab == .dashboard }
        try Data("invalid synthetic observability".utf8).write(to: url)
        external.systemEnabled = true
        try store.save(external)
        panel.select(.backups)
        host.layoutSubtreeIfNeeded()
        await settle { model.backupStatus == nil && model.config.systemEnabled }
        XCTAssertNil(model.backupStatus)
        XCTAssertTrue(model.config.systemEnabled)
        XCTAssertNotNil(model.backupStatusReason)
        XCTAssertEqual(model.pendingPlainAPIKey, "synthetic-unsaved-draft")
    }

    func testCheckingJobsDoNotInventAttentionButDisabledJobsStillDo() throws {
        let store = BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/unused/synthetic.env"),
            loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) })
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            refreshStatusOnLoad: false)
        XCTAssertEqual(model.groupStatus(.maintenance).health, .checking)
        XCTAssertTrue(model.dashboardJobAttentionReasons.isEmpty)
        model.config.launchdJobs[.maintenanceWeekly]?.enabled = false
        XCTAssertEqual(model.groupStatus(.maintenance).health, .unhealthy)
        XCTAssertEqual(model.dashboardJobAttentionReasons.count, 1)
    }

    private func settle(_ ready: () -> Bool) async {
        // Always yield once so SwiftUI processes a sidebar transition even for a ready predicate.
        for _ in 0..<100 {
            try? await Task.sleep(for: .milliseconds(10))
            if ready() { return }
        }
    }
}
