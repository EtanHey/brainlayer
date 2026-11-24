import AppKit
import XCTest
@testable import BrainBar

/// #963 PR 2: one window whose sidebar is Dashboard · Jobs · Backups · Advanced. The top
/// Dashboard/Settings tabs are gone, and the empty General page is absorbed into Dashboard.
@MainActor
final class BrainBarOneWindowSidebarTests: XCTestCase {
    private func controller() -> BrainBarDashboardPanelController {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        return BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime, designDefaults: brainBarIsolatedDefaults())
    }

    func test_the_sidebar_is_dashboard_then_the_settings_sections() {
        XCTAssertEqual(BrainBarSidebarItem.allCases.map(\.title), ["Dashboard", "Jobs", "Backups", "Advanced"])
        XCTAssertEqual(BrainBarSettingsSection.allCases.map(\.rawValue), ["jobs", "backups", "advanced"], "General is absorbed")
    }

}
