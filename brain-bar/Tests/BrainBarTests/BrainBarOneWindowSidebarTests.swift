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
        return BrainBarDashboardPanelController(runtime: runtime)
    }

    func test_the_sidebar_is_dashboard_then_the_settings_sections() {
        XCTAssertEqual(BrainBarSidebarItem.allCases.map(\.title), ["Dashboard", "Jobs", "Backups", "Advanced"])
        XCTAssertEqual(BrainBarSettingsSection.allCases.map(\.rawValue), ["jobs", "backups", "advanced"], "General is absorbed")
    }

    func test_every_route_maps_to_a_sidebar_selection() throws {
        let controller = controller()
        defer {
            controller.dismiss()
            BrainBarSettingsActions.installOpenHandler {}
        }
        let routes: [(String, BrainBarSidebarItem)] = [
            ("brainbar://settings/backups", .backups),
            ("brainbar://dashboard", .dashboard),
            ("brainbar://settings/jobs", .jobs),
            ("brainbar://settings/advanced", .advanced),
            ("brainbar://settings/general", .dashboard),
            ("brainbar://settings", .jobs),
            ("brainbar://settings/unknown", .jobs),
        ]
        for (url, expected) in routes {
            let action = try XCTUnwrap(BrainBarURLAction.parse(url: URL(string: url)!), url)
            controller.showURLDestination(action)
            XCTAssertTrue(controller.isShownForTesting, url)
            XCTAssertEqual(controller.sidebarSelectionForTesting, expected, url)
        }
        // Cmd-, and "Settings…" reopen the settings section the user was last on.
        controller.showURLDestination(.settings(.advanced))
        controller.showDashboard()
        BrainBarSettingsActions.openSettingsWindow(databasePath: nil)
        XCTAssertEqual(controller.sidebarSelectionForTesting, .advanced)
    }

    func test_selecting_a_sidebar_item_shows_its_page() {
        let controller = controller()
        controller.selectSidebarItemForTesting(.backups)
        XCTAssertEqual(controller.selectedTabForTesting, .settings)
        XCTAssertEqual(controller.selectedSettingsSectionForTesting, .backups)
        controller.selectSidebarItemForTesting(.dashboard)
        XCTAssertEqual(controller.selectedTabForTesting, .dashboard)
        XCTAssertEqual(controller.sidebarSelectionForTesting, .dashboard)
        controller.selectSidebarItemForTesting(.jobs)
        XCTAssertEqual(controller.sidebarSelectionForTesting, .jobs)
    }

    func test_the_window_has_no_top_tabs() throws {
        let controller = controller()
        let window = controller.windowForTesting
        window.setContentSize(NSSize(width: 960, height: 640))
        let content = try XCTUnwrap(window.contentView)
        content.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.4))
        content.layoutSubtreeIfNeeded()

        func segmentedControls(in view: NSView) -> [NSSegmentedControl] {
            (view as? NSSegmentedControl).map { [$0] } ?? [] + view.subviews.flatMap(segmentedControls)
        }
        XCTAssertEqual(segmentedControls(in: content).count, 0, "the Dashboard/Settings segmented tabs are replaced by the sidebar")
    }
}
