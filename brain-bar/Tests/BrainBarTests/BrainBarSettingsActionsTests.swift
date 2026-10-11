import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarSettingsActionsTests: XCTestCase {
    func testSettingsActionRoutesToUnifiedWindowWithoutCreatingLegacyWindow() {
        let app = NSApplication.shared
        let previousPolicy = app.activationPolicy()
        app.setActivationPolicy(.accessory)
        defer {
            BrainBarSettingsActions.installOpenHandler {}
            if previousPolicy != .accessory {
                app.setActivationPolicy(previousPolicy)
            }
        }

        var routedOpenCount = 0
        BrainBarSettingsActions.installOpenHandler { routedOpenCount += 1 }
        BrainBarSettingsActions.openSettingsWindow(databasePath: "/tmp/first.db")
        BrainBarSettingsActions.openSettingsWindow(databasePath: "/tmp/second.db")

        XCTAssertEqual(app.activationPolicy(), .accessory)
        XCTAssertEqual(routedOpenCount, 2)
        XCTAssertFalse(app.windows.contains { $0.title == "BrainLayer Settings" })
    }
}
