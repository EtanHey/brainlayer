import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarSettingsActionsTests: XCTestCase {
    func testSettingsStaysAccessoryAndTakesKeyboardFocus() {
        let databasePath = NSTemporaryDirectory() + "brainbar-settings-actions-\(UUID().uuidString).db"
        let app = NSApplication.shared
        let previousPolicy = app.activationPolicy()
        app.setActivationPolicy(.accessory)
        defer {
            app.windows.first { $0.title == "BrainLayer Settings" }?.close()
            try? FileManager.default.removeItem(atPath: databasePath)
            if previousPolicy != .accessory {
                app.setActivationPolicy(previousPolicy)
            }
        }

        BrainBarSettingsActions.openSettingsWindow(databasePath: databasePath)
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))

        let settingsWindow = BrainBarSettingsActions.windowForTesting
        XCTAssertEqual(app.activationPolicy(), .accessory)
        XCTAssertTrue(settingsWindow?.isVisible == true)
        XCTAssertTrue(settingsWindow?.canBecomeKey == true)
        XCTAssertNotNil(settingsWindow?.firstResponder)
    }
}
