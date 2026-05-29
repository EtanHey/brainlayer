import XCTest
@testable import BrainBarDaemon

final class BrainBarDaemonLaunchModeTests: XCTestCase {
    func testDaemonLaunchModeParsesUnifiedEnvironmentValues() {
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_LAUNCH_MODE": "app-window"],
                defaults: FakeKeyValueStore()
            ),
            .appWindow
        )

        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_LAUNCH_MODE": "menu-item-daemon"],
                defaults: FakeKeyValueStore()
            ),
            .menuItemDaemon
        )
    }

    func testDaemonLaunchModePreservesLegacyEnvironmentCompatibility() {
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_APP_WINDOW": "1"],
                defaults: FakeKeyValueStore()
            ),
            .appWindow
        )

        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_LEGACY": "1"],
                defaults: FakeKeyValueStore()
            ),
            .menuItemDaemon
        )
    }

    func testDaemonLaunchModeEnvironmentOverridesPersistedPreference() {
        let store = FakeKeyValueStore()
        BrainBarLaunchMode.setPreferred(.menuItemDaemon, defaults: store)

        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_LAUNCH_MODE": "APPWINDOW"],
                defaults: store
            ),
            .appWindow
        )
    }

    func testDaemonLaunchModeReadsPersistedPreferenceWrittenByUI() {
        let store = FakeKeyValueStore()

        BrainBarLaunchMode.setPreferred(.appWindow, defaults: store)

        XCTAssertEqual(
            store.string(forKey: BrainBarLaunchMode.defaultsKey),
            "app-window"
        )
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(environment: [:], defaults: store),
            .appWindow
        )
    }

    func testDaemonLaunchModeFallsBackToMenuItemDaemonForInvalidInputs() {
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_LAUNCH_MODE": "not-a-mode"],
                defaults: FakeKeyValueStore()
            ),
            .menuItemDaemon
        )

        XCTAssertEqual(
            BrainBarLaunchMode.resolve(
                environment: ["BRAINBAR_LAUNCH_MODE": ""],
                defaults: FakeKeyValueStore()
            ),
            .menuItemDaemon
        )
    }

    @MainActor
    func testDaemonRuntimeRoutesToggleThroughUnifiedWindowHandler() {
        var toggles = 0
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        runtime.onToggleRequested = {
            toggles += 1
        }

        runtime.handleToggleRequest()

        XCTAssertEqual(toggles, 1)
    }

    @MainActor
    func testDaemonRuntimeDefaultInitializerResolvesLaunchMode() {
        let runtime = BrainBarRuntime()

        XCTAssertEqual(runtime.launchMode, .menuItemDaemon)
    }
}

private final class FakeKeyValueStore: BrainBarKeyValueStoring {
    private var values: [String: String] = [:]

    func string(forKey defaultName: String) -> String? {
        values[defaultName]
    }

    func setString(_ value: String?, forKey defaultName: String) {
        values[defaultName] = value
    }
}
