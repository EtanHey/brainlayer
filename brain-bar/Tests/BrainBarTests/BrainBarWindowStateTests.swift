import CoreGraphics
import Foundation
import XCTest
@testable import BrainBar

final class BrainBarWindowStateTests: XCTestCase {
    func testLaunchModeDefaultsToMenuBarWindow() {
        XCTAssertEqual(BrainBarLaunchMode.resolve(environment: [:]), .menuBarWindow)
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(environment: ["BRAINBAR_LEGACY": "0"]),
            .menuBarWindow
        )
    }

    func testLaunchModeUsesLegacyModeWhenEscapeHatchEnabled() {
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(environment: ["BRAINBAR_LEGACY": "1"]),
            .legacyStatusItem
        )
        XCTAssertEqual(
            BrainBarLaunchMode.resolve(environment: ["BRAINBAR_LEGACY": "true"]),
            .legacyStatusItem
        )
    }

    func testBrainBarTabsCoverDashboardInjectionsAndGraph() {
        XCTAssertEqual(BrainBarTab.allCases, [.dashboard, .injections, .graph])
        XCTAssertEqual(BrainBarTab.dashboard.title, "Dashboard")
        XCTAssertEqual(BrainBarTab.injections.title, "Injections")
        XCTAssertEqual(BrainBarTab.graph.title, "Graph")
    }

    func testLivePresentationShowsRateBadgeWhenEnrichmentIsActive() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 24,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [0, 0, 0, 1, 3]
        )

        let presentation = BrainBarLivePresentation.derive(stats: stats)

        XCTAssertEqual(presentation.sparklineStyle, .active)
        XCTAssertEqual(presentation.badgeText, "24/min")
    }

    func testLivePresentationShowsIdleBadgeWhenEnrichmentRateDropsToZero() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [4, 3, 1, 0, 0]
        )

        let presentation = BrainBarLivePresentation.derive(stats: stats)

        XCTAssertEqual(presentation.sparklineStyle, .idle)
        XCTAssertEqual(presentation.badgeText, "idle")
    }
}

@MainActor
final class BrainBarWindowCoordinatorTests: XCTestCase {
    func testCoordinatorRestoresPersistedFrameWhenWindowAttaches() {
        let store = FakeKeyValueStore()
        let persisted = CGRect(x: 120, y: 80, width: 720, height: 560)
        let frameStore = BrainBarWindowFrameStore(defaults: store)
        frameStore.persist(frame: persisted)

        let coordinator = BrainBarWindowCoordinator(
            frameStore: frameStore,
            screenFramesProvider: { [CGRect(x: 0, y: 0, width: 1440, height: 900)] }
        )
        let window = FakeWindowHandle(frame: .zero)

        coordinator.attach(window: window)

        XCTAssertEqual(window.frame, persisted)
    }

    func testCoordinatorCentersDefaultFrameWhenNoPersistedFrameExists() {
        let coordinator = BrainBarWindowCoordinator(
            frameStore: BrainBarWindowFrameStore(defaults: FakeKeyValueStore()),
            screenFramesProvider: { [CGRect(x: 0, y: 0, width: 1440, height: 900)] }
        )
        let window = FakeWindowHandle(frame: .zero)

        coordinator.attach(window: window)

        XCTAssertEqual(window.frame, CGRect(x: 270, y: 130, width: 900, height: 640))
    }

    func testCoordinatorReplacesOffscreenPersistedFrameWithCenteredDefault() {
        let store = FakeKeyValueStore()
        let persisted = CGRect(x: 2_022, y: -218, width: 900, height: 640)
        let frameStore = BrainBarWindowFrameStore(defaults: store)
        frameStore.persist(frame: persisted)

        let coordinator = BrainBarWindowCoordinator(
            frameStore: frameStore,
            screenFramesProvider: { [CGRect(x: 0, y: 0, width: 1440, height: 900)] }
        )
        let window = FakeWindowHandle(frame: .zero)

        coordinator.attach(window: window)

        XCTAssertEqual(window.frame, CGRect(x: 270, y: 130, width: 900, height: 640))
    }

    func testAccessibilityFrameRoundTripsToAppKitCoordinates() {
        let screenFrames = [CGRect(x: 0, y: 0, width: 1440, height: 900)]
        let appKitFrame = CGRect(x: 240, y: 200, width: 760, height: 560)

        let accessibilityFrame = BrainBarWindowPlacement.accessibilityFrame(
            fromAppKit: appKitFrame,
            screenFrames: screenFrames
        )
        let roundTrippedFrame = accessibilityFrame.flatMap {
            BrainBarWindowPlacement.appKitFrame(
                fromAccessibility: $0,
                screenFrames: screenFrames
            )
        }

        XCTAssertEqual(roundTrippedFrame, appKitFrame)
    }

    func testAccessibilityFrameRoundTripsAcrossRaisedSecondaryDisplay() {
        let screenFrames = [
            CGRect(x: 0, y: 0, width: 1512, height: 982),
            CGRect(x: 1512, y: 355, width: 1920, height: 1080),
        ]
        let appKitFrame = CGRect(x: 2065, y: 767, width: 900, height: 640)

        let accessibilityFrame = BrainBarWindowPlacement.accessibilityFrame(
            fromAppKit: appKitFrame,
            screenFrames: screenFrames
        )
        let roundTrippedFrame = accessibilityFrame.flatMap {
            BrainBarWindowPlacement.appKitFrame(
                fromAccessibility: $0,
                screenFrames: screenFrames
            )
        }

        XCTAssertEqual(accessibilityFrame?.origin.y, -425)
        XCTAssertEqual(roundTrippedFrame, appKitFrame)
    }

    func testMenuBarClearanceShiftsWindowDownWhenTopEdgeTouchesVisibleFrame() {
        let screenFrames = [CGRect(x: 0, y: 0, width: 1440, height: 900)]
        let anchoredFrame = CGRect(x: 300, y: 340, width: 760, height: 560)

        let adjustedFrame = BrainBarWindowPlacement.clearingMenuBar(
            frame: anchoredFrame,
            screenFrames: screenFrames,
            topGap: 18
        )

        XCTAssertEqual(adjustedFrame.origin.y, 322)
        XCTAssertEqual(adjustedFrame.size, anchoredFrame.size)
    }

    func testMenuBarClearanceLeavesAlreadyOffsetWindowUntouched() {
        let screenFrames = [CGRect(x: 0, y: 0, width: 1440, height: 900)]
        let frame = CGRect(x: 300, y: 280, width: 760, height: 560)

        let adjustedFrame = BrainBarWindowPlacement.clearingMenuBar(
            frame: frame,
            screenFrames: screenFrames,
            topGap: 18
        )

        XCTAssertEqual(adjustedFrame, frame)
    }

    func testCoordinatorPersistsLatestFrameAcrossReattach() {
        let store = FakeKeyValueStore()
        let frameStore = BrainBarWindowFrameStore(defaults: store)
        let coordinator = BrainBarWindowCoordinator(
            frameStore: frameStore,
            screenFramesProvider: { [CGRect(x: 0, y: 0, width: 1440, height: 900)] }
        )
        let firstWindow = FakeWindowHandle(frame: CGRect(x: 20, y: 20, width: 600, height: 480))

        coordinator.attach(window: firstWindow)
        firstWindow.frame = CGRect(x: 140, y: 100, width: 840, height: 620)
        coordinator.captureCurrentFrame()

        let secondWindow = FakeWindowHandle(frame: .zero)
        coordinator.attach(window: secondWindow)

        XCTAssertEqual(secondWindow.frame, CGRect(x: 140, y: 100, width: 840, height: 620))
    }

    func testCoordinatorTogglesAttachedWindowVisibility() {
        let coordinator = BrainBarWindowCoordinator(
            frameStore: BrainBarWindowFrameStore(defaults: FakeKeyValueStore()),
            screenFramesProvider: { [CGRect(x: 0, y: 0, width: 1440, height: 900)] }
        )
        let window = FakeWindowHandle(frame: CGRect(x: 0, y: 0, width: 700, height: 520))
        coordinator.attach(window: window)

        XCTAssertFalse(window.isVisible)
        XCTAssertTrue(coordinator.toggleVisibility())
        XCTAssertTrue(window.isVisible)
        XCTAssertTrue(coordinator.toggleVisibility())
        XCTAssertFalse(window.isVisible)
    }

    func testAnchoredFramePlacesWindowBelowMenuBarIconUsingFullScreenCoordinates() {
        let fullScreenFrames = [CGRect(x: 0, y: 0, width: 1440, height: 900)]
        let visibleScreenFrames = [CGRect(x: 0, y: 0, width: 1440, height: 876)]
        let currentAppKitFrame = CGRect(x: 580, y: 200, width: 760, height: 560)
        let menuBarItemAppKitFrame = CGRect(x: 980, y: 876, width: 24, height: 24)

        let currentAccessibilityFrame = BrainBarWindowPlacement.accessibilityFrame(
            fromAppKit: currentAppKitFrame,
            screenFrames: fullScreenFrames
        )
        let menuBarItemAccessibilityFrame = BrainBarWindowPlacement.accessibilityFrame(
            fromAppKit: menuBarItemAppKitFrame,
            screenFrames: fullScreenFrames
        )

        let anchoredFrame = currentAccessibilityFrame.flatMap { currentAccessibilityFrame in
            menuBarItemAccessibilityFrame.flatMap { menuBarItemAccessibilityFrame in
                BrainBarWindowPlacement.anchoredFrameBelowMenuBarItem(
                    currentAccessibilityFrame: currentAccessibilityFrame,
                    menuBarItemAccessibilityFrame: menuBarItemAccessibilityFrame,
                    screenFrames: fullScreenFrames,
                    visibleScreenFrames: visibleScreenFrames,
                    gap: 4
                )
            }
        }

        XCTAssertEqual(
            anchoredFrame,
            CGRect(x: 580, y: 312, width: 760, height: 560)
        )
    }

    func testPreferredMenuBarItemFrameUsesMouseDisplayInsteadOfFirstMatch() {
        let candidates = [
            CGRect(x: 980, y: 876, width: 24, height: 24),
            CGRect(x: 2492, y: 1411, width: 24, height: 24),
        ]
        let mouseLocation = CGPoint(x: 2504, y: 1416)

        let preferred = BrainBarWindowPlacement.preferredMenuBarItemFrame(
            candidates: candidates,
            mouseLocation: mouseLocation
        )

        XCTAssertEqual(preferred, candidates[1])
    }

    @MainActor
    func testRuntimeTracksRequestedInlineQuickAction() {
        let runtime = BrainBarRuntime()

        runtime.presentQuickAction(.search)
        XCTAssertEqual(runtime.requestedQuickAction, .search)

        runtime.presentQuickAction(.capture)
        XCTAssertEqual(runtime.requestedQuickAction, .capture)

        runtime.clearQuickActionRequest()
        XCTAssertNil(runtime.requestedQuickAction)
    }

    @MainActor
    func testShowSearchPanelRoutesThroughSearchRequestedCallback() {
        let runtime = BrainBarRuntime()
        var searchRequests = 0
        runtime.onSearchRequested = { searchRequests += 1 }

        runtime.showSearchPanel()
        runtime.showSearchPanel()

        XCTAssertEqual(
            searchRequests,
            2,
            "Runtime.showSearchPanel must fire onSearchRequested so AppDelegate can route to the integrated command bar (menuBarWindow mode) or the legacy floating panel (legacyStatusItem mode)."
        )
    }

    @MainActor
    func testShowQuickCapturePanelRoutesThroughQuickCaptureRequestedCallback() {
        let runtime = BrainBarRuntime()
        var captureRequests = 0
        runtime.onQuickCaptureRequested = { captureRequests += 1 }

        runtime.showQuickCapturePanel()

        XCTAssertEqual(
            captureRequests,
            1,
            "Runtime.showQuickCapturePanel must fire onQuickCaptureRequested so AppDelegate can route to the integrated command bar (menuBarWindow mode) or the legacy floating panel (legacyStatusItem mode)."
        )
    }
}

private final class FakeKeyValueStore: BrainBarKeyValueStoring {
    private var storage: [String: String] = [:]

    func string(forKey defaultName: String) -> String? {
        storage[defaultName]
    }

    func setString(_ value: String?, forKey defaultName: String) {
        storage[defaultName] = value
    }
}

@MainActor
private final class FakeWindowHandle: BrainBarWindowHandling {
    var frame: CGRect
    var isVisible = false

    init(frame: CGRect) {
        self.frame = frame
    }

    func apply(frame: CGRect) {
        self.frame = frame
    }

    func show() {
        isVisible = true
    }

    func hide() {
        isVisible = false
    }
}
