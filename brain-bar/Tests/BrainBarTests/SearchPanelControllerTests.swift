import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class SearchPanelControllerTests: XCTestCase {
    func testPanelUsesNonactivatingStyleAndAcceptsKeyboardFocus() {
        let controller = SearchPanelController(
            viewModel: SearchViewModel(
                queryActor: SearchQueryActor(
                    lexicalSearch: { _, _, _ in [] },
                    rerank: { candidates, _ in candidates }
                )
            )
        )

        XCTAssertTrue(controller.panelForTesting.styleMask.contains(.nonactivatingPanel))
        XCTAssertTrue(controller.panelForTesting.canBecomeKey)
        XCTAssertFalse(controller.panelForTesting.canBecomeMain)
    }

    func testShowAndDismissTogglePanelVisibility() {
        let controller = SearchPanelController(
            viewModel: SearchViewModel(
                queryActor: SearchQueryActor(
                    lexicalSearch: { _, _, _ in [] },
                    rerank: { candidates, _ in candidates }
                )
            )
        )

        controller.show()
        XCTAssertTrue(controller.panelForTesting.isVisible)

        controller.dismiss()
        XCTAssertFalse(controller.panelForTesting.isVisible)
    }
}
