import XCTest
@testable import BrainBar

final class SearchFiltersTests: XCTestCase {
    func testDefaultsToAllResultsWithNoNarrowing() {
        let filters = SearchFilters()

        XCTAssertEqual(filters.primaryChip, .all)
        XCTAssertNil(filters.importanceMin)
        XCTAssertFalse(filters.unreadOnly)
    }

    func testSelectingImportantChipSetsImportanceThreshold() {
        let filters = SearchFilters().selecting(.important)

        XCTAssertEqual(filters.primaryChip, .important)
        XCTAssertEqual(filters.importanceMin, 7)
        XCTAssertFalse(filters.unreadOnly)
    }

    func testSelectingUnreadChipEnablesUnreadWithoutImportanceThreshold() {
        let filters = SearchFilters().selecting(.unread)

        XCTAssertEqual(filters.primaryChip, .unread)
        XCTAssertNil(filters.importanceMin)
        XCTAssertTrue(filters.unreadOnly)
    }

    func testSelectingAllClearsNarrowedModes() {
        let filters = SearchFilters()
            .selecting(.important)
            .selecting(.all)

        XCTAssertEqual(filters.primaryChip, .all)
        XCTAssertNil(filters.importanceMin)
        XCTAssertFalse(filters.unreadOnly)
    }

    func testFilterChipOrderIsStableForKeyboardNavigation() {
        XCTAssertEqual(SearchFilterChip.allCases, [.all, .important, .unread])
    }
}
