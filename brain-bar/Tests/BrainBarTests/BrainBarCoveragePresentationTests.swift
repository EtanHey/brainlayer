import XCTest

@testable import BrainBar

final class BrainBarCoveragePresentationTests: XCTestCase {
    func testCoverageTruthTable() {
        let cases: [(String, Int, Int, Bool, String?, String, String, Int?)] = [
            ("complete", 1_000, 1_000, true, nil, "100%", "1,000 / 1,000", 0),
            ("incomplete", 99, 100, true, nil, "99%", "99 / 100", 1),
            ("ordinary rounding", 807, 1_000, true, nil, "81%", "807 / 1,000", 193),
            ("tiny fraction", 1, 1_000, true, nil, "<1%", "1 / 1,000", 999),
            ("rounding edge", 999, 1_000, true, nil, "99%", "999 / 1,000", 1),
            ("zero eligible", 0, 0, true, nil, "No eligible", "0 eligible chunks", nil),
            ("loading", 0, 0, false, nil, "Computing…", "Counting eligible chunks…", nil),
            ("failed first load", 0, 0, false, "query failed", "Unavailable", "Counts unavailable: query failed", nil),
            ("denominator mismatch", 101, 100, true, nil, "Unavailable", "Counts disagree", nil),
        ]

        for (name, indexed, eligible, available, error, percent, counts, missing) in cases {
            let presentation = BrainBarCoveragePresentation(
                indexedCount: indexed, eligibleCount: eligible, isAvailable: available,
                lastError: error, locale: Locale(identifier: "en_US")
            )
            XCTAssertEqual(presentation.percentText, percent, name)
            XCTAssertEqual(presentation.countText, counts, name)
            XCTAssertEqual(presentation.missingCount, missing, name)
            XCTAssertEqual(presentation.isMeasurable, available && eligible > 0 && indexed <= eligible, name)
        }
    }

    func testAccessibleLabelsExplainTheSameDenominatorForEverySignal() {
        for name in ["Vector", "FTS5", "Trigram"] {
            let presentation = BrainBarCoveragePresentation(
                indexedCount: 999, eligibleCount: 1_000, isAvailable: true,
                locale: Locale(identifier: "en_US")
            )
            XCTAssertEqual(
                presentation.accessibilityLabel(for: name),
                "\(name): 99%; 999 of 1,000 eligible chunks indexed; 1 not indexed"
            )
        }
    }

    func testFirstCoverageLoadDoesNotClaimFailure() {
        let presentation = BrainBarCoveragePresentation(
            indexedCount: 0, eligibleCount: 0, isAvailable: false
        )
        XCTAssertEqual(presentation.percentText, "Computing…")
        XCTAssertEqual(presentation.countText, "Counting eligible chunks…")
        XCTAssertNil(presentation.missingCount)
    }

    func testLoadedRefreshAndUserLocale() {
        let presentation = BrainBarCoveragePresentation(
            indexedCount: 1_001, eligibleCount: 2_000, isAvailable: true,
            isRefreshing: true, locale: Locale(identifier: "de_DE")
        )
        XCTAssertEqual(presentation.percentText, "50%")
        XCTAssertEqual(presentation.countText, "1.001 / 2.000")
        XCTAssertEqual(presentation.missingText, "999")
    }
}
