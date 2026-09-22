import XCTest

@testable import BrainBar

final class BrainBarCoveragePresentationTests: XCTestCase {
    func testCoverageTruthTable() {
        let cases: [(String, Int, Int, Bool, String, String, Int?)] = [
            ("complete", 1_000, 1_000, true, "100%", "1,000 / 1,000", 0),
            ("incomplete", 99, 100, true, "99%", "99 / 100", 1),
            ("ordinary rounding", 807, 1_000, true, "81%", "807 / 1,000", 193),
            ("rounding edge", 999, 1_000, true, "99%", "999 / 1,000", 1),
            ("zero eligible", 0, 0, true, "No eligible", "0 eligible chunks", nil),
            ("unavailable", 0, 0, false, "Unavailable", "Counts unavailable", nil),
            ("denominator mismatch", 101, 100, true, "Unavailable", "Counts disagree", nil),
        ]

        for (name, indexed, eligible, available, percent, counts, missing) in cases {
            let presentation = BrainBarCoveragePresentation(
                indexedCount: indexed, eligibleCount: eligible, isAvailable: available
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
                indexedCount: 999, eligibleCount: 1_000, isAvailable: true
            )
            XCTAssertEqual(
                presentation.accessibilityLabel(for: name),
                "\(name): 99%; 999 of 1,000 eligible chunks indexed; 1 not indexed"
            )
        }
    }
}
