import SwiftUI
import XCTest
@testable import BrainBar

// QA #16: the entity card ("contentGolem") mixed a square "8" badge with a
// circular pill type chip — "the badges are just horrible, some are circle".
// The fix standardizes every chip onto one token: single shape (capsule),
// uniform height, semantic color only. These tests lock that token so the
// square/circle mix can't regress.
final class KGBadgeConsistencyTests: XCTestCase {
    func testAllChipsShareUniformHeightTokens() {
        // One font size + one vertical padding => one rendered height for every
        // chip, regardless of tint or text.
        XCTAssertEqual(KGSidebarView.KGBadgeStyle.fontSize, 10)
        XCTAssertEqual(KGSidebarView.KGBadgeStyle.verticalPadding, 5)
        XCTAssertEqual(KGSidebarView.KGBadgeStyle.horizontalPadding, 9)
    }

    func testChipTintsMapToSemanticColors() {
        XCTAssertEqual(KGSidebarView.ChipTint.primary.color, Color.primary)
        XCTAssertEqual(KGSidebarView.ChipTint.blue.color, Color.blue)
        XCTAssertEqual(KGSidebarView.ChipTint.green.color, Color.green)
        XCTAssertEqual(KGSidebarView.ChipTint.amber.color, Color.orange)
    }

    func testSaturatedTintsShareOneFillOpacity() {
        // Color carries the meaning; the saturated chips must read at one weight.
        let saturated: [KGSidebarView.ChipTint] = [.blue, .green, .amber]
        let opacities = Set(saturated.map(\.fillOpacity))
        XCTAssertEqual(opacities.count, 1, "All semantic-colored chips must share one fill opacity")
        XCTAssertEqual(opacities.first, 0.14)
    }

    func testNeutralChipIsLightenedForEqualPerceivedWeight() {
        // Neutral (black/white) reads heavier than a tint at equal alpha, so it
        // is intentionally lighter — but it is the only exception, and documented.
        XCTAssertLessThan(
            KGSidebarView.ChipTint.primary.fillOpacity,
            KGSidebarView.ChipTint.blue.fillOpacity
        )
    }
}
