import XCTest
@testable import BrainBar

/// Guards the Vector "see under the hood" popover positioning.
///
/// Regression (Etan live-QA 2026-06-20): the popover anchored at
/// `anchor.maxY + gap` with no clamp, so when the signal panel sat low in the
/// window the popover overflowed the bottom edge and was CLIPPED. The fix clamps
/// the Y so the full popover height stays within the container — it pops OVER the
/// content above rather than clipping below.
final class BrainBarWindowRootViewVectorDetailLayoutTests: XCTestCase {
    func testSitsBelowAnchorWhenThereIsRoom() {
        let y = BrainBarVectorDetailLayout.yOffset(
            anchorMaxY: 100, gap: 10, detailHeight: 50, containerHeight: 400, padding: 16
        )
        XCTAssertEqual(y, 110, accuracy: 0.001, "should sit just below the anchor when it fits")
    }

    func testClampsUpwardSoBottomStaysInBounds() {
        // anchor low in the window: below-offset would clip.
        let container: CGFloat = 400
        let detail: CGFloat = 120
        let padding: CGFloat = 16
        let y = BrainBarVectorDetailLayout.yOffset(
            anchorMaxY: 380, gap: 10, detailHeight: detail, containerHeight: container, padding: padding
        )
        // clamped so the bottom (y + detail) stays within container - padding.
        XCTAssertLessThanOrEqual(y + detail, container - padding + 0.001)
        XCTAssertEqual(y, container - padding - detail, accuracy: 0.001)
    }

    func testTopAlignsWhenTallerThanContainer() {
        let y = BrainBarVectorDetailLayout.yOffset(
            anchorMaxY: 380, gap: 10, detailHeight: 500, containerHeight: 400, padding: 16
        )
        XCTAssertEqual(y, 16, accuracy: 0.001, "falls back to padding when it cannot fully fit")
    }

    func testUnmeasuredHeightUsesPreferredBelowOffset() {
        // Before the popover height is measured (height == 0), use the preferred offset.
        let y = BrainBarVectorDetailLayout.yOffset(
            anchorMaxY: 200, gap: 8, detailHeight: 0, containerHeight: 400, padding: 16
        )
        XCTAssertEqual(y, 208, accuracy: 0.001)
    }
}
