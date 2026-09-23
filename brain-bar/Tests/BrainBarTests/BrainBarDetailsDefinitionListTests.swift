import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class BrainBarDetailsDefinitionListTests: XCTestCase {
    @MainActor
    func testLongDetailValueGetsEnoughHeightToRemainVisible() {
        let short = height(for: "23")
        let long = height(for: String(repeating: "Agent writes unavailable: Database path unavailable. ", count: 3))

        XCTAssertGreaterThan(long, short + 15)
    }

    @MainActor
    private func height(for value: String) -> CGFloat {
        let host = NSHostingView(rootView: BrainBarDefinitionList(
            title: "Activity",
            rows: [("Agent writes (24 h)", value)]
        ).frame(width: 360))
        host.layoutSubtreeIfNeeded()
        return host.fittingSize.height
    }
}
