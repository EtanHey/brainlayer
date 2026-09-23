import XCTest
@testable import BrainBar

@MainActor
final class DegradationStateTypeTests: XCTestCase {
    deinit {}

    func testHealthyIsNotDegraded() {
        let state: DegradationState = .healthy
        XCTAssertFalse(state.isDegraded)
        XCTAssertNil(state.reason)
    }

    func testDegradedExposesReason() {
        let state: DegradationState = .degraded(reason: "ReadOnly")
        XCTAssertTrue(state.isDegraded)
        XCTAssertEqual(state.reason, "ReadOnly")
    }

    func testDegradationStateEquatable() {
        XCTAssertEqual(DegradationState.healthy, .healthy)
        XCTAssertEqual(DegradationState.degraded(reason: "x"), .degraded(reason: "x"))
        XCTAssertNotEqual(DegradationState.healthy, .degraded(reason: "x"))
        XCTAssertNotEqual(
            DegradationState.degraded(reason: "x"),
            DegradationState.degraded(reason: "y")
        )
    }
}
