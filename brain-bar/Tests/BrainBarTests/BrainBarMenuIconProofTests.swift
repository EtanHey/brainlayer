import XCTest
@testable import BrainBar

@MainActor
final class BrainBarMenuIconProofTests: XCTestCase {
    func testActualStatusFactoryMatchesOnlyAgentWatcherPixelsAndPreservesBadge() throws {
        for name in BrainBarMenuIconProof.names {
            let capture = try BrainBarMenuIconProof.measure(name)
            let proof = try XCTUnwrap(capture["pixel_check"] as? [String: Any])
            XCTAssertEqual(proof["matches_reference"] as? Bool, true, name)
            XCTAssertGreaterThan(try XCTUnwrap(proof["nontransparent_pixels"] as? Int), 0, name)
            if name == "status-icon-badged" {
                XCTAssertGreaterThan(try XCTUnwrap(proof["red_pixels"] as? Int), 0)
            }
        }
    }
}
