import XCTest
@testable import BrainBarLifecycle

final class ChunkDedupeTests: XCTestCase {
    func testComputeDedupeFieldsPopulatesBands() {
        let fields = ChunkDedupe.computeDedupeFields(content: "Repair-d dedupe payload", createdAt: "2026-08-18T00:00:00Z")
        XCTAssertFalse(fields.dedupeHash.isEmpty)
        XCTAssertEqual(fields.simhash.count, 16)
        XCTAssertEqual(fields.bands.0.count, 4)
    }
}
