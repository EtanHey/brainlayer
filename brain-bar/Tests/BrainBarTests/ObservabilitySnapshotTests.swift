import Foundation
import XCTest

@MainActor
final class ObservabilitySnapshotTests: XCTestCase {
    func testDevGoldensAwaitThePhase2cRenderer() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixtureRoot = repositoryRoot.appendingPathComponent("tests/fixtures/observability")
        let casesURL = fixtureRoot.appendingPathComponent("cases.json")
        let data = try Data(contentsOf: casesURL)
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let cases = try XCTUnwrap(document["cases"] as? [[String: Any]])
        let devCases = cases.filter { $0["split"] as? String == "dev" }

        XCTAssertFalse(devCases.isEmpty, "Phase 0 must commit dev observability goldens.")
        for fixture in devCases {
            let golden = try XCTUnwrap(fixture["golden"] as? String)
            XCTAssertTrue(FileManager.default.fileExists(atPath: fixtureRoot.appendingPathComponent(golden).path))
        }

        XCTFail("RED until Phase 2c renders every dev golden through the real Observability view.")
    }
}
