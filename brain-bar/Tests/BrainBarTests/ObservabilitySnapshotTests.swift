import Foundation
import XCTest
@testable import BrainBar

@MainActor
final class ObservabilitySnapshotTests: XCTestCase {
    private var fixtureRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures/observability")
    }

    func testEveryDevGoldenRendersFourCards() throws {
        let manifest = try JSONDecoder().decode(
            FixtureManifest.self,
            from: Data(contentsOf: fixtureRoot.appendingPathComponent("cases.json"))
        )
        let devCases = manifest.cases.filter { $0.split == "dev" }
        XCTAssertFalse(devCases.isEmpty)

        for fixture in devCases {
            let result = ObservabilityReader.read(
                url: fixtureRoot.appendingPathComponent(fixture.golden)
            )
            guard case let .readable(document) = result else {
                return XCTFail("Expected readable golden: \(fixture.id)")
            }
            let snapshot = ObservabilityPresentation.snapshot(
                document: document,
                now: document.generatedAt,
                cadence: 300
            )
            XCTAssertEqual(snapshot.cards.map(\.title), ["Stores", "Emitters", "Author-unknown", "Backups"])
            XCTAssertEqual(snapshot.cards.count, 4, fixture.id)
            XCTAssertTrue(snapshot.cards.allSatisfy { !$0.detail.isEmpty }, fixture.id)
        }
    }

    func testUnmeasurableCardShowsReasonWithoutNumericFallback() throws {
        let result = ObservabilityReader.read(
            url: fixtureRoot.appendingPathComponent("golden/missing-source-class-dev.json")
        )
        guard case let .readable(document) = result else { return XCTFail("Expected readable fixture") }
        let card = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: 300)
                .cards.first { $0.title == "Stores" }
        )

        XCTAssertEqual(card.tone, .neutral)
        XCTAssertEqual(card.detail, "required column missing: chunks.source_class")
        XCTAssertFalse(card.detail.contains("0"))
    }

    func testSchemaVersionMismatchIsUnreadableWithReason() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("observability-schema-mismatch-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{\"schema_version\":2}".utf8).write(to: url)

        guard case let .unreadable(reason) = ObservabilityReader.read(url: url) else {
            return XCTFail("Expected schema mismatch to be unreadable")
        }
        XCTAssertTrue(reason.contains("schema_version 2"))
    }

    func testPathIsBesideDatabaseUnlessOverridden() {
        XCTAssertEqual(
            ObservabilityReader.url(dbPath: "/tmp/brainlayer/brainlayer.db", environment: [:]).path,
            "/tmp/brainlayer/observability.json"
        )
        XCTAssertEqual(
            ObservabilityReader.url(
                dbPath: "/tmp/brainlayer/brainlayer.db",
                environment: ["BRAINLAYER_OBSERVABILITY_PATH": "/tmp/custom.json"]
            ).path,
            "/tmp/custom.json"
        )
    }

    func testStaleDocumentRendersAmberWithAge() throws {
        let result = ObservabilityReader.read(
            url: fixtureRoot.appendingPathComponent("golden/healthy-dev.json")
        )
        guard case let .readable(document) = result else { return XCTFail("Expected readable fixture") }
        let snapshot = ObservabilityPresentation.snapshot(
            document: document,
            now: document.generatedAt.addingTimeInterval(601),
            cadence: 300
        )
        XCTAssertTrue(snapshot.isStale)
        XCTAssertTrue(snapshot.ageText.contains("old"))
        XCTAssertTrue(snapshot.cards.allSatisfy { $0.tone == .amber })
    }
}

private struct FixtureManifest: Decodable {
    let cases: [FixtureCase]
}

private struct FixtureCase: Decodable {
    let id: String
    let split: String
    let golden: String
}
