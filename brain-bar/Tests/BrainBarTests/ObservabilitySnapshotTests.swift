import AppKit
import Foundation
import SwiftUI
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
            try render(document: document, named: fixture.id)
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
        XCTAssertEqual(card.detail, "unmeasurable — required column missing: chunks.source_class")
        XCTAssertFalse(card.detail.contains("0"))
    }

    func testMeasuredUnknownBackupIsNeutralWithoutClaimingUnmeasurable() throws {
        let result = ObservabilityReader.read(
            url: fixtureRoot.appendingPathComponent("golden/legacy-no-op-dev.json")
        )
        guard case let .readable(document) = result else { return XCTFail("Expected readable fixture") }
        let card = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: 300)
                .cards.first { $0.title == "Backups" }
        )

        XCTAssertEqual(card.tone, .neutral)
        XCTAssertEqual(card.detail, "unknown · retention PASS")
        XCTAssertFalse(card.detail.contains("unmeasurable"))
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

    private func render(document: ObservabilityDocument, named name: String) throws {
        let view = NSHostingView(rootView: ObservabilityDashboardView(
            result: .readable(document), now: document.generatedAt, cadence: 300
        ).environment(\.colorScheme, .dark))
        view.frame = NSRect(x: 0, y: 0, width: 760, height: 560)
        view.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(png.count, 1_000, name)
        if let directory = ProcessInfo.processInfo.environment["BRAINBAR_OBSERVABILITY_RENDER_DIR"] {
            let url = URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent("\(name).png")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try png.write(to: url)
            print("[observability-render] wrote \(url.path) (\(png.count) bytes)")
        }
    }
}

private struct FixtureManifest: Decodable {
    let cases: [FixtureCase]
}

private struct FixtureCase: Decodable {
    let id: String
    let split: String
    let golden: String
    enum CodingKeys: String, CodingKey { case id = "case_id", split, golden }
}
