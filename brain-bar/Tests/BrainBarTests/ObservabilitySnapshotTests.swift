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
                cadence: .known(300)
            )
            XCTAssertEqual(snapshot.cards.map(\.title), ["Stores", "Emitters", "Author-unknown", "Backups"])
            XCTAssertEqual(snapshot.cards.count, 4, fixture.id)
            XCTAssertTrue(snapshot.cards.allSatisfy { !$0.detail.isEmpty }, fixture.id)
            for section in fixture.unmeasurableSections {
                let card = try XCTUnwrap(snapshot.cards.first { $0.title == title(for: section) }, "\(fixture.id): \(section)")
                XCTAssertEqual(card.tone, .neutral, "\(fixture.id): \(section)")
                XCTAssertEqual(card.detail, "unmeasurable — \(reason(for: section, in: document))", "\(fixture.id): \(section)")
                let fallback = card.detail.replacingOccurrences(of: reason(for: section, in: document), with: "")
                XCTAssertNil(fallback.rangeOfCharacter(from: .decimalDigits), "\(fixture.id): \(section)")
            }
            _ = try render(document: document, named: fixture.id)
        }
    }

    func testUnmeasurableCardShowsReasonWithoutNumericFallback() throws {
        let result = ObservabilityReader.read(
            url: fixtureRoot.appendingPathComponent("golden/missing-source-class-dev.json")
        )
        guard case let .readable(document) = result else { return XCTFail("Expected readable fixture") }
        let card = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: .known(300))
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
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: .known(300))
                .cards.first { $0.title == "Backups" }
        )

        XCTAssertEqual(card.tone, .neutral)
        XCTAssertEqual(card.detail, "unknown · retention PASS · 0 archives in 30d · jsonl_backup_attempt_invalid")
        XCTAssertFalse(card.detail.contains("unmeasurable"))
    }

    func testBackupDiagnosticsAreRendered() throws {
        let document = try readableDocument(named: "backup-errors-dev")
        let card = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: .known(300))
                .cards.first { $0.title == "Backups" }
        )
        XCTAssertTrue(card.detail.contains("FileNotFoundError"))
        XCTAssertTrue(card.detail.contains("1 archive in 30d"))
    }

    func testRenderedTonesAndOpaqueBackgroundAreDistinct() throws {
        let healthy = try render(document: readableDocument(named: "healthy-dev"), named: "tone-healthy")
        let neutral = try render(document: readableDocument(named: "missing-source-class-dev"), named: "tone-neutral")
        let amber = try render(document: readableDocument(named: "backup-errors-dev"), named: "tone-amber")

        let standardColor = try color(in: healthy, normalizedX: 0.22, normalizedY: 0.27)
        let neutralColor = try color(in: neutral, normalizedX: 0.22, normalizedY: 0.27)
        let amberColor = try color(in: amber, normalizedX: 0.72, normalizedY: 0.49)
        let backgroundColor = try XCTUnwrap(healthy.colorAt(x: 5, y: 5))

        XCTAssertNotEqual(standardColor, neutralColor)
        XCTAssertNotEqual(standardColor, amberColor)
        XCTAssertNotEqual(neutralColor, amberColor)
        XCTAssertNotEqual(standardColor, backgroundColor)
        XCTAssertNotEqual(neutralColor, backgroundColor)
        XCTAssertNotEqual(amberColor, backgroundColor)
        XCTAssertEqual(backgroundColor.alphaComponent, 1, accuracy: 0.001)
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

    func testMissingHealthCheckCadenceIsDisclosedOnCard() throws {
        let cadence = ObservabilityReader.healthCheckCadence(environment: [
            "BRAINLAYER_HEALTH_CHECK_PLIST_PATH": "/tmp/brainbar-missing-health-check-\(UUID().uuidString).plist",
        ])
        XCTAssertEqual(cadence.interval, 300)
        XCTAssertEqual(cadence.assumption, "cadence unknown, assuming 300s")

        let document = try readableDocument(named: "healthy-dev")
        let stores = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: cadence)
                .cards.first { $0.title == "Stores" }
        )
        XCTAssertEqual(stores.note, "cadence unknown, assuming 300s")
    }

    func testStaleDocumentRendersAmberWithAge() throws {
        let result = ObservabilityReader.read(
            url: fixtureRoot.appendingPathComponent("golden/healthy-dev.json")
        )
        guard case let .readable(document) = result else { return XCTFail("Expected readable fixture") }
        let snapshot = ObservabilityPresentation.snapshot(
            document: document,
            now: document.generatedAt.addingTimeInterval(601),
            cadence: .known(300)
        )
        XCTAssertTrue(snapshot.isStale)
        XCTAssertTrue(snapshot.ageText.contains("old"))
        XCTAssertTrue(snapshot.cards.allSatisfy { $0.tone == .amber })
    }

    func testSupersededLiveViewReadCancelsWorkerAndDoesNotOverwriteNewerResult() async {
        let cancellation = AsyncStream<Bool>.makeStream()
        var applied: [String] = []
        let url = URL(fileURLWithPath: "/tmp/unused")
        let old = ObservabilityLiveView.Loader.load(replacing: nil, url: url, using: { _ in
            do {
                try await Task.sleep(for: .seconds(1))
                cancellation.continuation.yield(false)
            } catch {
                cancellation.continuation.yield(Task.isCancelled)
            }
            cancellation.continuation.finish()
            return .unreadable("old")
        }, apply: { if case let .unreadable(value) = $0 { applied.append(value) } })
        await Task.yield()
        let new = ObservabilityLiveView.Loader.load(replacing: old, url: url, using: { _ in
            .unreadable("new")
        }, apply: { if case let .unreadable(value) = $0 { applied.append(value) } })

        await old.value
        await new.value
        var events = cancellation.stream.makeAsyncIterator()
        let workerWasCancelled = await events.next()
        XCTAssertEqual(workerWasCancelled, true)
        XCTAssertEqual(applied, ["new"])
    }

    private func readableDocument(named name: String) throws -> ObservabilityDocument {
        let result = ObservabilityReader.read(url: fixtureRoot.appendingPathComponent("golden/\(name).json"))
        guard case let .readable(document) = result else {
            XCTFail("Expected readable fixture: \(name)")
            throw FixtureError.unreadable(name)
        }
        return document
    }

    private func color(
        in bitmap: NSBitmapImageRep,
        normalizedX: Double,
        normalizedY: Double
    ) throws -> NSColor {
        try XCTUnwrap(bitmap.colorAt(
            x: Int(Double(bitmap.pixelsWide) * normalizedX),
            y: Int(Double(bitmap.pixelsHigh) * normalizedY)
        ))
    }

    private func title(for section: String) -> String {
        switch section {
        case "stores": "Stores"
        case "emitters": "Emitters"
        case "author_unknown": "Author-unknown"
        case "backups": "Backups"
        default: section
        }
    }

    private func reason(for section: String, in document: ObservabilityDocument) -> String {
        switch section {
        case "stores": document.stores.reason
        case "emitters": document.emitters.reason
        case "author_unknown": document.authorUnknown.reason
        case "backups": document.backups.reason
        default: ""
        }
    }

    private func render(document: ObservabilityDocument, named name: String) throws -> NSBitmapImageRep {
        let view = NSHostingView(rootView: ObservabilityDashboardView(
            result: .readable(document), now: document.generatedAt, cadence: .known(300)
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
        return bitmap
    }
}

private enum FixtureError: Error { case unreadable(String) }

private struct FixtureManifest: Decodable {
    let cases: [FixtureCase]
}

private struct FixtureCase: Decodable {
    let id: String
    let split: String
    let golden: String
    let unmeasurableSections: [String]
    enum CodingKeys: String, CodingKey {
        case id = "case_id", split, golden, unmeasurableSections = "unmeasurable_sections"
    }
}
