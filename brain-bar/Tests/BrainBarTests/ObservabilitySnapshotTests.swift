import AppKit
import Foundation
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class ObservabilitySnapshotTests: XCTestCase {
    func testLiveShapedBackupFieldsDecode() throws {
        let result = ObservabilityReader.read(
            url: Bundle.module.url(
                forResource: "observability-main-58849a70",
                withExtension: "json",
                subdirectory: "Fixtures"
            )!
        )
        guard case let .readable(document) = result else { return XCTFail("Expected live-shaped fixture to decode") }

        XCTAssertNil(document.backups.lastVerifiedUpload)
        XCTAssertNotNil(document.backups.dbSnapshot)
        XCTAssertEqual(document.backups.dbSnapshot?.destination, "2026-09-13.db.gz")
        XCTAssertEqual(document.backups.launchd?.bootstrapped, false)
        XCTAssertEqual(document.backups.thresholdHours, 36)
    }

    func testMeasuredCardsLabelEveryCountForHumans() throws {
        let document = try readableDocument(named: "healthy-dev")
        let cards = ObservabilityPresentation.snapshot(
            document: document,
            now: document.generatedAt,
            cadence: .known(300),
            locale: Locale(identifier: "en_US")
        ).cards

        XCTAssertEqual(cards.map(\.title), ["Chunks", "Stores", "Emitters", "Backups"])
        XCTAssertTrue(cards[0].detail.contains("chunks indexed"))
        XCTAssertTrue(cards[0].detail.contains("in the last 24 h"))
        XCTAssertTrue(cards[0].detail.contains("everything BrainLayer has read, all sources"))
        XCTAssertTrue(cards[0].detail.contains("chunks not yet attributed to a person or source class"))
        XCTAssertEqual(
            cards[1].detail,
            "2 MCP brain_store writes in the last 24 h — what agents wrote via brain_store"
        )
        XCTAssertFalse(cards[1].detail.contains("27 MCP"), "Stores must never reuse stores.total_chunks.")
        for meaning in [
            "CLI agents", "MCP brain_store", "subagents",
            "desktop apps hidden from search", "fleet coordination", "unclassified",
        ] {
            XCTAssertTrue(cards[2].detail.contains(meaning), cards[2].detail)
        }
        XCTAssertTrue(cards.allSatisfy { !($0.subtitle ?? "").isEmpty })

        let labeledInteger = try NSRegularExpression(
            pattern: #"\d[\d,.]*\s+(?:chunks?|MCP|CLI|subagent|desktop|fleet|unclassified|h\b|days?\b|archives?\b|verified|%|in the last)"#
        )
        for card in cards {
            let dateOrIdentifier = try NSRegularExpression(
                pattern: #"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4} at \d{1,2}:\d{2}|[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*"#
            )
            let originalRange = NSRange(card.detail.startIndex..., in: card.detail)
            let countText = dateOrIdentifier.stringByReplacingMatches(
                in: card.detail, range: originalRange, withTemplate: ""
            )
            let range = NSRange(countText.startIndex..., in: countText)
            let withoutLabeledIntegers = labeledInteger.stringByReplacingMatches(
                in: countText, range: range, withTemplate: ""
            )
            XCTAssertNil(
                withoutLabeledIntegers.rangeOfCharacter(from: .decimalDigits),
                card.detail
            )
        }
    }

    func testLiveShapedCountsUseThousandsSeparators() throws {
        let document = try liveShapedDocument()
        let cards = ObservabilityPresentation.snapshot(
            document: document,
            now: document.generatedAt,
            cadence: .known(300),
            locale: Locale(identifier: "en_US")
        ).cards

        XCTAssertTrue(cards[0].detail.contains("797,727 chunks indexed"), cards[0].detail)
        XCTAssertTrue(cards[2].detail.contains("587,430 chunks from CLI agents"), cards[2].detail)
    }

    func testBackupStatusUsesTruthfulGreenRedLogic() throws {
        let healthy = try readableDocument(named: "healthy-dev")
        let stale = try readableDocument(named: "backup-errors-dev")
        let missing = try readableDocument(named: "no-op-dev")
        let live = try liveShapedDocument()

        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: healthy.backups).upload.tone, .green)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: stale.backups).upload.tone, .green)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: stale.backups).freshness.tone, .red)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: missing.backups).upload.tone, .red)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: live.backups).job.tone, .red)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: live.backups).snapshot.tone, .green)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: live.backups).freshness.tone, .red)
    }

    func testMissingArchiveCountStaysUnknown() {
        let backups = ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [], freshness: nil,
            thresholdHours: nil, retentionInvariant: nil, survivingArchives30D: nil,
            errorType: nil, lastVerifiedUpload: nil, dbSnapshot: nil, launchd: nil
        )

        let archives = ObservabilityPresentation.backupStatus(for: backups).archives
        XCTAssertEqual(archives.text, "Verified transcript archives in the last 30 days: unknown")
        XCTAssertEqual(archives.tone, .red)
    }

    func testBackupStatusSaysWhenWhereAndWhy() throws {
        let live = try liveShapedDocument()
        let status = ObservabilityPresentation.backupStatus(for: live.backups)

        XCTAssertEqual(status.upload.text, "No verified transcript upload on record")
        XCTAssertTrue(status.snapshot.text.contains("Latest DB snapshot (verified):"))
        XCTAssertTrue(status.snapshot.text.contains("→ 2026-09-13.db.gz"))
        XCTAssertEqual(
            status.job.text,
            "Transcript backup (com.brainlayer.jsonl-backup): NOT loaded — parked in .disabled-retention-P0"
        )
        XCTAssertFalse(status.lines.map(\.text).contains { $0.hasPrefix("Backup job") })
        XCTAssertEqual(status.freshness.text, "Backup freshness (DB + transcript): stale (> 36 h)")
        XCTAssertEqual(status.retention.text, "Transcript retention invariant: PASS")
        XCTAssertEqual(status.archives.text, "0 verified transcript archives in the last 30 days")
        XCTAssertEqual(status.error?.text, "DB backup error: Google Drive credentials missing — re-auth needed")
    }

    func testRestoredDriveCredentialsRenderAsPendingInsteadOfCurrentError() {
        let backups = ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [], freshness: "stale",
            thresholdHours: 36, retentionInvariant: "PASS", survivingArchives30D: 0,
            errorType: "drive_credentials_restored_backup_pending",
            lastVerifiedUpload: nil, dbSnapshot: nil, launchd: nil
        )

        let status = ObservabilityPresentation.backupStatus(for: backups)

        XCTAssertEqual(status.error?.text, "Google Drive credentials restored — next backup pending")
        XCTAssertEqual(status.error?.tone, .neutral)
    }

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
            XCTAssertEqual(snapshot.cards.map(\.title), ["Chunks", "Stores", "Emitters", "Backups"])
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
        XCTAssertTrue(card.detail.contains("No verified transcript upload on record"))
        XCTAssertTrue(card.detail.contains("Backup freshness (DB + transcript): unknown"))
        XCTAssertTrue(card.detail.contains("Transcript retention invariant: PASS"))
        XCTAssertTrue(card.detail.contains("0 verified transcript archives in the last 30 days"))
        XCTAssertTrue(card.detail.contains("Transcript backup error: Jsonl Backup Attempt Invalid"))
        XCTAssertFalse(card.detail.contains("unmeasurable"))
    }

    func testBackupDiagnosticsAreRendered() throws {
        let document = try readableDocument(named: "backup-errors-dev")
        let card = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: .known(300))
                .cards.first { $0.title == "Backups" }
        )
        XCTAssertTrue(card.detail.contains("DB backup error: Backup input file missing"))
        XCTAssertTrue(card.detail.contains("1 verified transcript archive in the last 30 days"))
    }

    func testRenderedCardTonesAreDistinct() throws {
        let healthy = try render(document: readableDocument(named: "healthy-dev"), named: "tone-healthy")
        let neutral = try render(document: readableDocument(named: "missing-source-class-dev"), named: "tone-neutral")
        let amber = try render(document: readableDocument(named: "backup-errors-dev"), named: "tone-amber")

        let standardColor = try color(in: healthy, normalizedX: 0.22, normalizedY: 0.27)
        let neutralColor = try color(in: neutral, normalizedX: 0.22, normalizedY: 0.27)
        let amberColor = try color(in: amber, normalizedX: 0.72, normalizedY: 0.49)

        XCTAssertNotEqual(standardColor, neutralColor)
        XCTAssertNotEqual(standardColor, amberColor)
        XCTAssertNotEqual(neutralColor, amberColor)
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

    func testDashboardAndSettingsUseTheSameRuntimeDatabasePath() {
        let environment = [
            "BRAINLAYER_DB": "/tmp/brainlayer-override/brainlayer.db",
            "BRAINLAYER_OBSERVABILITY_PATH": "",
        ]
        let databasePath = environment["BRAINLAYER_DB"]!

        XCTAssertEqual(
            BrainBarSettingsView.observabilityURL(databasePath: databasePath, environment: environment),
            ObservabilityReader.url(dbPath: databasePath, environment: environment)
        )
        XCTAssertEqual(
            BrainBarSettingsView.observabilityURL(databasePath: databasePath, environment: environment).path,
            "/tmp/brainlayer-override/observability.json"
        )
    }

    func testMissingHealthCheckCadenceIsDisclosedOnCard() throws {
        let cadence = ObservabilityReader.healthCheckCadence(environment: [
            "BRAINLAYER_HEALTH_CHECK_PLIST_PATH": "/tmp/brainbar-missing-health-check-\(UUID().uuidString).plist",
        ])
        XCTAssertEqual(cadence.interval, 300)
        XCTAssertEqual(cadence.assumption, "cadence unknown, assuming 300s")

        let document = try readableDocument(named: "healthy-dev")
        let chunks = try XCTUnwrap(
            ObservabilityPresentation.snapshot(document: document, now: document.generatedAt, cadence: cadence)
                .cards.first { $0.title == "Chunks" }
        )
        XCTAssertEqual(chunks.note, "cadence unknown, assuming 300s")
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

    func testCancelledLiveWatcherFinishesWithoutApplyingSupersededResult() async {
        let finished = expectation(description: "cancelled watcher stream finishes")
        var applied: [String] = []
        let url = URL(fileURLWithPath: "/tmp/unused")
        let watcher = Task {
            for await result in ObservabilityLiveView.Reader.watch(
                url: url,
                every: .seconds(1),
                using: { _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                    return .unreadable("old")
                }
            ) {
                if case let .unreadable(value) = result {
                    applied.append(value)
                }
            }
            finished.fulfill()
        }

        await fulfillment(of: [finished], timeout: 1)
        watcher.cancel()
        await watcher.value
        XCTAssertTrue(applied.isEmpty)
    }

    func testCancellingLiveWatcherCancelsProducerWithoutApplyingSupersededResult() async {
        let operationStarted = expectation(description: "watch operation starts")
        let producerCancelled = expectation(description: "watch producer is cancelled")
        var applied: [String] = []
        let url = URL(fileURLWithPath: "/tmp/unused")
        let watcher = Task {
            for await result in ObservabilityLiveView.Reader.watch(
                url: url,
                every: .seconds(1),
                using: { _ in
                    operationStarted.fulfill()
                    do {
                        try await Task.sleep(for: .seconds(10))
                    } catch {
                        XCTAssertTrue(Task.isCancelled)
                        producerCancelled.fulfill()
                    }
                    return .unreadable("old")
                }
            ) {
                if case let .unreadable(value) = result {
                    applied.append(value)
                }
            }
        }

        await fulfillment(of: [operationStarted], timeout: 1)
        watcher.cancel()
        await fulfillment(of: [producerCancelled], timeout: 1)
        await watcher.value
        XCTAssertTrue(applied.isEmpty)
    }

    func testLiveReaderReloadsRewrittenFileAndSurfacesUnreadableReplacement() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("observability-live-reload-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }

        let fixtureURL = fixtureRoot.appendingPathComponent("golden/healthy-dev.json")
        let originalData = try Data(contentsOf: fixtureURL)
        try originalData.write(to: url, options: .atomic)

        var rewrittenObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: originalData) as? [String: Any]
        )
        rewrittenObject["generated_at"] = "2026-09-14T18:11:14Z"
        let rewrittenData = try JSONSerialization.data(withJSONObject: rewrittenObject, options: [.sortedKeys])

        let observed = expectation(description: "initial, rewritten, and unreadable snapshots applied")
        observed.expectedFulfillmentCount = 3
        var results: [ObservabilityReadResult] = []
        let watch = Task {
            for await result in ObservabilityLiveView.Reader.watch(url: url, every: .milliseconds(10)) {
                results.append(result)
                switch results.count {
                case 1:
                    try? rewrittenData.write(to: url, options: .atomic)
                case 2:
                    try? Data("{not-json".utf8).write(to: url, options: .atomic)
                default:
                    break
                }
                observed.fulfill()
            }
        }

        await fulfillment(of: [observed], timeout: 1)
        watch.cancel()
        await watch.value

        guard results.count >= 3 else {
            return XCTFail("Expected three results, received \(results.count)")
        }
        guard case let .readable(first) = results[0],
              case let .readable(second) = results[1],
              case let .unreadable(reason) = results[2] else {
            return XCTFail("Expected readable → rewritten readable → explicit unreadable results")
        }
        XCTAssertNotEqual(first.generatedAt, second.generatedAt)
        XCTAssertEqual(second.generatedAt, ISO8601DateFormatter().date(from: "2026-09-14T18:11:14Z"))
        XCTAssertTrue(reason.contains("Observability data unreadable"))
    }

    private func readableDocument(named name: String) throws -> ObservabilityDocument {
        let result = ObservabilityReader.read(url: fixtureRoot.appendingPathComponent("golden/\(name).json"))
        guard case let .readable(document) = result else {
            XCTFail("Expected readable fixture: \(name)")
            throw FixtureError.unreadable(name)
        }
        return document
    }

    private func liveShapedDocument() throws -> ObservabilityDocument {
        let result = ObservabilityReader.read(
            url: Bundle.module.url(
                forResource: "observability-main-58849a70",
                withExtension: "json",
                subdirectory: "Fixtures"
            )!
        )
        guard case let .readable(document) = result else {
            XCTFail("Expected live-shaped fixture")
            throw FixtureError.unreadable("live-shaped")
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
        case "stores", "author_unknown": "Chunks"
        case "emitters": "Emitters"
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
