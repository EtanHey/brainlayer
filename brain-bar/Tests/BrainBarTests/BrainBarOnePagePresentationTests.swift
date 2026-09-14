import Foundation
import XCTest
@testable import BrainBar

final class BrainBarOnePagePresentationTests: XCTestCase {
    @MainActor
    func testHealthyPresentationNamesBackupFactsAndEveryIngestNumber() throws {
        let now = BrainBarOnePageTestFixture.now
        let result = try BrainBarOnePageTestFixture.healthyResult()
        let presentation = try makePresentation(result: result, now: now)

        XCTAssertEqual(presentation.status.headline, "All good")
        XCTAssertNil(presentation.status.reason)
        XCTAssertEqual(presentation.status.tone, .green)
        XCTAssertEqual(presentation.backupLines.map(\.tone), [.green, .green])
        XCTAssertTrue(presentation.backupLines[0].text.hasPrefix("Database · Drive · last good today"))
        XCTAssertTrue(presentation.backupLines[1].text.hasPrefix("Transcripts · Drive · last good today"))
        XCTAssertFalse(presentation.backupLines.map(\.text).joined().contains("iCloud"))
        XCTAssertEqual(presentation.ingestRateText, "1.1 memories/min")
        XCTAssertEqual(presentation.ingestVolumeText, "67 new memories in 1 h")
        XCTAssertEqual(presentation.ingestChartLabel, "NEW MEMORIES")
        XCTAssertFalse(presentation.ingestAccessibilitySummary.contains("chunk"))
    }

    @MainActor
    func testNewTodayCountsHourlyBucketsSinceLocalMidnight() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.todayBoundaryResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertEqual(presentation.newToday, 7)
    }

    @MainActor
    func testMissingHourlyBucketsDoesNotSubstituteRollingWindowOrCollectorCount() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.missingHourlyBucketsResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertNil(presentation.newToday)
    }

    @MainActor
    func testLoadingObservabilityIsNeutralInsteadOfAThingNeedingAttention() throws {
        let presentation = try makePresentation(
            result: .unreadable("Loading observability data."),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertEqual(presentation.status.headline, "Checking…")
        XCTAssertNil(presentation.status.reason)
        XCTAssertEqual(presentation.status.tone, .neutral)
    }

    @MainActor
    private func makePresentation(
        result: ObservabilityReadResult,
        now: Date
    ) throws -> BrainBarOnePagePresentation {
        let collector = BrainBarDashboardFixture.makeCollector()
        let flow = DashboardFlowSummary.derive(
            daemon: collector.daemon,
            stats: collector.stats,
            now: now
        )
        let hero = BrainBarHeroPresentation.derive(
            flow: flow,
            stats: collector.stats,
            backupTruth: BrainBarHeroBackupTruth.derive(
                from: result,
                now: now,
                cadence: .known(300)
            ),
            locale: Locale(identifier: "en_US")
        )
        return BrainBarOnePagePresentation.derive(
            snapshotFreshness: collector.snapshotFreshnessState,
            hero: hero,
            observability: result,
            stats: collector.stats,
            ingest: flow.allCommits,
            now: now,
            calendar: BrainBarOnePageTestFixture.calendar,
            locale: Locale(identifier: "en_US")
        )
    }
}

enum BrainBarOnePageTestFixture {
    static let now = Date(timeIntervalSince1970: 1_789_387_200) // 2026-09-14 12:00:00Z
    static var calendar: Calendar {
        var value = Calendar(identifier: .gregorian)
        value.timeZone = TimeZone(identifier: "Asia/Jerusalem")!
        value.locale = Locale(identifier: "en_US")
        return value
    }

    static func healthyResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: [
            .init(hour: Date(timeIntervalSince1970: 1_789_354_800), count: 2),
            .init(hour: Date(timeIntervalSince1970: 1_789_383_600), count: 5),
        ]))
    }

    static func todayBoundaryResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: [
            .init(hour: Date(timeIntervalSince1970: 1_789_351_200), count: 11),
            .init(hour: Date(timeIntervalSince1970: 1_789_354_800), count: 2),
            .init(hour: Date(timeIntervalSince1970: 1_789_383_600), count: 5),
        ]))
    }

    static func missingHourlyBucketsResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: nil))
    }

    private static func document(
        byHour: [ObservabilityDocument.HourBucket]?
    ) throws -> ObservabilityDocument {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "observability-main-58849a70",
            withExtension: "json",
            subdirectory: "Fixtures"
        ))
        guard case let .readable(base) = ObservabilityReader.read(url: url) else {
            throw XCTSkip("observability fixture was unreadable")
        }
        return ObservabilityDocument(
            schemaVersion: base.schemaVersion,
            generatedAt: now,
            dbPath: base.dbPath,
            windowHours: base.windowHours,
            stores: .init(
                state: "measured",
                reason: "",
                inputs: [],
                totalChunks: 797_727,
                inWindow: .init(count: 18, byHour: byHour)
            ),
            emitters: base.emitters,
            authorUnknown: base.authorUnknown,
            backups: .init(
                state: "measured",
                reason: "",
                inputs: [],
                freshness: "fresh",
                thresholdHours: 36,
                retentionInvariant: "PASS",
                survivingArchives30D: 3,
                errorType: nil,
                lastVerifiedUpload: .init(
                    at: now.addingTimeInterval(-3_600),
                    ageHours: 1,
                    archiveId: "transcripts-verified",
                    verified: true
                ),
                dbSnapshot: .init(
                    lastAt: now.addingTimeInterval(-7_200),
                    destination: "brainlayer-verified.db.gz",
                    verified: true
                ),
                launchd: .init(
                    label: "com.brainlayer.jsonl-backup",
                    bootstrapped: true,
                    disabledDirPresent: false
                )
            )
        )
    }
}
