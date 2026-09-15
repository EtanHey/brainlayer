import Foundation
import XCTest
@testable import BrainBar

final class BrainBarOnePagePresentationTests: XCTestCase {
    @MainActor
    func testHealthyPresentationNamesIndexedChunksAndSeparatesAgentWrites() throws {
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
        XCTAssertEqual(presentation.totalIndexedChunks, 797_727)
        XCTAssertEqual(presentation.indexedToday, 7)
        XCTAssertNil(presentation.indexedTodayUnavailableText)
        XCTAssertEqual(presentation.agentWritesText, "175 writes via brain_store in 24 h")
    }

    @MainActor
    func testStaleObservabilityNeverReportsZeroTodayOrCurrentAgentWrites() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.staleResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertNil(presentation.indexedToday)
        XCTAssertEqual(
            presentation.indexedTodayUnavailableText,
            "Indexed today unavailable: observability as of 23:50"
        )
        XCTAssertEqual(
            presentation.agentWritesText,
            "brain_store writes unavailable: observability as of 23:50"
        )
    }

    @MainActor
    func testSameDayObservabilityPastCadenceIsUnavailable() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.staleSameDayResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertNil(presentation.indexedToday)
        XCTAssertEqual(
            presentation.indexedTodayUnavailableText,
            "Indexed today unavailable: observability as of 14:45"
        )
        XCTAssertEqual(
            presentation.agentWritesText,
            "brain_store writes unavailable: observability as of 14:45"
        )
    }

    @MainActor
    func testFreshRollingAgentWriteWindowRemainsAvailableAcrossMidnight() throws {
        let now = Date(timeIntervalSince1970: 1_789_419_780) // 2026-09-15 00:03:00 Asia/Jerusalem
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.crossMidnightFreshResult(now: now),
            now: now
        )

        XCTAssertNil(presentation.indexedToday)
        XCTAssertEqual(
            presentation.indexedTodayUnavailableText,
            "Indexed today unavailable: observability as of 23:58"
        )
        XCTAssertEqual(presentation.agentWritesText, "175 writes via brain_store in 24 h")
    }

    @MainActor
    func testUnmeasuredAgentActivityRaisesTopStripAlert() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.healthyResult(),
            now: BrainBarOnePageTestFixture.now,
            agentActivity: .unavailable("ps capture failed")
        )

        XCTAssertEqual(presentation.status.headline, "1 thing needs you")
        XCTAssertEqual(presentation.status.reason, "Agent activity could not be measured.")
        XCTAssertEqual(presentation.status.tone, .amber)
    }

    @MainActor
    func testNewTodayCountsHourlyBucketsSinceLocalMidnight() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.todayBoundaryResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertEqual(presentation.indexedToday, 7)
    }

    @MainActor
    func testMissingHourlyBucketsDoesNotSubstituteRollingWindowOrCollectorCount() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.missingHourlyBucketsResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertNil(presentation.indexedToday)
    }

    @MainActor
    func testUnverifiedBackupTimestampIsNotCalledLastGood() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.unverifiedResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertEqual(presentation.backupLines.map(\.tone), [.red, .red])
        XCTAssertTrue(presentation.backupLines.allSatisfy { $0.text.hasSuffix("no verified copy") })
        XCTAssertFalse(presentation.backupLines.map(\.text).joined().contains("last good"))
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
        now: Date,
        agentActivity: AgentActivitySnapshot = BrainBarDashboardFixture.agentActivity
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
            agentActivity: agentActivity,
            now: now,
            calendar: BrainBarOnePageTestFixture.calendar,
            locale: Locale(identifier: "en_US"),
            observabilityCadence: .known(300)
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
            .init(hour: Date(timeIntervalSince1970: 1_789_333_200), count: 2),
            .init(hour: Date(timeIntervalSince1970: 1_789_383_600), count: 5),
        ]))
    }

    static func dashboardResult(indexedToday: Int) throws -> ObservabilityReadResult {
        .readable(try document(byHour: [.init(hour: now, count: indexedToday)]))
    }

    static func todayBoundaryResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: [
            .init(hour: Date(timeIntervalSince1970: 1_789_329_600), count: 11),
            .init(hour: Date(timeIntervalSince1970: 1_789_333_200), count: 2),
            .init(hour: Date(timeIntervalSince1970: 1_789_383_600), count: 5),
        ]))
    }

    static func missingHourlyBucketsResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: nil))
    }

    static func unverifiedResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: [], backupsVerified: false))
    }

    static func staleResult() throws -> ObservabilityReadResult {
        .readable(try document(
            byHour: [],
            generatedAt: Date(timeIntervalSince1970: 1_789_332_600)
        ))
    }

    static func staleSameDayResult() throws -> ObservabilityReadResult {
        .readable(try document(
            byHour: [.init(hour: now.addingTimeInterval(-900), count: 99)],
            generatedAt: now.addingTimeInterval(-900)
        ))
    }

    static func crossMidnightFreshResult(now: Date) throws -> ObservabilityReadResult {
        .readable(try document(
            byHour: [.init(hour: now.addingTimeInterval(-300), count: 9)],
            generatedAt: now.addingTimeInterval(-300)
        ))
    }

    private static func document(
        byHour: [ObservabilityDocument.HourBucket]?,
        generatedAt: Date = now,
        backupsVerified: Bool = true
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
            generatedAt: generatedAt,
            dbPath: base.dbPath,
            windowHours: base.windowHours,
            stores: .init(
                state: "measured",
                reason: "",
                inputs: [],
                totalChunks: 797_727,
                inWindow: .init(count: 18, byHour: byHour)
            ),
            emitters: .init(
                state: "measured",
                reason: "",
                inputs: [],
                byEmitter: [.init(emitter: "mcp", countInWindow: 175)],
                bySourceClass: base.emitters.bySourceClass,
                hiddenFromDefaultSearch: base.emitters.hiddenFromDefaultSearch
            ),
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
                    at: generatedAt.addingTimeInterval(-3_600),
                    ageHours: 1,
                    archiveId: "transcripts-verified",
                    verified: backupsVerified
                ),
                dbSnapshot: .init(
                    lastAt: generatedAt.addingTimeInterval(-7_200),
                    destination: "brainlayer-verified.db.gz",
                    verified: backupsVerified
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
