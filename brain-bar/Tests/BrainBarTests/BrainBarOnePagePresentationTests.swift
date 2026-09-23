import Foundation
import XCTest
@testable import BrainBar

final class BrainBarOnePagePresentationTests: XCTestCase {
    func testQueueDirectionDistinguishesPausedFromRunningEnrichment() {
        let paused = BrainBarQueueDirectionPresentation.derive(.growing, backlogCount: 42, enrichmentPaused: true)
        XCTAssertEqual(paused.label, "Enrichment paused · 42 queued")
        XCTAssertEqual(paused.tone, .neutral)

        let running = BrainBarQueueDirectionPresentation.derive(.growing, backlogCount: 42, enrichmentPaused: false)
        XCTAssertEqual(running.label, "Queue growing")
        XCTAssertEqual(running.tone, .warning)

        let unknown = BrainBarQueueDirectionPresentation.derive(.growing, backlogCount: 42, enrichmentPaused: nil)
        XCTAssertEqual(unknown.label, "Queue growing")
        XCTAssertEqual(unknown.tone, .warning)

        let offline = BrainBarQueueDirectionPresentation.derive(.unavailable, backlogCount: 42, enrichmentPaused: true)
        XCTAssertEqual(offline.label, "Queue offline")
        XCTAssertEqual(offline.tone, .error)
        XCTAssertEqual(BrainBarQueueDirectionPresentation.derive(.empty, backlogCount: 0, enrichmentPaused: true).label, "Queue empty")
        XCTAssertEqual(BrainBarQueueDirectionPresentation.derive(.draining, backlogCount: 42, enrichmentPaused: true).tone, .neutral)
    }

    @MainActor
    func testRenderFixtureActuallyHasGrowingBacklog() {
        let collector = BrainBarDashboardFixture.makeCollector(stats: BrainBarDashboardFixture.growingQueueStats)
        let flow = DashboardFlowSummary.derive(
            daemon: collector.daemon,
            stats: collector.stats,
            now: BrainBarDashboardFixture.fetchedAt
        )
        XCTAssertEqual(flow.queue.status, .growing)
        XCTAssertEqual(flow.queue.backlogCount, 12_840)
    }

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
            "not measured yet today"
        )
        XCTAssertEqual(presentation.agentWritesCount, 175)
        XCTAssertEqual(
            presentation.agentWritesText,
            "brain_store writes (24 h) · as of 2026-09-13 23:50 (stale)"
        )
    }

    @MainActor
    func testSameDayObservabilityPastCadenceIsUnavailable() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.staleSameDayResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertEqual(presentation.indexedToday, 99)
        XCTAssertEqual(
            presentation.indexedTodayUnavailableText,
            "as of 14:45 (stale)"
        )
        XCTAssertEqual(presentation.agentWritesCount, 175)
        XCTAssertEqual(
            presentation.agentWritesText,
            "brain_store writes (24 h) · as of 14:45 (stale)"
        )
        XCTAssertEqual(presentation.agentWritesDetailText(locale: Locale(identifier: "en_US")), "175 · as of 14:45 (stale)")
    }

    @MainActor
    func testStaleUnmeasuredDocumentShowsNoInventedCounts() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.staleUnmeasuredResult(),
            now: BrainBarOnePageTestFixture.now
        )
        XCTAssertNil(presentation.indexedToday)
        XCTAssertEqual(presentation.indexedTodayUnavailableText, "not measured yet today")
        XCTAssertNil(presentation.agentWritesCount)
        XCTAssertEqual(presentation.agentWritesText, "brain_store writes · not measured yet")
    }

    @MainActor
    func testGeneratedAtTrustMatrixNeverRendersUntrustworthyValues() throws {
        let now = Date(timeIntervalSince1970: 1_789_419_780) // 2026-09-15 00:03:00 Asia/Jerusalem
        let missing = try BrainBarOnePageTestFixture.invalidGeneratedAtResult(.missing)
        let null = try BrainBarOnePageTestFixture.invalidGeneratedAtResult(.null)
        let unparseable = try BrainBarOnePageTestFixture.invalidGeneratedAtResult(.unparseable)
        let cases: [(
            name: String,
            result: ObservabilityReadResult,
            reachabilityNote: String,
            indexedReason: String,
            agentWritesText: String
        )] = [
            (
                "stale",
                try BrainBarOnePageTestFixture.result(generatedAt: now.addingTimeInterval(-901)),
                "A decoded document can exceed the two-cadence age bound.",
                "not measured yet today",
                "brain_store writes (24 h) · as of 2026-09-14 23:47 (stale)"
            ),
            (
                "future-dated",
                try BrainBarOnePageTestFixture.result(generatedAt: now.addingTimeInterval(300)),
                "Clock skew can decode to a generated_at later than now.",
                "observability generated_at is in the future",
                "brain_store writes · not measured yet"
            ),
            (
                "missing",
                missing.result,
                "The non-optional Date cannot be constructed; ObservabilityReader returns unreadable.",
                missing.reason,
                "brain_store writes · not measured yet"
            ),
            (
                "null",
                null.result,
                "The non-optional Date cannot decode null; ObservabilityReader returns unreadable.",
                null.reason,
                "brain_store writes · not measured yet"
            ),
            (
                "unparseable",
                unparseable.result,
                "The custom ISO-8601 decoder rejects invalid text before a document exists.",
                unparseable.reason,
                "brain_store writes · not measured yet"
            ),
            (
                "epoch-sentinel",
                try BrainBarOnePageTestFixture.result(generatedAt: Date(timeIntervalSince1970: 0)),
                "A syntactically valid epoch sentinel can decode but is not trustworthy evidence.",
                "observability generated_at is zero or epoch sentinel",
                "brain_store writes · not measured yet"
            ),
            (
                "pre-midnight-today-scope",
                try BrainBarOnePageTestFixture.result(generatedAt: now.addingTimeInterval(-300)),
                "Fresh evidence from 23:58 is outside the 00:03 today window but valid for rolling 24 h.",
                "not measured yet today",
                "175 writes via brain_store in 24 h"
            ),
        ]

        for item in cases {
            let context = "\(item.name): \(item.reachabilityNote)"
            let presentation = try makePresentation(result: item.result, now: now)
            XCTAssertNil(presentation.indexedToday, context)
            XCTAssertEqual(
                presentation.indexedTodayUnavailableText,
                ["stale", "pre-midnight-today-scope"].contains(item.name)
                    ? item.indexedReason : "Indexed today unavailable: \(item.indexedReason)",
                context
            )
            XCTAssertEqual(presentation.agentWritesText, item.agentWritesText, context)
            if item.name == "stale" { XCTAssertEqual(presentation.agentWritesCount, 175, context) }
        }
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
        XCTAssertEqual(presentation.attentionItems, ["Agent activity could not be measured."])
    }

    @MainActor
    func testBackupFailureOutranksUnmeasuredAgentActivityInTopStrip() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.unverifiedResult(),
            now: BrainBarOnePageTestFixture.now,
            agentActivity: .unavailable("ps capture failed")
        )

        XCTAssertEqual(presentation.status.headline, "2 things need you")
        XCTAssertTrue(presentation.status.reason?.hasPrefix("Last transcript upload (NOT verified):") == true)
        XCTAssertEqual(presentation.status.tone, .amber)
        XCTAssertEqual(presentation.attentionItems.count, 2)
    }

    @MainActor
    func testBackupFailureOutranksSnapshotStatesInTopStrip() throws {
        let cases: [(SnapshotFreshnessState, String)] = [
            (.loading, "1 thing needs you"),
            (.stale(ageSeconds: 61), "2 things need you"),
            (.error(message: "Fixture fetch failed", lastSuccessAgeSeconds: 15), "2 things need you"),
        ]

        for (snapshotFreshness, expectedHeadline) in cases {
            let presentation = try makePresentation(
                result: BrainBarOnePageTestFixture.unverifiedResult(),
                now: BrainBarOnePageTestFixture.now,
                snapshotFreshness: snapshotFreshness
            )

            XCTAssertEqual(presentation.status.headline, expectedHeadline)
            XCTAssertTrue(
                presentation.status.reason?.hasPrefix("Last transcript upload (NOT verified):") == true,
                "Backup reason must outrank \(snapshotFreshness)."
            )
            XCTAssertEqual(presentation.status.tone, .amber)
            XCTAssertEqual(presentation.attentionItems.count, expectedHeadline == "1 thing needs you" ? 1 : 2)
        }
    }

    @MainActor
    func testLoadingSnapshotDoesNotPromoteStatsDerivedHeroUnknowns() throws {
        let collector = StatsCollector(
            dbPath: "/dev/null",
            daemonMonitor: DaemonHealthMonitor(targetPID: getpid())
        )
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.healthyResult(),
            now: BrainBarOnePageTestFixture.now,
            agentActivity: .unavailable("not sampled yet"),
            snapshotFreshness: .loading,
            collector: collector
        )

        XCTAssertEqual(presentation.status.headline, "Checking…")
        XCTAssertNil(presentation.status.reason)
        XCTAssertEqual(presentation.status.tone, .neutral)
    }

    @MainActor
    func testBackupFailureIsStructurallyFirstAcrossEveryOperatorState() throws {
        for operatorState in BrainBarDashboardFixture.OperatorState.allCases {
            let collector = BrainBarDashboardFixture.makeCollector(operatorState)
            let presentation = try makePresentation(
                result: BrainBarOnePageTestFixture.unverifiedResult(),
                now: BrainBarOnePageTestFixture.now,
                snapshotFreshness: collector.snapshotFreshnessState,
                collector: collector
            )

            XCTAssertTrue(
                presentation.status.reason?.hasPrefix("Last transcript upload (NOT verified):") == true,
                "Backup failure must remain first for \(operatorState)."
            )
            XCTAssertEqual(presentation.status.tone, .amber)
        }
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
    func testIndexedTodayOverflowIsUnavailableInsteadOfTrapping() throws {
        let presentation = try makePresentation(
            result: BrainBarOnePageTestFixture.overflowingTodayResult(),
            now: BrainBarOnePageTestFixture.now
        )

        XCTAssertNil(presentation.indexedToday)
        XCTAssertEqual(
            presentation.indexedTodayUnavailableText,
            "Indexed today unavailable: hourly observability count overflow"
        )
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
        agentActivity: AgentActivitySnapshot = BrainBarDashboardFixture.agentActivity,
        snapshotFreshness: SnapshotFreshnessState? = nil,
        collector providedCollector: StatsCollector? = nil
    ) throws -> BrainBarOnePagePresentation {
        let collector = providedCollector ?? BrainBarDashboardFixture.makeCollector()
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
            snapshotFreshness: snapshotFreshness ?? collector.snapshotFreshnessState,
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
    enum InvalidGeneratedAtShape {
        case missing
        case null
        case unparseable
    }

    private enum FixtureError: Error {
        case expectedUnreadableGeneratedAt
    }

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

    static func overflowingTodayResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: [
            .init(hour: now.addingTimeInterval(-60), count: Int.max),
            .init(hour: now, count: 1),
        ]))
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

    static func staleUnmeasuredResult() throws -> ObservabilityReadResult {
        .readable(try document(byHour: nil, generatedAt: now.addingTimeInterval(-900), measured: false))
    }

    static func result(generatedAt: Date) throws -> ObservabilityReadResult {
        .readable(try document(
            byHour: [.init(hour: generatedAt, count: 9)],
            generatedAt: generatedAt
        ))
    }

    static func invalidGeneratedAtResult(
        _ shape: InvalidGeneratedAtShape
    ) throws -> (result: ObservabilityReadResult, reason: String) {
        let sourceURL = try XCTUnwrap(Bundle.module.url(
            forResource: "observability-main-58849a70",
            withExtension: "json",
            subdirectory: "Fixtures"
        ))
        let source = try Data(contentsOf: sourceURL)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: source) as? [String: Any])
        switch shape {
        case .missing:
            object.removeValue(forKey: "generated_at")
        case .null:
            object["generated_at"] = NSNull()
        case .unparseable:
            object["generated_at"] = "not-a-date"
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-generated-at-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        let result = ObservabilityReader.read(url: url)
        guard case let .unreadable(reason) = result else {
            throw FixtureError.expectedUnreadableGeneratedAt
        }
        return (result, reason)
    }

    private static func document(
        byHour: [ObservabilityDocument.HourBucket]?,
        generatedAt: Date = now,
        backupsVerified: Bool = true,
        measured: Bool = true
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
                state: measured ? "measured" : "unmeasurable",
                reason: measured ? "" : "no sample",
                inputs: [],
                totalChunks: measured ? 797_727 : nil,
                inWindow: measured ? .init(count: 18, byHour: byHour) : nil
            ),
            emitters: .init(
                state: measured ? "measured" : "unmeasurable",
                reason: measured ? "" : "no sample",
                inputs: [],
                byEmitter: measured ? [.init(emitter: "mcp", countInWindow: 175)] : nil,
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
