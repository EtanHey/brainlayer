import XCTest
@testable import BrainBar

final class InjectionPresentationTests: XCTestCase {
    func testSnapshotBuildsBurstGroupsAndWindowSummary() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:58:00Z", query: "latest cc release", chunkIDs: ["chunk-1", "chunk-2"], tokenCount: 48),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:55:00Z", query: "latest cc release", chunkIDs: ["chunk-3"], tokenCount: 52),
            makeEvent(id: 3, sessionID: "sess-a", timestamp: "2026-04-18T09:40:00Z", query: "why is ffmpeg stuck", chunkIDs: ["chunk-4", "chunk-5", "chunk-6"], tokenCount: 91),
            makeEvent(id: 4, sessionID: "sess-b", timestamp: "2026-04-18T09:38:00Z", query: "brainbar graph empty", chunkIDs: ["chunk-7"], tokenCount: 40),
        ]

        let snapshot = InjectionPresentation.snapshot(
            events: events,
            filterText: "",
            now: now,
            windowMinutes: 60,
            burstGapMinutes: 8,
            bucketCount: 6
        )

        XCTAssertEqual(snapshot.summary.queryCount, 4)
        XCTAssertEqual(snapshot.summary.chunkCount, 7)
        XCTAssertEqual(snapshot.summary.tokenCount, 231)
        XCTAssertEqual(snapshot.summary.activeSessionCount, 2)
        XCTAssertEqual(snapshot.summary.burstCount, 3)
        XCTAssertEqual(snapshot.bursts.count, 3)
        XCTAssertEqual(snapshot.bursts[0].sessionID, "sess-a")
        XCTAssertEqual(snapshot.bursts[0].events.map { $0.id }, [1, 2])
        XCTAssertEqual(snapshot.bursts[1].events.map { $0.id }, [3])
        XCTAssertEqual(snapshot.bursts[2].sessionID, "sess-b")
        XCTAssertEqual(snapshot.ribbonBuckets, [0, 0, 0, 1, 1, 2])
    }

    func testFilterMatchesSessionQueryAndChunkIDs() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-alpha", timestamp: "2026-04-18T09:58:00Z", query: "latest cc release", chunkIDs: ["chunk-a"], tokenCount: 12),
            makeEvent(id: 2, sessionID: "sess-beta", timestamp: "2026-04-18T09:55:00Z", query: "brainbar graph", chunkIDs: ["chunk-b", "ops-note"], tokenCount: 24),
        ]

        let querySnapshot = InjectionPresentation.snapshot(
            events: events,
            filterText: "graph",
            now: now
        )
        XCTAssertEqual(querySnapshot.filteredEvents.map { $0.id }, [2])

        let sessionSnapshot = InjectionPresentation.snapshot(
            events: events,
            filterText: "alpha",
            now: now
        )
        XCTAssertEqual(sessionSnapshot.filteredEvents.map { $0.id }, [1])

        let chunkSnapshot = InjectionPresentation.snapshot(
            events: events,
            filterText: "ops-note",
            now: now
        )
        XCTAssertEqual(chunkSnapshot.filteredEvents.map { $0.id }, [2])
    }

    func testBurstAggregationGroupsFiveEventsFromOneSessionAndTopicInsideFiveMinutes() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = (0..<5).map { offset in
            makeEvent(
                id: Int64(offset + 1),
                sessionID: "brain-c7a8",
                timestamp: isoTimestamp(now.addingTimeInterval(-Double(offset) * 60.0)),
                query: "auth refactor",
                chunkIDs: ["chunk-\(offset + 1)"],
                tokenCount: 10
            )
        }

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 1)
        XCTAssertEqual(snapshot.bursts[0].sessionID, "brain-c7a8")
        XCTAssertEqual(snapshot.bursts[0].topicOrSource, "auth refactor")
        XCTAssertEqual(snapshot.bursts[0].summaryTitle, "5 chunks injected from \"auth refactor\"")
        XCTAssertEqual(snapshot.bursts[0].events.map(\.id), [1, 2, 3, 4, 5])
        XCTAssertEqual(snapshot.bursts[0].chunkPreviewIDs, ["chunk-1", "chunk-2"])
        XCTAssertEqual(snapshot.bursts[0].remainingChunkCount(after: 2), 3)
    }

    func testBurstAggregationSplitsDifferentSessions() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-b", timestamp: "2026-04-18T09:58:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
            makeEvent(id: 3, sessionID: "sess-a", timestamp: "2026-04-18T09:57:00Z", query: "auth refactor", chunkIDs: ["chunk-3"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 3)
        XCTAssertEqual(snapshot.bursts.map(\.sessionID), ["sess-a", "sess-b", "sess-a"])
    }

    func testBurstAggregationSplitsSameSessionWhenTopicChanges() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:58:00Z", query: "db migration", chunkIDs: ["chunk-2"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 2)
        XCTAssertEqual(snapshot.bursts.map(\.topicOrSource), ["auth refactor", "db migration"])
    }

    func testBurstAggregationSplitsSameSessionWhenEventsAreMoreThanSixtyMinutesApart() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T08:58:59Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 2)
        XCTAssertEqual(snapshot.bursts.map { $0.events.map(\.id) }, [[1], [2]])
    }

    func testBurstAggregationChainsConsecutiveEventsUnderSixtyMinutesApart() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T10:00:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:01:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
            makeEvent(id: 3, sessionID: "sess-a", timestamp: "2026-04-18T08:02:00Z", query: "auth refactor", chunkIDs: ["chunk-3"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 1)
        XCTAssertEqual(snapshot.bursts[0].events.map(\.id), [1, 2, 3])
    }

    func testBurstSectionsSplitSpanningBurstAtRibbonBucketBoundaries() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:01:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
            makeEvent(id: 3, sessionID: "sess-a", timestamp: "2026-04-18T08:02:00Z", query: "auth refactor", chunkIDs: ["chunk-3"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 1)
        XCTAssertEqual(snapshot.bursts[0].events.map(\.id), [1, 2, 3])
        XCTAssertEqual(snapshot.burstSections.map(\.bucket), [.lastSixtyMinutes, .oneToTwoHoursAgo])
        XCTAssertEqual(snapshot.burstSections.map { $0.bursts.flatMap { $0.events.map(\.id) } }, [[1, 2], [3]])
    }

    func testRibbonBucketBoundaryTreatsExactlySixtyMinutesAsOlderBucket() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:00:01Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-b", timestamp: "2026-04-18T09:00:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.summary.queryCount, 1)
        XCTAssertEqual(snapshot.burstSections.map(\.bucket), [.lastSixtyMinutes, .oneToTwoHoursAgo])
    }

    func testBurstAggregationExcludesFutureDatedEvents() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T10:01:00Z", query: "future skew", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.filteredEvents.map(\.id), [2])
        XCTAssertEqual(snapshot.windowEvents.map(\.id), [2])
        XCTAssertEqual(snapshot.bursts.map { $0.events.map(\.id) }, [[2]])
        XCTAssertEqual(snapshot.burstSections.map(\.bucket), [.lastSixtyMinutes])
    }

    func testFilterCountExcludesFutureDatedMatches() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T10:01:00Z", query: "future skew", chunkIDs: ["chunk-1"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "future", now: now)

        XCTAssertTrue(snapshot.filteredEvents.isEmpty)
        XCTAssertTrue(snapshot.bursts.isEmpty)
    }

    func testSummaryBurstCountUsesLastHourWindowOnly() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-b", timestamp: "2026-04-18T07:59:00Z", query: "older migration", chunkIDs: ["chunk-2"], tokenCount: 10),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 2)
        XCTAssertEqual(snapshot.summary.burstCount, 1)
    }

    func testBurstIDStaysStableWhenNewerEventAppendsToSameBurst() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let initialEvents = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:58:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:57:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10),
        ]
        let updatedEvents = [
            makeEvent(id: 3, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-3"], tokenCount: 10),
        ] + initialEvents

        let initialSnapshot = InjectionPresentation.snapshot(events: initialEvents, filterText: "", now: now)
        let updatedSnapshot = InjectionPresentation.snapshot(events: updatedEvents, filterText: "", now: now)

        XCTAssertEqual(initialSnapshot.bursts.count, 1)
        XCTAssertEqual(updatedSnapshot.bursts.count, 1)
        XCTAssertEqual(updatedSnapshot.bursts[0].events.map(\.id), [3, 1, 2])
        XCTAssertEqual(initialSnapshot.bursts[0].id, updatedSnapshot.bursts[0].id)
    }

    private func makeEvent(
        id: Int64,
        sessionID: String,
        timestamp: String,
        query: String,
        chunkIDs: [String],
        tokenCount: Int
    ) -> InjectionEvent {
        InjectionEvent(
            id: id,
            sessionID: sessionID,
            timestamp: timestamp,
            query: query,
            chunkIDs: chunkIDs,
            tokenCount: tokenCount
        )
    }

    private func isoDate(_ text: String) -> Date {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: text)!
    }

    private func isoTimestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }
}
