import XCTest
@testable import BrainBar

final class InjectionPresentationTests: XCTestCase {
    func testSnapshotBuildsBurstGroupsAndWindowSummary() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:58:00Z", query: "latest cc release", chunkIDs: ["chunk-1", "chunk-2"], tokenCount: 48),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:55:00Z", query: "extract transcript", chunkIDs: ["chunk-3"], tokenCount: 52),
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
}
