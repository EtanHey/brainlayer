import XCTest
import AppKit
import SwiftUI
@testable import BrainBar

final class InjectionPresentationTests: XCTestCase {
    func testExpandedBurstDetailsIDTargetsRevealedContentForScroll() {
        XCTAssertEqual(
            InjectionFeedView.expandedBurstDetailsID(burstID: "burst-42"),
            "expanded-burst-details-burst-42"
        )
    }

    func testFilterControlsUseNeutralTint() {
        XCTAssertFalse(InjectionFeedView.filterControlsUseAccentTint)
    }

    func testBurstChunkCounterUsesMeaningfulLabel() {
        XCTAssertEqual(InjectionFeedView.burstChunkCounterLabel, "memories surfaced into context")
    }

    func testExpandedEventFieldsSuppressDuplicateKindAndTrigger() {
        let event = makeEvent(
            id: 1,
            sessionID: "sess-a",
            timestamp: "2026-04-18T09:58:00Z",
            query: "Realtime Capture",
            chunkIDs: ["chunk-1"],
            tokenCount: 10,
            chunks: [
                makeChunk(
                    id: "chunk-1",
                    content: "Realtime Capture",
                    source: "realtime_watcher"
                )
            ]
        )

        XCTAssertEqual(event.displayTitle, "Realtime Capture")
        XCTAssertNil(event.expandedRowKindLabel)
        XCTAssertNil(event.expandedRowTriggeredByText)
    }

    func testInjectionFeedRendersSummaryViewOnLivePath() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/InjectionFeedView.swift")
        XCTAssertTrue(source.contains("InjectionSummaryView(events: snapshot.windowEvents)"))
    }

    @MainActor
    func testRendersInjectionSummaryLivePathQAImage() throws {
        let events = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:58:00Z", query: "auth refactor", chunkIDs: ["chunk-1", "chunk-2"], tokenCount: 48),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:55:00Z", query: "db migration", chunkIDs: ["chunk-3"], tokenCount: 52),
        ]
        let view = InjectionSummaryView(events: events)
            .padding(24)
            .frame(width: 430, height: 120)

        try renderInjectionPNG(view, name: "bug2-injection-summary-live-path.png")
    }

    @MainActor
    func testRendersDedupedExpandedEventQAImage() throws {
        let event = makeEvent(
            id: 1,
            sessionID: "sess-a",
            timestamp: "2026-04-18T09:58:00Z",
            query: "Realtime Capture",
            chunkIDs: ["chunk-1"],
            tokenCount: 10,
            chunks: [
                makeChunk(id: "chunk-1", content: "Realtime Capture", source: "realtime_watcher")
            ]
        )

        let view = VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 7) {
                Text(event.primaryKind.glyph)
                if let kindLabel = event.expandedRowKindLabel {
                    Text(kindLabel)
                        .font(.system(size: 11, weight: .semibold))
                }
                Text(event.displayTitle)
                    .font(.system(size: 14, weight: .semibold))
            }
            if let triggeredBy = event.expandedRowTriggeredByText {
                Text(triggeredBy)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            Text("Session sess-a   1 chunks   10 tok")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(width: 420, height: 120, alignment: .leading)

        try renderInjectionPNG(view, name: "bug4-deduped-expanded-event.png")
    }

    @MainActor
    func testRendersBurstChunkCounterQAImage() throws {
        let view = VStack(alignment: .trailing, spacing: 2) {
            Text("3")
                .font(.system(size: 24, weight: .bold, design: .rounded))
            Text(InjectionFeedView.burstChunkCounterLabel)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
                .lineLimit(2)
        }
        .padding(24)
        .frame(width: 240, height: 120, alignment: .trailing)

        try renderInjectionPNG(view, name: "bug5-meaningful-chunk-counter.png")
    }

    @MainActor
    func testRendersNeutralFilterControlsQAImage() throws {
        let view = VStack(alignment: .trailing, spacing: 8) {
            TextField("Filter injections", text: .constant(""))
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .frame(width: 260)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(InjectionFeedView.filterControlFillColor)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(InjectionFeedView.filterControlBorderColor, lineWidth: 1)
                )

            HStack(spacing: 6) {
                Text("All")
                    .font(.system(size: 11, weight: .semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(InjectionFeedView.filterControlSelectedFillColor))
                    .overlay(Capsule().stroke(InjectionFeedView.filterControlBorderColor, lineWidth: 1))
                Text("Claude")
                    .font(.system(size: 11, weight: .semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(InjectionFeedView.filterControlFillColor))
            }
        }
        .padding(24)
        .frame(width: 340, height: 140)

        try renderInjectionPNG(view, name: "bug3-neutral-filter-controls.png")
    }

    func testInjectionEventDeduplicatesChunkIDsForClickableRows() {
        let event = makeEvent(
            id: 1,
            sessionID: "sess-a",
            timestamp: "2026-04-18T09:58:00Z",
            query: "duplicate hit IDs",
            chunkIDs: ["chunk-a", "chunk-a", "chunk-b", "chunk-a"],
            tokenCount: 48
        )

        XCTAssertEqual(event.uniqueChunkIDs, ["chunk-a", "chunk-b"])
    }

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
        XCTAssertEqual(snapshot.bursts[0].summaryTitle, "5 chunks injected")
        XCTAssertEqual(snapshot.bursts[0].events.map(\.id), [1, 2, 3, 4, 5])
        XCTAssertEqual(snapshot.bursts[0].chunkPreviewIDs, ["chunk-1", "chunk-2"])
        XCTAssertEqual(snapshot.bursts[0].remainingChunkCount(after: 2), 3)
    }

    func testBurstSummaryTitleUsesChunkContentInsteadOfSourcePrompt() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let event = makeEvent(
            id: 1,
            sessionID: "sess-a",
            timestamp: "2026-04-18T09:58:00Z",
            query: "[orc gen-7 s:1 — researchLead monitor v2 — prompts-deliverable workflow]",
            chunkIDs: ["chunk-1"],
            tokenCount: 10,
            chunks: [
                makeChunk(id: "chunk-1", content: "Coverage moved to terminal-status semantics after the dashboard fix")
            ]
        )

        let snapshot = InjectionPresentation.snapshot(events: [event], filterText: "", now: now)

        XCTAssertEqual(
            snapshot.bursts[0].summaryTitle,
            "Coverage moved to terminal-status semantics after the dashboard fix"
        )
        XCTAssertEqual(snapshot.bursts[0].topicOrSource, event.query)
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

    func testBurstAggregationKeepsConversationIDOnlyWhenEventsAgree() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let conversationID = "3679128a-f371-445f-82ba-b3946e2f20b6"
        let matchingEvents = [
            makeEvent(id: 1, sessionID: "sess-a", timestamp: "2026-04-18T09:59:00Z", query: "auth refactor", chunkIDs: ["chunk-1"], tokenCount: 10, claudeConversationID: conversationID),
            makeEvent(id: 2, sessionID: "sess-a", timestamp: "2026-04-18T09:58:00Z", query: "auth refactor", chunkIDs: ["chunk-2"], tokenCount: 10, claudeConversationID: conversationID),
        ]
        let mixedEvents = [
            makeEvent(id: 3, sessionID: "sess-b", timestamp: "2026-04-18T09:57:00Z", query: "db migration", chunkIDs: ["chunk-3"], tokenCount: 10, claudeConversationID: conversationID),
            makeEvent(id: 4, sessionID: "sess-b", timestamp: "2026-04-18T09:56:00Z", query: "db migration", chunkIDs: ["chunk-4"], tokenCount: 10, claudeConversationID: "84e23cfb-dce6-4f93-a2be-41f74ae5f43f"),
        ]

        let matchingSnapshot = InjectionPresentation.snapshot(events: matchingEvents, filterText: "", now: now)
        let mixedSnapshot = InjectionPresentation.snapshot(events: mixedEvents, filterText: "", now: now)

        XCTAssertEqual(matchingSnapshot.bursts[0].claudeConversationID, conversationID)
        XCTAssertEqual(mixedSnapshot.bursts[0].claudeConversationID, "")
    }

    func testBurstAggregationUsesTriggerQueryWhenChunkSummariesDiffer() {
        let now = isoDate("2026-04-18T10:00:00Z")
        let events = [
            makeEvent(
                id: 1,
                sessionID: "sess-a",
                timestamp: "2026-04-18T09:59:00Z",
                query: "auth refactor",
                chunkIDs: ["chunk-1"],
                tokenCount: 10,
                chunks: [makeChunk(id: "chunk-1", content: "Auth service boundaries")]
            ),
            makeEvent(
                id: 2,
                sessionID: "sess-a",
                timestamp: "2026-04-18T09:58:00Z",
                query: "auth refactor",
                chunkIDs: ["chunk-2"],
                tokenCount: 10,
                chunks: [makeChunk(id: "chunk-2", content: "Session token migration")]
            ),
        ]

        let snapshot = InjectionPresentation.snapshot(events: events, filterText: "", now: now)

        XCTAssertEqual(snapshot.bursts.count, 1)
        XCTAssertEqual(snapshot.bursts[0].topicOrSource, "auth refactor")
        XCTAssertEqual(snapshot.bursts[0].events.map(\.id), [1, 2])
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

    func testPreviewChunksDeduplicatesRepeatedChunkIDsAcrossBurstEvents() {
        let events = [
            makeEvent(
                id: 1,
                sessionID: "sess-a",
                timestamp: "2026-04-18T09:58:00Z",
                query: "auth refactor",
                chunkIDs: ["shared", "unique-a"],
                tokenCount: 10,
                chunks: [
                    makeChunk(id: "shared", content: "Shared hit"),
                    makeChunk(id: "unique-a", content: "Unique hit A")
                ]
            ),
            makeEvent(
                id: 2,
                sessionID: "sess-a",
                timestamp: "2026-04-18T09:57:00Z",
                query: "auth refactor",
                chunkIDs: ["shared", "unique-b"],
                tokenCount: 10,
                chunks: [
                    makeChunk(id: "shared", content: "Shared hit again"),
                    makeChunk(id: "unique-b", content: "Unique hit B")
                ]
            )
        ]

        let preview = InjectionPresentation.previewChunks(for: events, limit: 3)

        XCTAssertEqual(preview.map(\.id), ["shared", "unique-a", "unique-b"])
    }

    private func makeEvent(
        id: Int64,
        sessionID: String,
        timestamp: String,
        query: String,
        chunkIDs: [String],
        tokenCount: Int,
        chunks: [InjectionChunk] = [],
        claudeConversationID: String = ""
    ) -> InjectionEvent {
        InjectionEvent(
            id: id,
            sessionID: sessionID,
            timestamp: timestamp,
            query: query,
            chunkIDs: chunkIDs,
            tokenCount: tokenCount,
            chunks: chunks,
            claudeConversationID: claudeConversationID
        )
    }

    private func makeChunk(id: String, content: String, source: String = "mcp") -> InjectionChunk {
        InjectionChunk(
            id: id,
            content: content,
            summary: "",
            source: source,
            sourceFile: "",
            tags: [],
            contentType: "memory"
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

    @MainActor
    private func renderInjectionPNG<V: View>(_ view: V, name: String) throws {
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else {
            XCTFail("Expected renderer to produce a PNG")
            return
        }

        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("docs.local/wave3-qa/\(name)")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try png.write(to: url)
        XCTAssertGreaterThan(png.count, 1_000)
    }

    private func brainBarSourceFile(_ relativePath: String, testFilePath: StaticString = #filePath) throws -> String {
        let testsDir = URL(fileURLWithPath: "\(testFilePath)").deletingLastPathComponent()
        let packageRoot = testsDir.deletingLastPathComponent().deletingLastPathComponent()
        let url = packageRoot.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }
}
