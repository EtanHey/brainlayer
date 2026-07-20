import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class InjectionSignalDensityContractTests: XCTestCase {
    deinit {}

    func testCollapsedBurstPresentsTriggerBeforeSelectedResultAndKeepsProvenance() throws {
        let now = isoDate("2026-07-19T10:00:00Z")
        let sourceFile = "/Users/example/project/.claude/projects/-Users-example-project/session.jsonl"
        let event = InjectionEvent(
            id: 42,
            sessionID: "session-12345678-raw",
            timestamp: "2026-07-19T09:58:00Z",
            query: "why did retrieval choose this memory",
            chunkIDs: ["chunk-raw-uuid", "chunk-secondary"],
            tokenCount: 88,
            chunks: [
                InjectionChunk(
                    id: "chunk-raw-uuid",
                    content: "The selected memory explains the routing decision.",
                    summary: "Routing decision memory",
                    source: "claude_code",
                    sourceFile: sourceFile,
                    tags: ["retrieval"],
                    contentType: "memory",
                    claudeConversationID: "conversation-1"
                ),
                InjectionChunk(
                    id: "chunk-secondary",
                    content: "A secondary memory adds supporting context.",
                    summary: "Supporting context",
                    source: "mcp",
                    sourceFile: sourceFile,
                    tags: ["retrieval"],
                    contentType: "memory"
                ),
            ]
        )

        let burst = try XCTUnwrap(
            InjectionPresentation.snapshot(events: [event], filterText: "", now: now).bursts.first
        )

        XCTAssertEqual(burst.queryTitle, event.query)
        XCTAssertEqual(burst.selectedResultSummary, "Routing decision memory")
        XCTAssertEqual(burst.sourceLabel, "Realtime Capture · claude_code")
        XCTAssertEqual(burst.projectLabel, "/Users/example/project")
        XCTAssertEqual(burst.resultCount, 2)
        XCTAssertEqual(burst.additionalResultPreviews.map(\.id), ["chunk-secondary"])
        XCTAssertEqual(burst.remainingCollapsedResultCount, 0)
        XCTAssertFalse(burst.timestampLabel.isEmpty)
    }

    func testMissingSelectedChunkNeverBorrowsAnotherResultsSummary() throws {
        let now = isoDate("2026-07-19T10:00:00Z")
        let available = InjectionChunk(
            id: "available-chunk",
            content: "Content belonging to a different result",
            summary: "Different result summary",
            source: "mcp",
            sourceFile: "",
            tags: [],
            contentType: "memory"
        )
        let event = InjectionEvent(
            id: 43,
            sessionID: "session-missing-selected",
            timestamp: "2026-07-19T09:58:00Z",
            query: "show exact attribution",
            chunkIDs: ["missing-selected", available.id],
            tokenCount: 10,
            chunks: [available]
        )

        let burst = try XCTUnwrap(
            InjectionPresentation.snapshot(events: [event], filterText: "", now: now).bursts.first
        )

        XCTAssertEqual(burst.selectedResultProvenance?.chunkID, "missing-selected")
        XCTAssertEqual(burst.selectedResultSummary, "Result unavailable")
        XCTAssertEqual(
            burst.remainingCollapsedResultCount,
            0,
            "The unresolved selected provenance is still rendered and must consume the selected slot."
        )
    }

    func testRepeatedResultUpgradesMissingProvenanceWhenLaterEventSuppliesChunk() throws {
        let now = isoDate("2026-07-19T10:00:00Z")
        let recovered = InjectionChunk(
            id: "repeated-chunk",
            content: "Recovered content from a later event in the burst",
            summary: "Recovered result summary",
            source: "claude_code",
            sourceFile: "/Users/example/project/.claude/projects/-Users-example-project/session.jsonl",
            tags: ["recovered"],
            contentType: "memory"
        )
        let events = [
            InjectionEvent(
                id: 52,
                sessionID: "session-repeated-result",
                timestamp: "2026-07-19T09:59:00Z",
                query: "show repeated result truth",
                chunkIDs: [recovered.id],
                tokenCount: 10,
                chunks: []
            ),
            InjectionEvent(
                id: 51,
                sessionID: "session-repeated-result",
                timestamp: "2026-07-19T09:58:30Z",
                query: "show repeated result truth",
                chunkIDs: [recovered.id],
                tokenCount: 9,
                chunks: [recovered]
            ),
        ]

        let burst = try XCTUnwrap(
            InjectionPresentation.snapshot(events: events, filterText: "", now: now).bursts.first
        )
        let provenance = try XCTUnwrap(burst.selectedResultProvenance)

        XCTAssertEqual(burst.resultCount, 1)
        XCTAssertEqual(provenance.eventID, 51)
        XCTAssertEqual(provenance.chunk, recovered)
        XCTAssertEqual(burst.selectedResultSummary, "Recovered result summary")
        XCTAssertEqual(burst.projectLabel, "/Users/example/project")
    }

    func testBurstProvenanceStaysPairedWithEachResult() throws {
        let now = isoDate("2026-07-19T10:00:00Z")
        let selected = InjectionChunk(
            id: "selected-chunk",
            content: "Selected result",
            summary: "Selected result summary",
            source: "claude_code",
            sourceFile: "",
            tags: ["selected", "truth"],
            contentType: "memory"
        )
        let later = InjectionChunk(
            id: "later-chunk",
            content: "Later result",
            summary: "Later result summary",
            source: "mcp",
            sourceFile: "/Users/other/project/.claude/projects/-Users-other-project/session.jsonl",
            tags: ["later"],
            contentType: "assistant_text"
        )
        let events = [
            InjectionEvent(
                id: 2,
                sessionID: "same-session",
                timestamp: "2026-07-19T09:59:00Z",
                query: "same trigger",
                chunkIDs: [selected.id],
                tokenCount: 10,
                chunks: [selected]
            ),
            InjectionEvent(
                id: 1,
                sessionID: "same-session",
                timestamp: "2026-07-19T09:58:00Z",
                query: "same trigger",
                chunkIDs: [later.id],
                tokenCount: 9,
                chunks: [later]
            ),
        ]

        let burst = try XCTUnwrap(
            InjectionPresentation.snapshot(events: events, filterText: "", now: now).bursts.first
        )
        let selectedProvenance = try XCTUnwrap(burst.selectedResultProvenance)

        XCTAssertEqual(burst.projectLabel, "Project unavailable")
        XCTAssertEqual(selectedProvenance.chunkID, selected.id)
        XCTAssertEqual(selectedProvenance.source, "claude_code")
        XCTAssertEqual(selectedProvenance.sourceFile, "")
        XCTAssertEqual(selectedProvenance.projectPath, "")
        XCTAssertEqual(selectedProvenance.contentType, "memory")
        XCTAssertEqual(selectedProvenance.tags, ["selected", "truth"])
        XCTAssertEqual(
            selectedProvenance.expandedMetadataLabels(isSelected: true),
            [
                "Selected result",
                "ID selected-chunk",
                "Source claude_code",
                "Source file unavailable",
                "Project unavailable",
                "Type memory",
                "Tags selected, truth",
            ]
        )
        XCTAssertEqual(
            burst.resultProvenance.map { "\($0.chunkID)|\($0.source)|\($0.projectPath)" },
            [
                "selected-chunk|claude_code|",
                "later-chunk|mcp|/Users/other/project",
            ]
        )
    }

    func testPresentationModelStartsInLoadingStateInsteadOfEmptyState() {
        XCTAssertEqual(
            reflectedDescription(named: "loadState", in: InjectionFeedPresentationState.empty),
            "loading"
        )
    }

    func testSurfaceStateDistinguishesLoadingDisconnectedEmptyFilteredAndDegraded() {
        let now = isoDate("2026-07-19T10:00:00Z")
        let emptySnapshot = InjectionPresentation.snapshot(events: [], filterText: "", now: now)
        let loadedEmpty = InjectionFeedPresentationState(events: [], degradationState: .healthy)

        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: emptySnapshot,
                presentationState: .empty,
                connectionState: .connected,
                filterActive: false
            ),
            .loading
        )
        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: emptySnapshot,
                presentationState: loadedEmpty,
                connectionState: .disconnected,
                filterActive: false
            ),
            .disconnected
        )
        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: emptySnapshot,
                presentationState: loadedEmpty,
                connectionState: .connected,
                filterActive: false
            ),
            .empty
        )
        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: emptySnapshot,
                presentationState: loadedEmpty,
                connectionState: .connected,
                filterActive: true
            ),
            .noMatches
        )
        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: emptySnapshot,
                presentationState: InjectionFeedPresentationState(
                    events: [],
                    degradationState: .degraded(reason: "read failed")
                ),
                connectionState: .connected,
                filterActive: false
            ),
            .degraded(reason: "read failed", retainsContent: false)
        )
    }

    func testActionReceiptsExposeSuccessFailureAndDisconnectedOutcomes() {
        XCTAssertEqual(
            InjectionActionReceipt.copyResult(copied: true),
            InjectionActionReceipt(kind: .success, message: "Resume command copied")
        )
        XCTAssertEqual(
            InjectionActionReceipt.copyResult(copied: false),
            InjectionActionReceipt(kind: .failure, message: "Couldn’t copy the resume command")
        )
        XCTAssertEqual(
            InjectionActionReceipt.threadOpenResult(errorDescription: nil),
            InjectionActionReceipt(kind: .success, message: "Thread opened")
        )
        XCTAssertEqual(
            InjectionActionReceipt.threadOpenResult(errorDescription: "database locked"),
            InjectionActionReceipt(kind: .failure, message: "Couldn’t open thread · database locked")
        )
        XCTAssertEqual(
            InjectionActionReceipt.disconnectedThread,
            InjectionActionReceipt(
                kind: .failure,
                message: "Thread unavailable in this disconnected snapshot"
            )
        )
    }

    func testActionReceiptGenerationPreventsOlderExpiryFromClearingNewerReceipt() {
        var generation = InjectionActionReceiptGeneration()
        let first = generation.next()
        let second = generation.next()

        XCTAssertFalse(generation.isCurrent(first))
        XCTAssertTrue(generation.isCurrent(second))
    }

    func testFailedInitialLoadResolvesToExplicitDegradedSurface() {
        let now = isoDate("2026-07-19T10:00:00Z")
        let emptySnapshot = InjectionPresentation.snapshot(events: [], filterText: "", now: now)
        let failed = InjectionFeedPresentationState(
            events: [],
            degradationState: .degraded(reason: "database locked"),
            loadState: .failed
        )

        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: emptySnapshot,
                presentationState: failed,
                connectionState: .connected,
                filterActive: false
            ),
            .degraded(reason: "database locked", retainsContent: false)
        )
    }

    func testDegradedStateRetainsContentAndSessionRailUsesTriggerLabel() throws {
        let now = isoDate("2026-07-19T10:00:00Z")
        let event = InjectionEvent(
            id: 7,
            sessionID: "550e8400-e29b-41d4-a716-446655440000",
            timestamp: "2026-07-19T09:59:00Z",
            query: "show operator truth",
            chunkIDs: ["chunk-7"],
            tokenCount: 21
        )
        let snapshot = InjectionPresentation.snapshot(events: [event], filterText: "", now: now)
        let degraded = InjectionFeedPresentationState(
            events: [event],
            degradationState: .degraded(reason: "database locked")
        )

        XCTAssertEqual(
            InjectionFeedSurfaceState.resolve(
                snapshot: snapshot,
                presentationState: degraded,
                connectionState: .connected,
                filterActive: false
            ),
            .degraded(reason: "database locked", retainsContent: true)
        )
        XCTAssertEqual(try XCTUnwrap(snapshot.sessions.first).displayLabel, "show operator truth")
        XCTAssertEqual(
            InjectionFeedView.burstGroupingDisclosure,
            "Retrieval bursts group the same session and trigger topic while consecutive events remain under 60 minutes."
        )
    }

    func testEmptyQueryUsesTruthfulSessionRailFallback() throws {
        let now = isoDate("2026-07-19T10:00:00Z")
        let event = InjectionEvent(
            id: 8,
            sessionID: "empty-query-session",
            timestamp: "2026-07-19T09:59:00Z",
            query: "   ",
            chunkIDs: [],
            tokenCount: 0
        )

        let snapshot = InjectionPresentation.snapshot(events: [event], filterText: "", now: now)

        XCTAssertEqual(try XCTUnwrap(snapshot.sessions.first).displayLabel, "Retrieval session")
    }

    func testInjectionViewDeclaresLiteralGroupingAccessibilityStateAndFixtureContracts() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/InjectionFeedView.swift")
            + (try brainBarSourceFile("Sources/BrainBar/InjectionPresentation.swift"))

        XCTAssertTrue(source.contains("Retrieval bursts"))
        XCTAssertTrue(source.contains("same session and trigger topic"))
        XCTAssertTrue(source.contains("under 60 minutes"))
        XCTAssertTrue(source.contains("filterSearchAccessibilityID"))
        XCTAssertTrue(source.contains("filterTypeAccessibilityID"))
        XCTAssertTrue(source.contains("burstActionAccessibilityID"))
        XCTAssertTrue(source.contains("Picker(\"Retrieval type\""))
        XCTAssertTrue(source.contains("actionReceipt"))
        XCTAssertTrue(source.contains("InjectionFeedFixture"))
        XCTAssertTrue(source.contains("case disconnected"))
        XCTAssertTrue(source.contains("case degraded"))
        XCTAssertTrue(source.contains("Text(\"\\(burst.resultCount)\")"))
        XCTAssertTrue(source.contains("Text(\"Timestamp \\(event.timestamp)\")"))
        XCTAssertTrue(source.contains("let copied = pasteboard.setString"))
        XCTAssertFalse(source.contains("Showing the full retrieval stream."))
        XCTAssertFalse(source.contains("Text(session.sessionID)"))
        XCTAssertFalse(source.contains("Text(chunkPreviewText(chunkID))"))
        XCTAssertFalse(source.contains("chip(text: \"\\(burst.resultCount) result"))
    }

    func testInjectionRenderHelperUsesOnlyTheCallerProvidedIsolatedDirectory() throws {
        let source = try brainBarSourceFile("Tests/BrainBarTests/InjectionPresentationTests.swift")

        XCTAssertTrue(source.contains("BRAINBAR_RENDER_DIR"))
        XCTAssertFalse(source.contains("docs.local/wave3-qa"))
    }

    func testWindowRootRoutesUnavailableInjectionStoreToDisconnectedFeed() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(source.contains("InjectionFeedView(disconnectedAt: Date())"))
        XCTAssertFalse(
            source.contains(
                "BrainBarLoadingView(title: \"Injections\", subtitle: BrainBarPlaceholderCopy.injectionFeedNotWired)"
            )
        )
    }

    @MainActor
    func testRendersOverviewExpandedEmptyAndDegradedFixtureStates() throws {
        guard let renderDirectory = ProcessInfo.processInfo.environment["BRAINBAR_RENDER_DIR"],
              !renderDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("Set BRAINBAR_RENDER_DIR to render isolated P3 fixture snapshots")
        }

        let now = isoDate("2026-07-19T10:00:00Z")
        let events = fixtureEvents()
        let burstID = try XCTUnwrap(
            InjectionPresentation.snapshot(events: events, filterText: "", now: now).bursts.first?.id
        )
        let scenarios: [(String, InjectionFeedFixture)] = [
            (
                "p3-injections-overview.png",
                InjectionFeedFixture(
                    events: events,
                    now: now,
                    actionReceipt: InjectionActionReceipt(kind: .success, message: "Resume command copied")
                )
            ),
            (
                "p3-injections-expanded.png",
                InjectionFeedFixture(events: events, now: now, expandedBurstIDs: [burstID])
            ),
            (
                "p3-injections-empty.png",
                InjectionFeedFixture(events: [], now: now)
            ),
            (
                "p3-injections-disconnected.png",
                InjectionFeedFixture(
                    events: [],
                    now: now,
                    connectionState: .disconnected
                )
            ),
            (
                "p3-injections-degraded.png",
                InjectionFeedFixture(
                    events: events,
                    now: now,
                    degradationState: .degraded(reason: "Read-only database snapshot is temporarily unavailable")
                )
            ),
        ]

        for (name, fixture) in scenarios {
            let view = InjectionFeedView(fixture: fixture)
                .environment(\.colorScheme, .dark)
                .frame(width: 1_180, height: 820)

            try render(view, name: name, directory: renderDirectory)
        }
    }

    private func reflectedDescription(named name: String, in value: Any) -> String? {
        reflectedValue(named: name, in: value).map { String(describing: $0) }
    }

    private func reflectedValue(named name: String, in value: Any) -> Any? {
        Mirror(reflecting: value).children.first { $0.label == name }?.value
    }

    private func isoDate(_ text: String) -> Date {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: text)!
    }

    private func fixtureEvents() -> [InjectionEvent] {
        let sourceFile = "/Users/example/brainlayer/.claude/projects/-Users-example-brainlayer/session.jsonl"
        return [
            InjectionEvent(
                id: 101,
                sessionID: "session-operator-truth-1234567890",
                timestamp: "2026-07-19T09:58:00Z",
                query: "why is watcher flow marked stalled",
                chunkIDs: ["chunk-watcher-contract", "chunk-freshness-contract"],
                tokenCount: 144,
                chunks: [
                    InjectionChunk(
                        id: "chunk-watcher-contract",
                        content: "Watcher flow uses process and recent distinct-ingest evidence.",
                        summary: "Watcher truth contract",
                        source: "claude_code",
                        sourceFile: sourceFile,
                        tags: ["brainbar", "truth"],
                        contentType: "memory"
                    ),
                    InjectionChunk(
                        id: "chunk-freshness-contract",
                        content: "Snapshots become stale strictly after sixty seconds.",
                        summary: "Freshness threshold",
                        source: "mcp",
                        sourceFile: sourceFile,
                        tags: ["brainbar", "freshness"],
                        contentType: "memory"
                    ),
                ],
                claudeConversationID: "3679128a-f371-445f-82ba-b3946e2f20b6"
            ),
            InjectionEvent(
                id: 100,
                sessionID: "session-operator-truth-1234567890",
                timestamp: "2026-07-19T09:54:00Z",
                query: "why is watcher flow marked stalled",
                chunkIDs: ["chunk-replay-debt"],
                tokenCount: 76,
                chunks: [
                    InjectionChunk(
                        id: "chunk-replay-debt",
                        content: "Replay debt stays decomposed and exposes unreadable inputs.",
                        summary: "Replay-debt evidence",
                        source: "mcp",
                        sourceFile: sourceFile,
                        tags: ["brainbar"],
                        contentType: "memory"
                    )
                ]
            ),
        ]
    }

    @MainActor
    private func render<V: View>(_ view: V, name: String, directory: String) throws {
        let size = NSSize(width: 1_180, height: 820)
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: size)
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()

        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            XCTFail("Expected an AppKit bitmap for \(name)")
            return
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            XCTFail("Expected AppKit to encode \(name)")
            return
        }

        let outputDirectory = URL(fileURLWithPath: directory, isDirectory: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let outputURL = outputDirectory.appendingPathComponent(name)
        try png.write(to: outputURL)
        XCTAssertGreaterThan(png.count, 5_000, "Expected a non-empty render at \(outputURL.path)")
        XCTAssertGreaterThan(
            distinctSampledColorCount(in: bitmap),
            16,
            "Expected visible feed content in \(outputURL.path)"
        )
    }

    private func distinctSampledColorCount(in bitmap: NSBitmapImageRep) -> Int {
        guard let data = bitmap.bitmapData else { return 0 }
        let bytesPerPixel = max(bitmap.bitsPerPixel / 8, 1)
        let baseStride = max(bitmap.bytesPerRow / 32, bytesPerPixel)
        let sampleStride = baseStride - (baseStride % bytesPerPixel)
        var colors = Set<String>()
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 24) {
            let rowStart = y * bitmap.bytesPerRow
            for x in stride(from: 0, to: bitmap.bytesPerRow, by: sampleStride) {
                let offset = rowStart + x
                guard offset + 2 < bitmap.bytesPerRow * bitmap.pixelsHigh else { continue }
                colors.insert("\(data[offset])-\(data[offset + 1])-\(data[offset + 2])")
            }
        }
        return colors.count
    }

    private func brainBarSourceFile(
        _ relativePath: String,
        testFilePath: StaticString = #filePath
    ) throws -> String {
        let testsDirectory = URL(fileURLWithPath: "\(testFilePath)").deletingLastPathComponent()
        let packageRoot = testsDirectory.deletingLastPathComponent().deletingLastPathComponent()
        return try String(
            contentsOf: packageRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }
}
