// StatsCollectorTests.swift
//
// Covers the collector-side wiring for the shared Live/3h/24h timeframe selector
// (dashboard redesign). `selectTimeframe(windowMinutes:isLive:)` must:
//  - clear `windowedBuckets` for the live lens (charts fall back to live stats),
//  - publish REAL windowed buckets for a wider lens (3h/24h),
// proving the selector re-fetches genuine DB history rather than relabeling.

import XCTest
import SQLite3
@testable import BrainBar

private final class SequencedWindowBucketsProvider: @unchecked Sendable {
    private let lock = NSLock()
    private let responses: [BrainDatabase.PipelineWindowBuckets]
    private var nextResponseIndex = 0

    init(responses: [BrainDatabase.PipelineWindowBuckets]) {
        self.responses = responses
    }

    var callCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return nextResponseIndex
    }

    func fetch(windowMinutes: Int, bucketCount: Int) throws -> BrainDatabase.PipelineWindowBuckets {
        lock.lock()
        defer { lock.unlock() }
        let index = min(nextResponseIndex, responses.count - 1)
        nextResponseIndex += 1
        return responses[index]
    }
}

private struct WindowBucketsProviderFailure: Error {}

private final class SucceedOnceWindowBucketsProvider: @unchecked Sendable {
    private let lock = NSLock()
    private let response: BrainDatabase.PipelineWindowBuckets
    private var calls = 0

    init(response: BrainDatabase.PipelineWindowBuckets) {
        self.response = response
    }

    func fetch(windowMinutes: Int, bucketCount: Int) throws -> BrainDatabase.PipelineWindowBuckets {
        lock.lock()
        defer { lock.unlock() }
        calls += 1
        guard calls == 1 else { throw WindowBucketsProviderFailure() }
        return response
    }
}

private final class SequencedBlockingWatcherProbe: WatcherProcessProbing, @unchecked Sendable {
    private let lock = NSLock()
    private let blockedSampleStarted = DispatchSemaphore(value: 0)
    private let releaseBlockedSampleSemaphore = DispatchSemaphore(value: 0)
    private var calls = 0

    var callCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }

    func sample() -> WatcherProcessProbeResult {
        lock.lock()
        let call = calls
        calls += 1
        lock.unlock()

        switch call {
        case 0:
            return .running(pid: 1001)
        case 1:
            blockedSampleStarted.signal()
            releaseBlockedSampleSemaphore.wait()
            return .failure("stale standalone probe")
        default:
            return .running(pid: 4242)
        }
    }

    func waitForBlockedSample(timeout: DispatchTime) -> Bool {
        blockedSampleStarted.wait(timeout: timeout) == .success
    }

    func releaseBlockedSample() {
        releaseBlockedSampleSemaphore.signal()
    }
}

private final class OlderFullRefreshProbe: WatcherProcessProbing, @unchecked Sendable {
    private let lock = NSLock()
    private let firstSampleStarted = DispatchSemaphore(value: 0)
    private let releaseFirstSampleSemaphore = DispatchSemaphore(value: 0)
    private var calls = 0

    func sample() -> WatcherProcessProbeResult {
        lock.lock()
        let call = calls
        calls += 1
        lock.unlock()

        if call == 0 {
            firstSampleStarted.signal()
            releaseFirstSampleSemaphore.wait()
            return .running(pid: 1111)
        }
        return .running(pid: 2222)
    }

    func waitForFirstSample(timeout: DispatchTime) -> Bool {
        firstSampleStarted.wait(timeout: timeout) == .success
    }

    func releaseFirstSample() {
        releaseFirstSampleSemaphore.signal()
    }
}

@MainActor
final class StatsCollectorTests: XCTestCase {
    private struct WindowFetchFailure: Error {}

    private struct BlockingWatcherProbe: WatcherProcessProbing {
        let delay: TimeInterval

        func sample() -> WatcherProcessProbeResult {
            Thread.sleep(forTimeInterval: delay)
            return .running(pid: 4242)
        }
    }

    private var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-statscollector-\(UUID().uuidString).db"
        // Constructor opens + ensures schema.
        let db = BrainDatabase(path: tempDBPath)
        db.close()
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    private static let utcISO8601: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        return f
    }()

    private func insertWrite(id: String, source: String, minutesAgo: Double) throws {
        let createdText = Self.utcISO8601.string(from: Date().addingTimeInterval(-minutesAgo * 60))
        var db: OpaquePointer?
        let rc = sqlite3_open_v2(tempDBPath, &db, SQLITE_OPEN_READWRITE, nil)
        guard rc == SQLITE_OK, let db else { throw NSError(domain: "StatsCollectorTests", code: Int(rc)) }
        defer { sqlite3_close(db) }
        let sql = """
            INSERT INTO chunks (id, content, source, created_at, status)
            VALUES ('\(id)', 'probe \(id)', '\(source)', '\(createdText)', 'active');
        """
        let execRC = sqlite3_exec(db, sql, nil, nil, nil)
        guard execRC == SQLITE_OK else { throw NSError(domain: "StatsCollectorTests", code: Int(execRC)) }
    }

    private func insertWatcherLiveness(chunkID: String, minutesAgo: Double) throws {
        let ingestedAt = Int(Date().addingTimeInterval(-minutesAgo * 60).timeIntervalSince1970)
        var db: OpaquePointer?
        let rc = sqlite3_open_v2(tempDBPath, &db, SQLITE_OPEN_READWRITE, nil)
        guard rc == SQLITE_OK, let db else { throw NSError(domain: "StatsCollectorTests", code: Int(rc)) }
        defer { sqlite3_close(db) }
        let sql = """
            CREATE TABLE IF NOT EXISTS watcher_liveness_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                ingested_at INTEGER NOT NULL
            );
            INSERT INTO watcher_liveness_events (chunk_id, ingested_at)
            VALUES ('\(chunkID)', \(ingestedAt));
        """
        let execRC = sqlite3_exec(db, sql, nil, nil, nil)
        guard execRC == SQLITE_OK else { throw NSError(domain: "StatsCollectorTests", code: Int(execRC)) }
    }

    func testSelectTimeframeLiveClearsWindowedBuckets() async throws {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        // Widen first, then go back to live — live must clear the windowed fetch.
        try insertWrite(id: "agent-old", source: "mcp", minutesAgo: 120)
        collector.selectTimeframe(windowMinutes: 180, isLive: false)
        let deadline = Date().addingTimeInterval(2.0)
        while collector.windowedBuckets == nil && Date() < deadline {
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTAssertNotNil(collector.windowedBuckets, "wider lens should fetch windowed buckets")

        collector.selectTimeframe(windowMinutes: 30, isLive: true)
        XCTAssertNil(collector.windowedBuckets, "live lens must clear windowed buckets")
        XCTAssertNil(collector.windowedBucketsWindowMinutes)
    }

    func testSelectTimeframeWiderPublishesRealHistoricalBuckets() async throws {
        // Inside 1h (live) plus older rows only a wider window can see.
        try insertWrite(id: "agent-live", source: "mcp", minutesAgo: 5)
        try insertWrite(id: "agent-old-1", source: "mcp", minutesAgo: 90)
        try insertWrite(id: "agent-old-2", source: "manual", minutesAgo: 300)
        try insertWrite(id: "watcher-old", source: "realtime_watcher", minutesAgo: 200)
        try insertWatcherLiveness(chunkID: "watcher-old", minutesAgo: 200)

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        collector.selectTimeframe(windowMinutes: 1_440, isLive: false)
        let deadline = Date().addingTimeInterval(2.0)
        while collector.windowedBuckets == nil && Date() < deadline {
            try await Task.sleep(for: .milliseconds(25))
        }

        let buckets = try XCTUnwrap(collector.windowedBuckets, "24h lens must publish real buckets")
        XCTAssertEqual(collector.windowedBucketsWindowMinutes, 1_440)
        XCTAssertEqual(buckets.agentTotal, 3, "24h agent window sees live + 2 older agent writes")
        XCTAssertEqual(buckets.watcherTotal, 1, "24h watcher window sees the older watcher write")
        XCTAssertFalse(collector.isWindowedBucketsLoading)
    }

    func testSuccessfulDashboardRefreshRefetchesSelectedWiderWindow() async throws {
        let initialBuckets = BrainDatabase.PipelineWindowBuckets(
            activityWindowMinutes: 180,
            bucketCount: 1,
            allWriteBuckets: [1],
            agentWriteBuckets: [1],
            watcherWriteBuckets: [0],
            enrichmentBuckets: [0],
            watcherFlowReadability: .readable
        )
        let refreshedBuckets = BrainDatabase.PipelineWindowBuckets(
            activityWindowMinutes: 180,
            bucketCount: 1,
            allWriteBuckets: [2],
            agentWriteBuckets: [2],
            watcherWriteBuckets: [0],
            enrichmentBuckets: [0],
            watcherFlowReadability: .readable
        )
        let provider = SequencedWindowBucketsProvider(responses: [initialBuckets, refreshedBuckets])
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            windowedBucketsProvider: provider.fetch,
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        collector.selectTimeframe(windowMinutes: 180, isLive: false)
        let initialDeadline = Date().addingTimeInterval(2)
        while collector.windowedBuckets != initialBuckets && Date() < initialDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.windowedBuckets, initialBuckets)

        collector.refresh(force: true)
        let refreshDeadline = Date().addingTimeInterval(3)
        while (collector.isRefreshing || provider.callCount < 2) && Date() < refreshDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(provider.callCount, 2, "A successful dashboard refresh must refetch the selected 3h window.")
        XCTAssertEqual(collector.windowedBuckets, refreshedBuckets)
        XCTAssertEqual(collector.windowedBucketsWindowMinutes, 180)
    }

    func testWindowFetchFailurePreservesLastGoodBucketsAndPublishesTruthfulError() async throws {
        let threeHourBuckets = BrainDatabase.PipelineWindowBuckets(
            activityWindowMinutes: 180,
            bucketCount: 1,
            allWriteBuckets: [3],
            agentWriteBuckets: [2],
            watcherWriteBuckets: [1],
            enrichmentBuckets: [1],
            watcherFlowReadability: .readable
        )
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            windowedBucketsProvider: { windowMinutes, _ in
                guard windowMinutes == 180 else { throw WindowFetchFailure() }
                return threeHourBuckets
            },
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        collector.selectTimeframe(windowMinutes: 180, isLive: false)
        let successDeadline = Date().addingTimeInterval(2)
        while collector.isWindowedBucketsLoading && Date() < successDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.windowedBuckets, threeHourBuckets)
        XCTAssertNil(collector.windowedBucketsError)

        collector.selectTimeframe(windowMinutes: 1_440, isLive: false)
        let failureDeadline = Date().addingTimeInterval(2)
        while collector.isWindowedBucketsLoading && Date() < failureDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(collector.windowedBuckets, threeHourBuckets, "A failed wider fetch must retain last-good evidence.")
        XCTAssertEqual(collector.windowedBucketsWindowMinutes, 180)
        XCTAssertEqual(collector.windowedBucketsError, "Could not load Last 24h; showing Last 1h.")
        XCTAssertEqual(
            PipelineTimeframe.truthfulDisplay(selected: .day, loadedWindowMinutes: collector.windowedBucketsWindowMinutes),
            .live,
            "Live buckets must never be relabeled as the failed 24h selection."
        )
    }

    func testWindowRefreshFailureDescribesRetainedMatchingLastGoodBuckets() async throws {
        let threeHourBuckets = BrainDatabase.PipelineWindowBuckets(
            activityWindowMinutes: 180,
            bucketCount: 1,
            allWriteBuckets: [3],
            agentWriteBuckets: [2],
            watcherWriteBuckets: [1],
            enrichmentBuckets: [1],
            watcherFlowReadability: .readable
        )
        let provider = SucceedOnceWindowBucketsProvider(response: threeHourBuckets)
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            windowedBucketsProvider: provider.fetch,
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        collector.selectTimeframe(windowMinutes: 180, isLive: false)
        let successDeadline = Date().addingTimeInterval(2)
        while collector.isWindowedBucketsLoading && Date() < successDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.windowedBuckets, threeHourBuckets)

        collector.selectTimeframe(windowMinutes: 180, isLive: false)
        let failureDeadline = Date().addingTimeInterval(2)
        while collector.isWindowedBucketsLoading && Date() < failureDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(collector.windowedBuckets, threeHourBuckets, "Section 4 requires retaining last-good values on fetch error.")
        XCTAssertEqual(collector.windowedBucketsWindowMinutes, 180)
        XCTAssertEqual(collector.windowedBucketsError, "Could not refresh Last 3h; showing previous Last 3h.")
        XCTAssertEqual(
            PipelineTimeframe.truthfulDisplay(selected: .threeHour, loadedWindowMinutes: collector.windowedBucketsWindowMinutes),
            .threeHour
        )
    }

    func testReturningToLiveDiscardsLateWindowFetch() async throws {
        let delayedBuckets = BrainDatabase.PipelineWindowBuckets(
            activityWindowMinutes: 1_440,
            bucketCount: 1,
            allWriteBuckets: [1],
            agentWriteBuckets: [1],
            watcherWriteBuckets: [0],
            enrichmentBuckets: [0],
            watcherFlowReadability: .readable
        )
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            windowedBucketsProvider: { _, _ in
                Thread.sleep(forTimeInterval: 0.1)
                return delayedBuckets
            },
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        collector.selectTimeframe(windowMinutes: 1_440, isLive: false)
        collector.selectTimeframe(windowMinutes: 60, isLive: true)
        try await Task.sleep(for: .milliseconds(200))

        XCTAssertNil(collector.windowedBuckets)
        XCTAssertNil(collector.windowedBucketsWindowMinutes)
        XCTAssertNil(collector.windowedBucketsError)
        XCTAssertFalse(collector.isWindowedBucketsLoading)
    }

    func testWatcherProbeDoesNotBlockMainActorRefresh() async throws {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            watcherProcessProbe: BlockingWatcherProbe(delay: 0.6),
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        let startedAt = Date()
        collector.refresh(force: true)
        let elapsed = Date().timeIntervalSince(startedAt)

        XCTAssertLessThan(elapsed, 0.3, "Watcher sampling must leave the main actor before it can block dashboard refresh.")
        XCTAssertTrue(collector.isRefreshing)

        let deadline = Date().addingTimeInterval(3)
        while collector.isRefreshing && Date() < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.stats.watcherProcessProbeResult, .running(pid: 4242))
    }

    func testFullRefreshInvalidatesOlderStandaloneWatcherProbe() async throws {
        let probe = SequencedBlockingWatcherProbe()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            watcherProcessProbe: probe,
            statsRefreshCoalesceInterval: 5,
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer {
            probe.releaseBlockedSample()
            collector.stop()
        }

        collector.refresh(force: false)
        let initialDeadline = Date().addingTimeInterval(3)
        while collector.isRefreshing && Date() < initialDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.stats.watcherProcessProbeResult, .running(pid: 1001))

        collector.refresh(force: false)
        XCTAssertTrue(
            probe.waitForBlockedSample(timeout: .now() + 2),
            "The coalesced refresh must start the standalone probe used to reproduce the race."
        )

        collector.refresh(force: true)
        let fullRefreshDeadline = Date().addingTimeInterval(3)
        while (collector.isRefreshing || probe.callCount < 3) && Date() < fullRefreshDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.stats.watcherProcessProbeResult, .running(pid: 4242))

        probe.releaseBlockedSample()
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(
            collector.stats.watcherProcessProbeResult,
            .running(pid: 4242),
            "An older standalone probe must not overwrite the full refresh's newer watcher truth."
        )
    }

    func testFullRefreshDoesNotOverwriteNewerStandaloneWatcherProbe() async throws {
        let probe = OlderFullRefreshProbe()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            watcherProcessProbe: probe,
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer {
            probe.releaseFirstSample()
            collector.stop()
        }

        collector.refresh(force: true)
        XCTAssertTrue(
            probe.waitForFirstSample(timeout: .now() + 2),
            "The full refresh must hold its older process sample in flight."
        )

        collector.refresh(force: false)
        let standaloneDeadline = Date().addingTimeInterval(2)
        while collector.stats.watcherProcessProbeResult != .running(pid: 2222) && Date() < standaloneDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(collector.stats.watcherProcessProbeResult, .running(pid: 2222))

        probe.releaseFirstSample()
        let fullRefreshDeadline = Date().addingTimeInterval(3)
        while collector.isRefreshing && Date() < fullRefreshDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(
            collector.stats.watcherProcessProbeResult,
            .running(pid: 2222),
            "The older process sample embedded in a full refresh must not overwrite a newer standalone probe."
        )
    }

    func testLaunchctlCommandRunnerTimesOutHungProbe() {
        let startedAt = Date()
        let result = LaunchctlWatcherProcessProbe.run(
            ["/bin/sleep", "1"],
            timeout: 0.05
        )

        XCTAssertEqual(result.terminationStatus, 124)
        XCTAssertTrue(result.output.localizedCaseInsensitiveContains("timed out"))
        XCTAssertLessThan(Date().timeIntervalSince(startedAt), 0.5)
    }
}
