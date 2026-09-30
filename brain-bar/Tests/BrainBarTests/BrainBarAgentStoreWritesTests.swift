import XCTest
@testable import BrainBar

/// #965 (Etan's ruling A, 2026-09-30): "Agent writes (24 h)" counts BrainLayer AGENT `brain_store`
/// writes only: new chunks BrainBar's brain_store wrote (`source = 'mcp'`), rolling 24 h by
/// `created_at`. Watcher-ingested transcripts, hooks, enrichment, digest and replays are not
/// counted. When it cannot be measured it says unknown with the reason, never 0.
final class BrainBarAgentStoreWritesTests: XCTestCase {
    private var db: BrainDatabase!
    private var tempDBPath: String!
    private var root: URL!
    private var restoreWatcherHealthEnvironment: (() -> Void)?

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-store-writes-\(UUID().uuidString).db"
        root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("brainbar-store-writes-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        restoreWatcherHealthEnvironment = isolateWatcherHealthEnvironment(root: root)
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        restoreWatcherHealthEnvironment?()
        for suffix in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(atPath: tempDBPath + suffix) }
        try? FileManager.default.removeItem(at: root)
        super.tearDown()
    }

    private func insert(_ id: String, source: String, createdAt: String) throws {
        _ = try db.store(content: "store-writes fixture \(id)", tags: [], importance: 5, source: source, chunkID: id)
        db.exec("UPDATE chunks SET created_at = '\(createdAt)' WHERE id = '\(id)'")
    }

    func test_only_agent_brain_store_rows_in_the_last_24_hours_count() throws {
        let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
        // Counted: brain_store (source mcp, any case/padding), inside the rolling 24 h.
        try insert("store-1h", source: "mcp", createdAt: "2026-09-30T11:00:00Z")
        try insert("store-edge", source: "mcp", createdAt: "2026-09-29T12:00:30Z")
        try insert("store-upper", source: " MCP ", createdAt: "2026-09-30 06:00:00")
        try insert("store-micro", source: "mcp", createdAt: "2026-09-30T09:15:00.123456+00:00")
        // Not counted: outside the window, or in the future.
        try insert("store-25h", source: "mcp", createdAt: "2026-09-29T11:00:00Z")
        try insert("store-future", source: "mcp", createdAt: "2026-09-30T13:00:00Z")
        // Not counted: every other writer, inside the window.
        for (id, source) in [
            ("watcher", "realtime_watcher"), ("transcript", "claude_code"), ("hook", "precompact-hook"),
            ("digest", "digest"), ("manual-cli", "manual"), ("pending-replay", "pending"),
            ("fallback-replay", "fallback-replay"), ("whatsapp", "whatsapp"),
        ] {
            try insert(id, source: source, createdAt: "2026-09-30T11:30:00Z")
        }
        // Enrichment updates an OLD store row; it is not a new write.
        try insert("enriched-old", source: "mcp", createdAt: "2026-09-20T11:00:00Z")
        db.exec("UPDATE chunks SET enriched_at = '2026-09-30T11:45:00Z', enrich_status = 'success' WHERE id = 'enriched-old'")

        XCTAssertEqual(db.brainStoreWriteCount(now: now), .measured(4))
    }

    /// #1026 review B1: the hot-currentness benchmark enqueues synthetic chunks through the same
    /// queue. It now writes `source = 'benchmark'`, and rows it wrote earlier as `mcp` carry its
    /// `benchmark_label` metadata key, which brain_store never sets. Neither is an agent write.
    func test_benchmark_writes_are_not_agent_brain_store_writes() throws {
        let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
        try insert("store", source: "mcp", createdAt: "2026-09-30T11:00:00Z")
        try insert("store-empty-meta", source: "mcp", createdAt: "2026-09-30T11:00:00Z")
        db.exec("UPDATE chunks SET metadata = '' WHERE id = 'store-empty-meta'")
        try insert("store-bad-meta", source: "mcp", createdAt: "2026-09-30T11:00:00Z")
        db.exec("UPDATE chunks SET metadata = 'not json' WHERE id = 'store-bad-meta'")
        try insert("store-other-meta", source: "mcp", createdAt: "2026-09-30T11:00:00Z")
        db.exec("UPDATE chunks SET metadata = '{\"entity_id\":\"e1\"}' WHERE id = 'store-other-meta'")
        try insert("benchmark-new", source: "benchmark", createdAt: "2026-09-30T11:00:00Z")
        db.exec("UPDATE chunks SET metadata = '{\"benchmark_label\":\"queue-drain\"}' WHERE id = 'benchmark-new'")
        try insert("benchmark-legacy", source: "mcp", createdAt: "2026-09-30T11:00:00Z")
        db.exec("UPDATE chunks SET metadata = '{\"benchmark_label\":\"queue-drain\"}' WHERE id = 'benchmark-legacy'")

        XCTAssertEqual(db.brainStoreWriteCount(now: now), .measured(4))
    }

    /// #1026 review B2: the rolling 24 h compares at the stored precision, not whole seconds.
    func test_the_24_hour_boundary_keeps_fractional_seconds() throws {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let now = formatter.date(from: "2026-09-30T12:00:00.500Z")!
        // 24 h + 0.4 s ago, in three stored formats: outside.
        try insert("old-z", source: "mcp", createdAt: "2026-09-29T12:00:00.100Z")
        try insert("old-offset", source: "mcp", createdAt: "2026-09-29T12:00:00.100000+00:00")
        try insert("old-space", source: "mcp", createdAt: "2026-09-29 12:00:00.100")
        // 24 h - 0.4 s ago: inside.
        try insert("edge-z", source: "mcp", createdAt: "2026-09-29T12:00:00.900Z")
        try insert("edge-space", source: "mcp", createdAt: "2026-09-29 12:00:00.900")
        // 0.2 s before now: inside. 0.2 s after now, in the same whole second: the future.
        try insert("just-before", source: "mcp", createdAt: "2026-09-30T12:00:00.300Z")
        try insert("just-after", source: "mcp", createdAt: "2026-09-30T12:00:00.700Z")

        XCTAssertEqual(db.brainStoreWriteCount(now: now), .measured(3))
    }

    func test_dashboard_stats_carry_the_measured_count() throws {
        try insert("store-now", source: "mcp", createdAt: ISO8601DateFormatter().string(from: Date().addingTimeInterval(-60)))
        try insert("watcher-now", source: "realtime_watcher", createdAt: ISO8601DateFormatter().string(from: Date().addingTimeInterval(-60)))
        XCTAssertEqual(try db.dashboardStats().brainStoreWrites, .measured(1))
    }

    func test_unmeasurable_is_unknown_with_a_reason_never_zero() {
        db.close()
        XCTAssertEqual(db.brainStoreWriteCount(now: Date()), .unknown("database not open"))
    }

    // MARK: the Dashboard tile

    @MainActor
    private func presentation(_ writes: BrainDatabase.BrainStoreWriteCount, observability: ObservabilityReadResult) throws -> BrainBarOnePagePresentation {
        let now = BrainBarOnePageTestFixture.now
        let stats = BrainDatabase.DashboardStats(
            chunkCount: 120, enrichedChunkCount: 120, pendingEnrichmentCount: 0, enrichmentPercent: 100,
            enrichmentRatePerMinute: 0, databaseSizeBytes: 4_096, recentActivityBuckets: [0, 0],
            recentEnrichmentBuckets: [0, 0], brainStoreWrites: writes
        )
        let collector = BrainBarDashboardFixture.makeCollector(stats: stats)
        let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: stats, now: now)
        let hero = BrainBarHeroPresentation.derive(
            flow: flow, stats: stats,
            backupTruth: BrainBarHeroBackupTruth.derive(from: observability, now: now, cadence: .known(300)),
            locale: Locale(identifier: "en_US")
        )
        return BrainBarOnePagePresentation.derive(
            snapshotFreshness: .live(ageSeconds: 0), hero: hero, observability: observability, stats: stats,
            agentActivity: BrainBarDashboardFixture.agentActivity, now: now,
            calendar: Calendar(identifier: .gregorian), locale: Locale(identifier: "en_US")
        )
    }

    /// The tile is the live DB count with its one-line definition, whatever state the
    /// observability document is in: fresh, stale or unreadable.
    @MainActor
    func test_the_tile_shows_the_live_count_and_its_definition() throws {
        for observability in [
            try BrainBarOnePageTestFixture.healthyResult(),
            try BrainBarOnePageTestFixture.staleResult(),
            ObservabilityReadResult.unreadable("observability.json missing"),
        ] {
            let tile = try presentation(.measured(42), observability: observability)
            XCTAssertEqual(tile.agentWritesCount, 42)
            XCTAssertEqual(tile.agentWritesText, "brain_store calls by agents, last 24 h")
            XCTAssertEqual(tile.agentWritesDetailText(locale: Locale(identifier: "en_US")), "42")
        }
        XCTAssertEqual(try presentation(.measured(0), observability: .unreadable("x")).agentWritesCount, 0, "a measured zero is a real zero")
    }

    @MainActor
    func test_an_unmeasurable_count_is_unknown_with_its_reason() throws {
        let tile = try presentation(.unknown("database not open"), observability: try BrainBarOnePageTestFixture.healthyResult())
        XCTAssertNil(tile.agentWritesCount, "never a silent 0")
        XCTAssertEqual(tile.agentWritesText, "brain_store writes (24 h) unknown — database not open")
        XCTAssertEqual(tile.agentWritesDetailText(locale: Locale(identifier: "en_US")), "unknown — database not open")
    }
}
