// BrainDatabase.swift — SQLite database layer for BrainBar.
//
// Wraps SQLite3 directly. BrainBar now keeps its pub/sub metadata in the main
// BrainLayer database so agent state and chunk writes share one durable store.

import Darwin
import Foundation
import SQLite3

final class BrainDatabase: @unchecked Sendable {
    struct OpenConfiguration: Sendable, Equatable {
        let busyTimeoutMillis: Int32
        let readOnly: Bool

        init(busyTimeoutMillis: Int32 = 30_000, readOnly: Bool = false) {
            self.busyTimeoutMillis = max(1, busyTimeoutMillis)
            self.readOnly = readOnly
        }
    }

    static let dashboardDidChangeNotification = "com.brainlayer.brainbar.database-changed"
    private static let previewExpression = """
        trim(substr(replace(replace(replace(coalesce(nullif(summary, ''), content), char(10), ' '), char(13), ' '), char(9), ' '), 1, 220))
    """
    private static let ftsColumns = "content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED"
    private static let ftsOptions = "prefix='2 3 4', tokenize='unicode61 remove_diacritics 2'"
    private static let trigramFTSOptions = "tokenize='trigram'"
    private static let synchronousTrigramBackfillChunkLimit = 10_000
    static let maximumTrigramMaintenanceBatchSize = 10_000
    private static let defaultPendingStoreMaxLines = 10_000
    private static let pendingStoreMaxLinesEnv = "BRAINBAR_PENDING_STORES_MAX_LINES"
    private static let agentWriteSourceWhereClause = "COALESCE(LOWER(TRIM(source)), '') IN ('mcp', 'manual', 'digest', 'precompact-hook', 'brain_store')"
    private static let watcherWriteSourceWhereClause = "COALESCE(LOWER(TRIM(source)), '') IN ('realtime_watcher', 'realtime')"
    private static let lexicalDefenseReplacements: [String: [String]] = [
        "hershkovitz": ["Hershkovits"],
        "hershkovits": ["Hershkovitz"]
    ]

    enum TrigramStartupRepairDecision: Equatable {
        case noRepairNeeded
        case rebuildSynchronously
        case skipBackfill
    }

    struct TrigramMaintenanceProgress: Sendable, Equatable {
        enum State: String, Sendable, Equatable {
            case running
            case done
            case cancelled
            case failed
        }

        let state: State
        let processed: Int
        let total: Int
        let etaSeconds: Double?

        var metadata: [String: Any] {
            var payload: [String: Any] = [
                "state": state.rawValue,
                "processed": processed,
                "total": total,
            ]
            if let etaSeconds {
                payload["eta_seconds"] = etaSeconds
            }
            return payload
        }
    }

    struct DashboardStats: Sendable, Equatable {
        struct WatcherHealth: Sendable, Equatable {
            let alerting: Bool
            let filesTracked: Int
            let maxOffsetLagBytes: Int
            let activeEntriesPerMinute: Double
            let realtimeInsertsPerMinute: Double

            var summaryText: String {
                if alerting {
                    if maxOffsetLagBytes >= 1_048_576 {
                        return "lag \(Int((Double(maxOffsetLagBytes) / 1_048_576.0).rounded())) MB"
                    }
                    return "coverage alert"
                }
                if filesTracked > 0 {
                    return "\(filesTracked) files"
                }
                return "idle"
            }
        }

        let chunkCount: Int
        let enrichedChunkCount: Int
        let pendingEnrichmentCount: Int
        let enrichmentPercent: Double
        let enrichmentRatePerMinute: Double
        let databaseSizeBytes: Int64
        let recentActivityBuckets: [Int]
        let recentAgentWriteBuckets: [Int]
        let recentWatcherWriteBuckets: [Int]
        let recentEnrichmentBuckets: [Int]
        let recentWriteFiveMinuteCount: Int
        let recentEnrichmentFiveMinuteCount: Int
        let activityWindowMinutes: Int
        let bucketCount: Int
        let liveWindowMinutes: Int
        let lastWriteAt: Date?
        let lastEnrichedAt: Date?
        let signalEligibleChunkCount: Int
        let vectorIndexedChunkCount: Int
        let ftsIndexedChunkCount: Int
        let trigramIndexedChunkCount: Int
        let pendingStoreQueueDepth: Int
        let pendingStoreOldestQueuedAt: Date?
        let pendingStoreFlushRatePerMinute: Double
        let watcherHealth: WatcherHealth?

        init(
            chunkCount: Int,
            enrichedChunkCount: Int,
            pendingEnrichmentCount: Int,
            enrichmentPercent: Double,
            enrichmentRatePerMinute: Double,
            databaseSizeBytes: Int64,
            recentActivityBuckets: [Int],
            recentAgentWriteBuckets: [Int]? = nil,
            recentWatcherWriteBuckets: [Int]? = nil,
            recentEnrichmentBuckets: [Int],
            recentWriteFiveMinuteCount: Int = 0,
            recentEnrichmentFiveMinuteCount: Int = 0,
            activityWindowMinutes: Int = 30,
            bucketCount: Int = 12,
            liveWindowMinutes: Int = 1,
            lastWriteAt: Date? = nil,
            lastEnrichedAt: Date? = nil,
            signalEligibleChunkCount: Int? = nil,
            vectorIndexedChunkCount: Int = 0,
            ftsIndexedChunkCount: Int = 0,
            trigramIndexedChunkCount: Int = 0,
            pendingStoreQueueDepth: Int = 0,
            pendingStoreOldestQueuedAt: Date? = nil,
            pendingStoreFlushRatePerMinute: Double = 0,
            watcherHealth: WatcherHealth? = nil
        ) {
            self.chunkCount = chunkCount
            self.enrichedChunkCount = enrichedChunkCount
            self.pendingEnrichmentCount = pendingEnrichmentCount
            self.enrichmentPercent = enrichmentPercent
            self.enrichmentRatePerMinute = enrichmentRatePerMinute
            self.databaseSizeBytes = databaseSizeBytes
            self.recentActivityBuckets = recentActivityBuckets
            self.recentAgentWriteBuckets = recentAgentWriteBuckets ?? recentActivityBuckets
            self.recentWatcherWriteBuckets = recentWatcherWriteBuckets ?? Array(repeating: 0, count: recentActivityBuckets.count)
            self.recentEnrichmentBuckets = recentEnrichmentBuckets
            self.recentWriteFiveMinuteCount = recentWriteFiveMinuteCount
            self.recentEnrichmentFiveMinuteCount = recentEnrichmentFiveMinuteCount
            self.activityWindowMinutes = activityWindowMinutes
            self.bucketCount = bucketCount
            self.liveWindowMinutes = liveWindowMinutes
            self.lastWriteAt = lastWriteAt
            self.lastEnrichedAt = lastEnrichedAt
            self.signalEligibleChunkCount = signalEligibleChunkCount ?? chunkCount
            self.vectorIndexedChunkCount = vectorIndexedChunkCount
            self.ftsIndexedChunkCount = ftsIndexedChunkCount
            self.trigramIndexedChunkCount = trigramIndexedChunkCount
            self.pendingStoreQueueDepth = pendingStoreQueueDepth
            self.pendingStoreOldestQueuedAt = pendingStoreOldestQueuedAt
            self.pendingStoreFlushRatePerMinute = pendingStoreFlushRatePerMinute
            self.watcherHealth = watcherHealth
        }

        var recentWriteCount: Int {
            recentActivityBuckets.reduce(0, +)
        }

        var recentAgentWriteCount: Int {
            recentAgentWriteBuckets.reduce(0, +)
        }

        var recentWatcherWriteCount: Int {
            recentWatcherWriteBuckets.reduce(0, +)
        }

        var recentEnrichmentCount: Int {
            recentEnrichmentBuckets.reduce(0, +)
        }

        var writeRatePerMinute: Double {
            guard activityWindowMinutes > 0 else { return 0 }
            return Double(recentWriteCount) / Double(activityWindowMinutes)
        }

        var recentWriteRatePerHour: Double {
            writeRatePerMinute * 60
        }

        var recentEnrichmentRatePerHour: Double {
            guard activityWindowMinutes > 0 else { return 0 }
            return (Double(recentEnrichmentCount) / Double(activityWindowMinutes)) * 60
        }

        var vectorNetDrainRatePerHour: Double {
            // Vector rows do not currently carry per-vector timestamps. Until
            // they do, use the real dashboard window: recent enrichment
            // completions minus recent writes, which is the net backlog drain
            // signal available in DashboardStats.
            max(recentEnrichmentRatePerHour - recentWriteRatePerHour, 0)
        }

        var vectorBacklogETAHours: Double? {
            guard vectorBacklogCount > 0, vectorNetDrainRatePerHour > 0 else { return nil }
            return Double(vectorBacklogCount) / vectorNetDrainRatePerHour
        }

        func eventIsLive(_ date: Date?, now: Date = Date()) -> Bool {
            guard let date else { return false }
            let liveWindowSeconds = max(Double(liveWindowMinutes) * 60, 0)
            return now.timeIntervalSince(date) < liveWindowSeconds
        }

        func hasRecentWriteActivity(now: Date = Date()) -> Bool {
            eventIsLive(lastWriteAt, now: now) || recentWriteCount > 0
        }

        func hasRecentEnrichmentActivity(now: Date = Date()) -> Bool {
            eventIsLive(lastEnrichedAt, now: now) || recentEnrichmentCount > 0
        }

        var vectorBacklogCount: Int {
            max(signalEligibleChunkCount - vectorIndexedChunkCount, 0)
        }

        var ftsBacklogCount: Int {
            max(signalEligibleChunkCount - ftsIndexedChunkCount, 0)
        }

        var trigramBacklogCount: Int {
            max(signalEligibleChunkCount - trigramIndexedChunkCount, 0)
        }

        var vectorCoveragePercent: Double {
            coveragePercent(indexedCount: vectorIndexedChunkCount)
        }

        var ftsCoveragePercent: Double {
            coveragePercent(indexedCount: ftsIndexedChunkCount)
        }

        var trigramCoveragePercent: Double {
            coveragePercent(indexedCount: trigramIndexedChunkCount)
        }

        private func coveragePercent(indexedCount: Int) -> Double {
            guard signalEligibleChunkCount > 0 else { return 100 }
            return (Double(indexedCount) / Double(signalEligibleChunkCount)) * 100
        }

        /// Returns a copy whose per-series buckets + activity window are replaced
        /// with REAL windowed data (the shared Live/3h/24h selector). Everything
        /// else (counts, coverage, queue, last-event) is preserved so the lanes
        /// re-derive their volume/rate text from the wider window while the rest
        /// of the dashboard stays consistent.
        func withWindowedPipelineBuckets(_ buckets: PipelineWindowBuckets) -> DashboardStats {
            DashboardStats(
                chunkCount: chunkCount,
                enrichedChunkCount: enrichedChunkCount,
                pendingEnrichmentCount: pendingEnrichmentCount,
                enrichmentPercent: enrichmentPercent,
                enrichmentRatePerMinute: enrichmentRatePerMinute,
                databaseSizeBytes: databaseSizeBytes,
                recentActivityBuckets: zip(
                    buckets.agentWriteBuckets,
                    buckets.watcherWriteBuckets
                ).map(+),
                recentAgentWriteBuckets: buckets.agentWriteBuckets,
                recentWatcherWriteBuckets: buckets.watcherWriteBuckets,
                recentEnrichmentBuckets: buckets.enrichmentBuckets,
                recentWriteFiveMinuteCount: recentWriteFiveMinuteCount,
                recentEnrichmentFiveMinuteCount: recentEnrichmentFiveMinuteCount,
                activityWindowMinutes: buckets.activityWindowMinutes,
                bucketCount: buckets.bucketCount,
                liveWindowMinutes: liveWindowMinutes,
                lastWriteAt: lastWriteAt,
                lastEnrichedAt: lastEnrichedAt,
                signalEligibleChunkCount: signalEligibleChunkCount,
                vectorIndexedChunkCount: vectorIndexedChunkCount,
                ftsIndexedChunkCount: ftsIndexedChunkCount,
                trigramIndexedChunkCount: trigramIndexedChunkCount,
                pendingStoreQueueDepth: pendingStoreQueueDepth,
                pendingStoreOldestQueuedAt: pendingStoreOldestQueuedAt,
                pendingStoreFlushRatePerMinute: pendingStoreFlushRatePerMinute,
                watcherHealth: watcherHealth
            )
        }

        func withPendingStoreFlushRate(_ rate: Double) -> DashboardStats {
            DashboardStats(
                chunkCount: chunkCount,
                enrichedChunkCount: enrichedChunkCount,
                pendingEnrichmentCount: pendingEnrichmentCount,
                enrichmentPercent: enrichmentPercent,
                enrichmentRatePerMinute: enrichmentRatePerMinute,
                databaseSizeBytes: databaseSizeBytes,
                recentActivityBuckets: recentActivityBuckets,
                recentAgentWriteBuckets: recentAgentWriteBuckets,
                recentWatcherWriteBuckets: recentWatcherWriteBuckets,
                recentEnrichmentBuckets: recentEnrichmentBuckets,
                recentWriteFiveMinuteCount: recentWriteFiveMinuteCount,
                recentEnrichmentFiveMinuteCount: recentEnrichmentFiveMinuteCount,
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                liveWindowMinutes: liveWindowMinutes,
                lastWriteAt: lastWriteAt,
                lastEnrichedAt: lastEnrichedAt,
                signalEligibleChunkCount: signalEligibleChunkCount,
                vectorIndexedChunkCount: vectorIndexedChunkCount,
                ftsIndexedChunkCount: ftsIndexedChunkCount,
                trigramIndexedChunkCount: trigramIndexedChunkCount,
                pendingStoreQueueDepth: pendingStoreQueueDepth,
                pendingStoreOldestQueuedAt: pendingStoreOldestQueuedAt,
                pendingStoreFlushRatePerMinute: rate,
                watcherHealth: watcherHealth
            )
        }
    }

    struct PendingStoreQueueSnapshot: Sendable, Equatable {
        let depth: Int
        let oldestQueuedAt: Date?
    }

    /// The three pipeline series (agent stores / JSONL watcher / enrichment)
    /// bucketed over an ARBITRARY window — the data primitive behind the shared
    /// Live(30m)/3h/24h selector. Unlike `DashboardStats` (which is pinned to the
    /// resting window), this is a lightweight windowed re-fetch the dashboard
    /// asks for when the timeframe changes, so 3h/24h show REAL historical DB
    /// data rather than relabeling the live buckets.
    struct PipelineWindowBuckets: Sendable, Equatable {
        let activityWindowMinutes: Int
        let bucketCount: Int
        let agentWriteBuckets: [Int]
        let watcherWriteBuckets: [Int]
        let enrichmentBuckets: [Int]

        var agentTotal: Int { agentWriteBuckets.reduce(0, +) }
        var watcherTotal: Int { watcherWriteBuckets.reduce(0, +) }
        var enrichmentTotal: Int { enrichmentBuckets.reduce(0, +) }
    }

    struct SubscriberRecord: Sendable {
        let agentID: String
        let generation: Int
        let connectionState: String
        let tags: [String]
        let lastDeliveredSeq: Int64
        let lastAckedSeq: Int64
        let lastConnectedAt: String?
        let disconnectedAt: String?
    }

    struct StoredChunk: Sendable, Equatable {
        let chunkID: String
        let rowID: Int64
    }

    struct StoredChunkDetails: Sendable, Equatable {
        let content: String
        let tags: [String]
        let importance: Int
    }

    struct FlushedPendingStore: Sendable {
        let storedChunk: StoredChunk
        let content: String
        let tags: [String]
        let importance: Int
    }

    enum StoreWriteOutcome: Sendable, Equatable {
        case stored(StoredChunk)
        case queued(queueID: String, queuedAt: String, chunkID: String)
    }

    struct PendingStoreItem: Codable, Sendable {
        let content: String
        let tags: [String]
        let importance: Int
        let source: String
        let project: String?
        let queueID: String?
        let queuedAt: String?
        let createdAt: String?
        let chunkID: String?

        enum CodingKeys: String, CodingKey {
            case content
            case tags
            case importance
            case source
            case project
            case queueID = "queue_id"
            case queuedAt = "queued_at"
            case createdAt = "created_at"
            case chunkID = "chunk_id"
        }

        init(
            content: String,
            tags: [String],
            importance: Int,
            source: String,
            project: String? = nil,
            queueID: String? = nil,
            queuedAt: String? = nil,
            createdAt: String? = nil,
            chunkID: String? = nil
        ) {
            self.content = content
            self.tags = tags
            self.importance = importance
            self.source = source
            self.project = project
            self.queueID = queueID
            self.queuedAt = queuedAt
            self.createdAt = createdAt
            self.chunkID = chunkID
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            content = try container.decode(String.self, forKey: .content)
            tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? []
            importance = try container.decodeIfPresent(Int.self, forKey: .importance) ?? 5
            source = try container.decodeIfPresent(String.self, forKey: .source) ?? "mcp"
            project = try container.decodeIfPresent(String.self, forKey: .project)
            queueID = try container.decodeIfPresent(String.self, forKey: .queueID)
            queuedAt = try container.decodeIfPresent(String.self, forKey: .queuedAt)
            createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt)
            chunkID = try container.decodeIfPresent(String.self, forKey: .chunkID)
        }
    }

    struct ConversationChunk: Sendable, Equatable, Identifiable {
        let chunkID: String
        let content: String
        let contentType: String
        let sender: String
        let importance: Double
        let createdAt: String
        let summary: String
        let isTarget: Bool

        var id: String { chunkID }

        init(
            chunkID: String,
            content: String,
            contentType: String,
            sender: String = "",
            importance: Double,
            createdAt: String,
            summary: String,
            isTarget: Bool
        ) {
            self.chunkID = chunkID
            self.content = content
            self.contentType = contentType
            self.sender = sender
            self.importance = importance
            self.createdAt = createdAt
            self.summary = summary
            self.isTarget = isTarget
        }
    }

    struct ExpandedConversation: Sendable, Equatable, Identifiable {
        let target: ConversationChunk
        let entries: [ConversationChunk]

        var id: String { target.chunkID }
    }

    private var db: OpaquePointer?
    private let path: String
    private let openConfiguration: OpenConfiguration
    private let transactionLock = NSRecursiveLock()
    private var explicitTransactionIsOpen = false
    var failNextStoreAfterInsertForTesting = false
    var failNextStoreWithBusyForTesting = false
    private static let pendingStoreFileLock = NSLock()
    private(set) var isOpen = false
    private(set) var lastOpenError: Error?
    private static let maximumConversationContextPerSide = 40
    private static let maximumConversationContentCharacters = 4_000

    init(path: String, openConfiguration: OpenConfiguration = OpenConfiguration()) {
        self.path = path
        self.openConfiguration = openConfiguration
        openAndConfigure()
    }

    private func openAndConfigure() {
        do {
            db = try openConnection(path: path)
            NSLog("[BrainBar] Connection opened, configuring...")
            try configureConnection(db)
            NSLog("[BrainBar] Connection configured, checking schema...")
            if openConfiguration.readOnly {
                NSLog("[BrainBar] Read-only connection - skipping schema migration checks")
                lastOpenError = nil
                isOpen = true
                return
            }
            // AIDEV-NOTE: Skip ensureSchema if chunks table already exists (Python creates all tables).
            // CREATE TABLE IF NOT EXISTS still acquires a RESERVED lock which blocks on watch agent.
            if (try? tableExists("chunks")) == true {
                NSLog("[BrainBar] Schema already exists — skipping ensureSchema")
                try ensureMigrations()
            } else {
                try ensureSchema()
                NSLog("[BrainBar] Schema created")
            }
            lastOpenError = nil
            isOpen = true
            NSLog("[BrainBar] Database ready")
        } catch {
            if let db {
                sqlite3_close(db)
                self.db = nil
            }
            isOpen = false
            lastOpenError = error
            NSLog("[BrainBar] Failed to open/configure database at %@: %@", path, String(describing: error))
        }
    }

    func reopenIfNeeded() {
        guard !isOpen else { return }
        openAndConfigure()
    }

    // tableExists defined at line ~237 (existing method)

    private func ensureSchema() throws {
        if (try? tableExists("chunks")) != true {
            try execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    source_file TEXT NOT NULL DEFAULT 'brainbar',
                    project TEXT,
                    content_type TEXT DEFAULT 'assistant_text',
                    value_type TEXT,
                    char_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'claude_code',
                    sender TEXT,
                    language TEXT,
                    conversation_id TEXT,
                    position INTEGER,
                    context_summary TEXT,
                    tags TEXT DEFAULT '[]',
                    tag_confidence REAL,
                    summary TEXT,
                    preview_text TEXT,
                    resolved_query TEXT,
                    key_facts TEXT,
                    resolved_queries TEXT,
                    importance REAL DEFAULT 5,
                    intent TEXT,
                    enriched_at TEXT,
                    enrich_status TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    aggregated_into TEXT,
                    brick_id TEXT,
                    source_uri TEXT,
                    status TEXT DEFAULT 'active',
                    ingested_at INTEGER,
                    topic_cluster TEXT
                )
            """)
        }

        try ensureChunkColumns()
        try ensurePreviewTextTriggers()
        try rebuildFTSTableIfNeeded()
        try rebuildTrigramFTSTable()
        try backfillPreviewText()

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_created_at
            ON chunks(created_at)
        """)
        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_enrich_status
            ON chunks(enrich_status)
        """)

        try ensurePendingStoreQueueIndex()

        try ensureAuxiliarySchema()
        try refreshSearchStatistics()
    }

    private func ensureAuxiliarySchema() throws {
        try execute("""
            CREATE TABLE IF NOT EXISTS brainbar_agents (
                agent_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL DEFAULT 1,
                connection_state TEXT NOT NULL DEFAULT 'disconnected',
                last_connected_at TEXT,
                disconnected_at TEXT,
                last_delivered_seq INTEGER NOT NULL DEFAULT 0,
                last_acked_seq INTEGER NOT NULL DEFAULT 0
            )
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS brainbar_subscriptions (
                agent_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                subscribed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (agent_id, tag),
                FOREIGN KEY (agent_id) REFERENCES brainbar_agents(agent_id) ON DELETE CASCADE
            ) WITHOUT ROWID
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_brainbar_subscriptions_tag
            ON brainbar_subscriptions(tag)
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                description TEXT,
                importance REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(entity_type, name)
            )
        """)
        try ensureKGEntityColumns()

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        try ensureKGRelationColumns()

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_chunks (
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                PRIMARY KEY (entity_id, chunk_id)
            )
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_aliases (
                alias TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                alias_type TEXT DEFAULT 'name',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                valid_from TEXT,
                valid_to TEXT,
                PRIMARY KEY (alias, entity_id)
            )
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_ec_entity
            ON kg_entity_chunks(entity_id)
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_alias_lookup
            ON kg_entity_aliases(alias COLLATE NOCASE)
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_alias_entity
            ON kg_entity_aliases(entity_id)
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS injection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                query TEXT NOT NULL,
                chunk_ids TEXT NOT NULL DEFAULT '[]',
                token_count INTEGER NOT NULL DEFAULT 0,
                mode TEXT NOT NULL DEFAULT 'normal'
            )
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_injection_events_session_timestamp
            ON injection_events(session_id, timestamp DESC)
        """)
        try ensureInjectionEventColumns()

        // Filtered VIEW excludes co_occurs_with noise from KG queries
        try execute("""
            CREATE VIEW IF NOT EXISTS kg_relations_typed AS
            SELECT id, source_id, target_id, relation_type, properties, created_at
            FROM kg_relations
            WHERE relation_type != 'co_occurs_with'
        """)
    }

    /// Lightweight migrations that run even when the full schema already exists.
    /// Safe to call repeatedly — all statements use IF NOT EXISTS.
    private func ensureMigrations() throws {
        // PR 1: Filtered VIEW for KG relations (excludes co_occurs_with)
        try execute("""
            CREATE VIEW IF NOT EXISTS kg_relations_typed AS
            SELECT id, source_id, target_id, relation_type, properties, created_at
            FROM kg_relations
            WHERE relation_type != 'co_occurs_with'
        """)

        try ensureChunkColumns()
        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_enrich_status
            ON chunks(enrich_status)
        """)
        try ensurePendingStoreQueueIndex()
        try ensureKGEntityColumns()
        try ensureKGRelationColumns()
        try ensureKGEntityAliasTable()
        try ensureInjectionEventColumns()
        try rebuildFTSTableIfNeeded()
        try rebuildTrigramFTSTableIfNeeded()
    }

    func close() {
        if let db {
            sqlite3_close(db)
            self.db = nil
        }
        isOpen = false
    }

    func vacuumInto(targetPath: String) throws -> Int64 {
        guard let db else { throw DBError.notOpen }
        let trimmedTarget = targetPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTarget.isEmpty else {
            throw DBError.exec(SQLITE_MISUSE, "target_path is required")
        }

        let targetURL = URL(fileURLWithPath: trimmedTarget).standardizedFileURL
        let sourceURL = URL(fileURLWithPath: path).standardizedFileURL
        guard targetURL.path != sourceURL.path else {
            throw DBError.exec(SQLITE_MISUSE, "refusing to VACUUM INTO the live database path")
        }

        try FileManager.default.createDirectory(
            at: targetURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard !FileManager.default.fileExists(atPath: targetURL.path) else {
            throw DBError.exec(SQLITE_CANTOPEN, "backup target already exists: \(targetURL.path)")
        }

        var stmt: OpaquePointer?
        let prepareRC = sqlite3_prepare_v2(db, "VACUUM INTO ?", -1, &stmt, nil)
        guard prepareRC == SQLITE_OK, let stmt else { throw DBError.prepare(prepareRC) }
        defer { sqlite3_finalize(stmt) }

        bindText(targetURL.path, to: stmt, index: 1)
        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }

        let attributes = try FileManager.default.attributesOfItem(atPath: targetURL.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static let allowedPragmas: Set<String> = [
        "journal_mode", "busy_timeout", "cache_size", "synchronous",
        "wal_checkpoint", "page_count", "page_size", "freelist_count"
    ]

    func pragma(_ name: String) throws -> String {
        guard Self.allowedPragmas.contains(name) else {
            throw DBError.invalidPragma(name)
        }
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA \(name)", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        guard let cStr = sqlite3_column_text(stmt, 0) else { throw DBError.noResult }
        return String(cString: cStr)
    }

    func tableExists(_ name: String) throws -> Bool {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name = ?"
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        bindText(name, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return false }
        return sqlite3_column_int(stmt, 0) > 0
    }

    func insertChunk(
        id: String,
        content: String,
        sessionId: String,
        project: String,
        contentType: String,
        importance: Int,
        sender: String? = nil,
        tags: String = "[]",
        createdAt: String? = nil
    ) throws {
        guard let db else { throw DBError.notOpen }
        if (try? tableExists("chunks")) != true {
            try ensureSchema()
        }
        let createdAt = createdAt ?? Self.timestamp()
        let sql = """
            INSERT OR REPLACE INTO chunks (
                id, content, metadata, source_file, project, content_type,
                importance, conversation_id, sender, char_count, tags, summary,
                preview_text, created_at, brick_id, source_uri, status, ingested_at
            )
            VALUES (?, ?, '{}', 'brainbar', ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, 'brainbar', 'active', CAST(strftime('%s', 'now') AS INTEGER))
        """
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(id, to: stmt, index: 1)
            bindText(content, to: stmt, index: 2)
            bindText(project, to: stmt, index: 3)
            bindText(contentType, to: stmt, index: 4)
            sqlite3_bind_int(stmt, 5, Int32(importance))
            bindText(sessionId, to: stmt, index: 6)
            if let sender {
                bindText(sender, to: stmt, index: 7)
            } else {
                sqlite3_bind_null(stmt, 7)
            }
            sqlite3_bind_int(stmt, 8, Int32(content.count))
            bindText(tags, to: stmt, index: 9)
            bindText(Self.previewText(summary: "", content: content), to: stmt, index: 10)
            bindText(createdAt, to: stmt, index: 11)
            bindText(id, to: stmt, index: 12)
        }
        try refreshSearchStatistics()
    }

    func search(
        query: String,
        limit: Int,
        project: String? = nil,
        source: String? = nil,
        tag: String? = nil,
        importanceMin: Double? = nil,
        subscriberID: String? = nil,
        unreadOnly: Bool = false
    ) throws -> [[String: Any]] {
        guard db != nil else { throw DBError.notOpen }
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)

        var subscribedTags: [String] = []
        var ackFloor: Int64 = 0
        if unreadOnly, let subscriberID, let record = try subscription(agentID: subscriberID) {
            subscribedTags = record.tags
            ackFloor = record.lastAckedSeq
        }

        if !unreadOnly,
           let exact = try exactChunkIDSearchResult(
               query: trimmedQuery,
               limit: limit,
               project: project,
               source: source,
               tag: tag,
               importanceMin: importanceMin
           ) {
            return exact
        }

        let expandedQueries = expandedSearchQueries(trimmedQuery)
        var results: [[String: Any]] = []
        var seenChunkIDs = Set<String>()
        var maxRowID: Int64 = 0

        for expandedQuery in expandedQueries {
            let searchResult = try runFTSSearch(
                tableName: "chunks_fts",
                matchQuery: sanitizeFTS5Query(expandedQuery),
                limit: max(limit, 1),
                project: project,
                source: source,
                tag: tag,
                importanceMin: importanceMin,
                subscribedTags: subscribedTags,
                ackFloor: ackFloor,
                unreadOnly: unreadOnly
            )
            maxRowID = max(maxRowID, searchResult.maxRowID)
            appendDeduped(searchResult.rows, to: &results, seenChunkIDs: &seenChunkIDs, limit: limit)
            if results.count >= limit { break }
        }

        if results.count < limit {
            for expandedQuery in expandedQueries {
                let trigramQuery = sanitizeTrigramFTS5Query(expandedQuery)
                guard !trigramQuery.isEmpty else { continue }
                let searchResult = try runFTSSearch(
                    tableName: "chunks_fts_trigram",
                    matchQuery: trigramQuery,
                    limit: max(limit, 1),
                    project: project,
                    source: source,
                    tag: tag,
                    importanceMin: importanceMin,
                    subscribedTags: subscribedTags,
                    ackFloor: ackFloor,
                    unreadOnly: unreadOnly
                )
                maxRowID = max(maxRowID, searchResult.maxRowID)
                appendDeduped(searchResult.rows, to: &results, seenChunkIDs: &seenChunkIDs, limit: limit)
                if results.count >= limit { break }
            }
        }

        if unreadOnly, let subscriberID, maxRowID > 0 {
            try markDelivered(agentID: subscriberID, seq: maxRowID)
        }

        return results
    }

    private func runFTSSearch(
        tableName: String,
        matchQuery: String,
        limit: Int,
        project: String?,
        source: String?,
        tag: String?,
        importanceMin: Double?,
        subscribedTags: [String],
        ackFloor: Int64,
        unreadOnly: Bool
    ) throws -> (rows: [[String: Any]], maxRowID: Int64) {
        guard let db else { throw DBError.notOpen }
        let allowedTables = ["chunks_fts", "chunks_fts_trigram"]
        guard allowedTables.contains(tableName) else { throw DBError.exec(SQLITE_ERROR, "invalid FTS table") }

        var subscribedTags = subscribedTags
        let sourceFilter = normalizedSourceFilter(source)
        var conditions = [
            "\(tableName) MATCH ?",
            "c.superseded_by IS NULL",
            "c.aggregated_into IS NULL",
            "c.archived_at IS NULL",
            "COALESCE(c.archived, 0) = 0",
            "COALESCE(c.status, 'active') = 'active'"
        ]
        if project != nil { conditions.append("(c.project = ? OR c.project IS NULL)") }
        if sourceFilter != nil { conditions.append("c.source = ?") }
        if let explicitTag = tag {
            conditions.append("c.tags LIKE ?")
            subscribedTags = [explicitTag]
        } else if unreadOnly, !subscribedTags.isEmpty {
            let tagTerms = Array(repeating: "c.tags LIKE ?", count: subscribedTags.count).joined(separator: " OR ")
            conditions.append("(\(tagTerms))")
        }
        if importanceMin != nil { conditions.append("c.importance >= ?") }
        if unreadOnly { conditions.append("c.rowid > ?") }

        // Unread mode needs contiguous rowid ordering for watermark semantics.
        // Normal search uses FTS5 BM25 rank for relevance ordering.
        let orderByClause = unreadOnly ? "c.rowid ASC" : "f.rank"
        let sql = """
            SELECT c.rowid, c.id, c.content, c.source_file, c.project, c.content_type, c.importance,
                   c.created_at, c.summary, c.tags, c.conversation_id, c.source, f.rank
            FROM \(tableName) f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE \(conditions.joined(separator: " AND "))
            ORDER BY \(orderByClause)
            LIMIT ?
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var paramIdx: Int32 = 1
        bindText(matchQuery, to: stmt, index: paramIdx)
        paramIdx += 1
        if let project {
            bindText(project, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let sourceFilter {
            bindText(sourceFilter, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let explicitTag = tag {
            bindText("%\"\(explicitTag)\"%", to: stmt, index: paramIdx)
            paramIdx += 1
        } else if unreadOnly, !subscribedTags.isEmpty {
            for subscribedTag in subscribedTags {
                bindText("%\"\(subscribedTag)\"%", to: stmt, index: paramIdx)
                paramIdx += 1
            }
        }
        if let importanceMin {
            sqlite3_bind_double(stmt, paramIdx, importanceMin)
            paramIdx += 1
        }
        if unreadOnly {
            sqlite3_bind_int64(stmt, paramIdx, ackFloor)
            paramIdx += 1
        }
        sqlite3_bind_int(stmt, paramIdx, Int32(limit))

        var results: [[String: Any]] = []
        var maxRowID: Int64 = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let rowID = sqlite3_column_int64(stmt, 0)
            maxRowID = max(maxRowID, rowID)
            // FTS5 rank is negative (lower = better match). Negate for a positive score.
            let rawRank = sqlite3_column_double(stmt, 12)
            let score = max(0, -rawRank)
            results.append(searchRow(from: stmt, score: score))
        }

        return (results, maxRowID)
    }

    private static func makeChunkID() -> String {
        "brainbar-\(UUID().uuidString.lowercased().prefix(12))"
    }

    func store(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        project: String? = nil,
        chunkID: String? = nil,
        queueID: String? = nil,
        createdAt: String? = nil,
        refreshStatistics: Bool = true,
        retries: Int = 3,
        busyTimeoutMillis: Int32? = nil
    ) throws -> StoredChunk {
        guard let db else { throw DBError.notOpen }
        let chunkID = chunkID ?? Self.makeChunkID()
        let createdAt = createdAt ?? Self.timestamp()
        let tagsJSON = (try? encodeJSON(tags)) ?? "[]"
        let metadataJSON = Self.storeMetadataJSON(queueID: queueID)
        let sql = """
            INSERT INTO chunks (id, content, metadata, source_file, project, tags, importance, source, content_type, char_count, created_at, preview_text)
            VALUES (?, ?, ?, 'brainbar-store', ?, ?, ?, ?, 'user_message', ?, ?, ?)
        """
        let previousBusyTimeout = busyTimeoutMillis.flatMap { _ in queryPragma(db, name: "busy_timeout") }
        if let busyTimeoutMillis {
            try executeOnHandle(db, sql: "PRAGMA busy_timeout = \(max(1, busyTimeoutMillis))")
        }
        defer {
            if let previousBusyTimeout {
                try? executeOnHandle(db, sql: "PRAGMA busy_timeout = \(previousBusyTimeout)")
            }
        }

        return try withImmediateTransaction(retries: retries) {
            try runWriteStatement(on: db, sql: sql, retries: retries) { stmt in
                bindText(chunkID, to: stmt, index: 1)
                bindText(content, to: stmt, index: 2)
                bindText(metadataJSON, to: stmt, index: 3)
                if let project {
                    bindText(project, to: stmt, index: 4)
                } else {
                    sqlite3_bind_null(stmt, 4)
                }
                bindText(tagsJSON, to: stmt, index: 5)
                sqlite3_bind_int(stmt, 6, Int32(importance))
                bindText(source, to: stmt, index: 7)
                sqlite3_bind_int(stmt, 8, Int32(content.count))
                bindText(createdAt, to: stmt, index: 9)
                bindText(Self.previewText(summary: "", content: content), to: stmt, index: 10)
            }
            if failNextStoreAfterInsertForTesting {
                failNextStoreAfterInsertForTesting = false
                throw DBError.exec(SQLITE_IOERR, "simulated post-insert validation failure")
            }
            guard let rowID = try chunkRowID(forChunkID: chunkID) else {
                throw DBError.noResult
            }
            if refreshStatistics {
                refreshSearchStatisticsBestEffort()
            }
            return StoredChunk(chunkID: chunkID, rowID: rowID)
        }
    }

    func storeOrQueueWithinBudget(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        project: String? = nil,
        busyTimeoutMillis: Int32 = 30_000,
        retries: Int = 3
    ) throws -> StoreWriteOutcome {
        let chunkID = Self.makeChunkID()
        let createdAt = Self.timestamp()
        do {
            if failNextStoreWithBusyForTesting {
                failNextStoreWithBusyForTesting = false
                throw DBError.exec(SQLITE_BUSY, "simulated busy store for queue fallback")
            }
            let stored = try store(
                content: content,
                tags: tags,
                importance: importance,
                source: source,
                project: project,
                chunkID: chunkID,
                createdAt: createdAt,
                refreshStatistics: true,
                retries: retries,
                busyTimeoutMillis: busyTimeoutMillis
            )
            return .stored(stored)
        } catch {
            guard shouldQueueStoreError(error) else {
                throw error
            }
            let queued = try queuePendingStore(
                content: content,
                tags: tags,
                importance: importance,
                source: source,
                project: project,
                createdAt: createdAt,
                chunkID: chunkID
            )
            return .queued(queueID: queued.queueID, queuedAt: queued.queuedAt, chunkID: queued.chunkID)
        }
    }

    /// Async wrapper for store() — runs DB write off the main thread.
    func storeAsync(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        project: String? = nil,
        createdAt: String? = nil
    ) async throws -> StoredChunk {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                do {
                    let result = try self.store(
                        content: content,
                        tags: tags,
                        importance: importance,
                        source: source,
                        project: project,
                        createdAt: createdAt
                    )
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func shouldQueueStoreError(_ error: Error) -> Bool {
        guard let dbError = error as? DBError else { return false }
        switch dbError {
        case .prepare(let rc), .step(let rc):
            return Self.isRetryableQueueErrorCode(rc)
        case .exec(let rc, _):
            return Self.isRetryableQueueErrorCode(rc)
        case .notOpen, .open, .noResult, .invalidPragma:
            return false
        }
    }

    @discardableResult
    func queuePendingStore(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        project: String? = nil,
        createdAt: String? = nil,
        chunkID: String? = nil
    ) throws -> (queueID: String, queuedAt: String, chunkID: String) {
        try Self.queuePendingStore(
            dbPath: path,
            content: content,
            tags: tags,
            importance: importance,
            source: source,
            project: project,
            createdAt: createdAt,
            chunkID: chunkID
        )
    }

    func pendingStoreQueuePathForReceipt() -> URL {
        pendingStorePath()
    }

    static func pendingStoreQueuePathForReceipt(dbPath: String) -> URL {
        pendingStorePath(forDBPath: dbPath)
    }

    @discardableResult
    static func queuePendingStore(
        dbPath: String,
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        project: String? = nil,
        createdAt: String? = nil,
        chunkID: String? = nil
    ) throws -> (queueID: String, queuedAt: String, chunkID: String) {
        Self.pendingStoreFileLock.lock()
        defer { Self.pendingStoreFileLock.unlock() }

        let path = pendingStorePath(forDBPath: dbPath)
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let queueID = UUID().uuidString.lowercased()
        let queuedAt = Self.timestamp()
        let createdAt = createdAt ?? queuedAt
        let chunkID = chunkID ?? Self.makeChunkID()
        let item = PendingStoreItem(
            content: content,
            tags: tags,
            importance: importance,
            source: source,
            project: project,
            queueID: queueID,
            queuedAt: queuedAt,
            createdAt: createdAt,
            chunkID: chunkID
        )
        var line = try JSONEncoder().encode(item)
        line.append(0x0A)
        try appendPendingStoreLine(line, to: path)
        return (queueID: queueID, queuedAt: queuedAt, chunkID: chunkID)
    }

    func flushPendingStores(excludingChunkIDs excludedChunkIDs: Set<String> = []) -> [FlushedPendingStore] {
        let path = pendingStorePath()
        Self.pendingStoreFileLock.lock()
        defer { Self.pendingStoreFileLock.unlock() }

        do {
            return try Self.withPendingStoreProcessLock(for: path) {
                guard FileManager.default.fileExists(atPath: path.path) else { return [] }
                guard let snapshot = Self.readPendingStoreData(at: path) else {
                    NSLog("[BrainBar] Failed to read pending stores queue at %@", path.path)
                    return []
                }
                let lines = Self.pendingStoreLines(from: snapshot)

                guard !lines.isEmpty else { return [] }

                let decoder = JSONDecoder()
                var flushed: [FlushedPendingStore] = []
                var remaining: [Data] = []

                for (lineIndex, line) in lines.enumerated() {
                    let item: PendingStoreItem
                    do {
                        item = try decoder.decode(PendingStoreItem.self, from: line)
                    } catch {
                        remaining.append(line)
                        continue
                    }

                    let queueID = Self.pendingStoreQueueID(for: item, lineIndex: lineIndex)
                    let chunkID = item.chunkID ?? Self.makeChunkID()
                    let replayLine = Self.pendingStoreReplayLine(for: item, queueID: queueID, chunkID: chunkID) ?? line
                    if excludedChunkIDs.contains(chunkID) {
                        remaining.append(replayLine)
                        continue
                    }
                    do {
                        if try hasStoredQueuedItem(queueID: queueID) {
                            continue
                        }
                    } catch {
                        NSLog("[BrainBar] Failed pending store dedupe lookup for %@: %@", queueID, String(describing: error))
                        remaining.append(replayLine)
                        continue
                    }

                    do {
                        let stored = try store(
                            content: item.content,
                            tags: item.tags,
                            importance: item.importance,
                            source: item.source,
                            project: item.project,
                            chunkID: chunkID,
                            queueID: queueID,
                            createdAt: item.createdAt ?? item.queuedAt,
                            refreshStatistics: false
                        )
                        flushed.append(
                            FlushedPendingStore(
                                storedChunk: stored,
                                content: item.content,
                                tags: item.tags,
                                importance: item.importance
                            )
                        )
                    } catch {
                        NSLog("[BrainBar] Failed to flush pending store item: %@", String(describing: error))
                        remaining.append(replayLine)
                    }
                }

                rewritePendingStoreFile(path: path, snapshot: snapshot, remainingLines: remaining)
                if !flushed.isEmpty {
                    refreshSearchStatisticsBestEffort()
                }
                return flushed
            }
        } catch {
            NSLog("[BrainBar] Failed to lock pending stores queue for flush: %@", String(describing: error))
            return []
        }
    }

    func searchCandidates(
        query: String,
        limit: Int,
        project: String? = nil,
        source: String? = nil,
        tag: String? = nil,
        importanceMin: Double? = nil,
        subscriberID: String? = nil,
        unreadOnly: Bool = false
    ) throws -> [SearchQueryCandidate] {
        guard let db else { throw DBError.notOpen }
        let sanitized = sanitizeFTS5Query(query)

        var subscribedTags: [String] = []
        var ackFloor: Int64 = 0
        if unreadOnly, let subscriberID, let record = try subscription(agentID: subscriberID) {
            subscribedTags = record.tags
            ackFloor = record.lastAckedSeq
        }

        let sourceFilter = normalizedSourceFilter(source)
        var conditions = [
            "chunks_fts MATCH ?",
            "c.superseded_by IS NULL",
            "c.aggregated_into IS NULL",
            "c.archived_at IS NULL",
            "COALESCE(c.archived, 0) = 0",
            "COALESCE(c.status, 'active') = 'active'"
        ]
        if project != nil { conditions.append("(c.project = ? OR c.project IS NULL)") }
        if sourceFilter != nil { conditions.append("c.source = ?") }
        if let explicitTag = tag {
            conditions.append("c.tags LIKE ?")
            subscribedTags = [explicitTag]
        } else if unreadOnly, !subscribedTags.isEmpty {
            let tagTerms = Array(repeating: "c.tags LIKE ?", count: subscribedTags.count).joined(separator: " OR ")
            conditions.append("(\(tagTerms))")
        }
        if importanceMin != nil { conditions.append("c.importance >= ?") }
        if unreadOnly { conditions.append("c.rowid > ?") }

        let orderByClause = unreadOnly ? "c.rowid ASC" : "f.rank"
        let sql = """
            SELECT c.rowid, c.id, c.preview_text, f.rank, c.created_at, c.project, c.importance
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE \(conditions.joined(separator: " AND "))
            ORDER BY \(orderByClause)
            LIMIT ?
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var paramIdx: Int32 = 1
        bindText(sanitized, to: stmt, index: paramIdx)
        paramIdx += 1
        if let project {
            bindText(project, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let sourceFilter {
            bindText(sourceFilter, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let explicitTag = tag {
            bindText("%\"\(explicitTag)\"%", to: stmt, index: paramIdx)
            paramIdx += 1
        } else if unreadOnly, !subscribedTags.isEmpty {
            for subscribedTag in subscribedTags {
                bindText("%\"\(subscribedTag)\"%", to: stmt, index: paramIdx)
                paramIdx += 1
            }
        }
        if let importanceMin {
            sqlite3_bind_double(stmt, paramIdx, importanceMin)
            paramIdx += 1
        }
        if unreadOnly {
            sqlite3_bind_int64(stmt, paramIdx, ackFloor)
            paramIdx += 1
        }
        sqlite3_bind_int(stmt, paramIdx, Int32(limit))

        var results: [SearchQueryCandidate] = []
        var maxRowID: Int64 = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let rowID = sqlite3_column_int64(stmt, 0)
            maxRowID = max(maxRowID, rowID)
            let rawRank = sqlite3_column_double(stmt, 3)
            let score = max(0, -rawRank)
            results.append(
                SearchQueryCandidate(
                    id: columnText(stmt, 1) ?? "",
                    previewText: columnText(stmt, 2) ?? "",
                    lexicalScore: score,
                    date: columnText(stmt, 4) ?? "",
                    project: columnText(stmt, 5) ?? "",
                    importance: Int(sqlite3_column_int(stmt, 6))
                )
            )
        }

        if unreadOnly, let subscriberID, maxRowID > 0 {
            try markDelivered(agentID: subscriberID, seq: maxRowID)
        }

        return results
    }

    func upsertSubscription(agentID: String, tags: [String], incrementGeneration: Bool = false) throws -> SubscriberRecord {
        try ensureAuxiliarySchema()
        let now = Self.timestamp()
        try withImmediateTransaction {
            if let existing = try subscription(agentID: agentID) {
                let nextGeneration = incrementGeneration ? existing.generation + 1 : existing.generation
                try executeUpdate(
                    """
                    UPDATE brainbar_agents
                    SET generation = ?, connection_state = 'connected', last_connected_at = ?, disconnected_at = NULL
                    WHERE agent_id = ?
                    """,
                    binds: { stmt in
                        sqlite3_bind_int(stmt, 1, Int32(nextGeneration))
                        bindText(now, to: stmt, index: 2)
                        bindText(agentID, to: stmt, index: 3)
                    }
                )
            } else {
                try executeUpdate(
                    """
                    INSERT INTO brainbar_agents (
                        agent_id, generation, connection_state, last_connected_at, disconnected_at, last_delivered_seq, last_acked_seq
                    ) VALUES (?, 1, 'connected', ?, NULL, 0, 0)
                    """,
                    binds: { stmt in
                        bindText(agentID, to: stmt, index: 1)
                        bindText(now, to: stmt, index: 2)
                    }
                )
            }

            for tag in Set(tags).sorted() where !tag.isEmpty {
                try executeUpdate(
                    """
                    INSERT OR IGNORE INTO brainbar_subscriptions (agent_id, tag)
                    VALUES (?, ?)
                    """,
                    binds: { stmt in
                        bindText(agentID, to: stmt, index: 1)
                        bindText(tag, to: stmt, index: 2)
                    }
                )
            }
        }

        return try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            generation: 1,
            connectionState: "connected",
            tags: tags.sorted(),
            lastDeliveredSeq: 0,
            lastAckedSeq: 0,
            lastConnectedAt: now,
            disconnectedAt: nil
        )
    }

    func removeSubscription(agentID: String, tags: [String]?) throws -> SubscriberRecord {
        try ensureAuxiliarySchema()
        try withImmediateTransaction {
            try ensureAgentRow(agentID: agentID)
            if let tags, !tags.isEmpty {
                for tag in tags where !tag.isEmpty {
                    try executeUpdate(
                        "DELETE FROM brainbar_subscriptions WHERE agent_id = ? AND tag = ?",
                        binds: { stmt in
                            bindText(agentID, to: stmt, index: 1)
                            bindText(tag, to: stmt, index: 2)
                        }
                    )
                }
            } else {
                try executeUpdate(
                    "DELETE FROM brainbar_subscriptions WHERE agent_id = ?",
                    binds: { stmt in bindText(agentID, to: stmt, index: 1) }
                )
            }
        }

        return try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            generation: 1,
            connectionState: "disconnected",
            tags: [],
            lastDeliveredSeq: 0,
            lastAckedSeq: 0,
            lastConnectedAt: nil,
            disconnectedAt: nil
        )
    }

    func subscription(agentID: String) throws -> SubscriberRecord? {
        try ensureAuxiliarySchema()
        guard let db else { throw DBError.notOpen }

        var stmt: OpaquePointer?
        let sql = """
            SELECT agent_id, generation, connection_state, last_connected_at, disconnected_at,
                   last_delivered_seq, last_acked_seq
            FROM brainbar_agents
            WHERE agent_id = ?
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(agentID, to: stmt, index: 1)

        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        let tags = try subscriptionTags(agentID: agentID)
        return SubscriberRecord(
            agentID: columnText(stmt, 0) ?? agentID,
            generation: Int(sqlite3_column_int(stmt, 1)),
            connectionState: columnText(stmt, 2) ?? "disconnected",
            tags: tags,
            lastDeliveredSeq: sqlite3_column_int64(stmt, 5),
            lastAckedSeq: sqlite3_column_int64(stmt, 6),
            lastConnectedAt: columnText(stmt, 3),
            disconnectedAt: columnText(stmt, 4)
        )
    }

    func markSubscriberDisconnected(agentID: String) throws {
        try ensureAuxiliarySchema()
        try ensureAgentRow(agentID: agentID)
        try executeUpdate(
            """
            UPDATE brainbar_agents
            SET connection_state = 'disconnected', disconnected_at = ?
            WHERE agent_id = ?
            """,
            binds: { stmt in
                bindText(Self.timestamp(), to: stmt, index: 1)
                bindText(agentID, to: stmt, index: 2)
            }
        )
    }

    func markDelivered(agentID: String, seq: Int64) throws {
        try ensureAuxiliarySchema()
        try ensureAgentRow(agentID: agentID)
        try executeUpdate(
            """
            UPDATE brainbar_agents
            SET last_delivered_seq = MAX(last_delivered_seq, ?), connection_state = 'connected', disconnected_at = NULL
            WHERE agent_id = ?
            """,
            binds: { stmt in
                sqlite3_bind_int64(stmt, 1, seq)
                bindText(agentID, to: stmt, index: 2)
            }
        )
    }

    func acknowledge(agentID: String, seq: Int64) throws {
        try ensureAuxiliarySchema()
        try ensureAgentRow(agentID: agentID)
        try executeUpdate(
            """
            UPDATE brainbar_agents
            SET last_delivered_seq = MAX(last_delivered_seq, ?),
                last_acked_seq = MAX(last_acked_seq, ?),
                connection_state = 'connected',
                disconnected_at = NULL
            WHERE agent_id = ?
            """,
            binds: { stmt in
                sqlite3_bind_int64(stmt, 1, seq)
                sqlite3_bind_int64(stmt, 2, seq)
                bindText(agentID, to: stmt, index: 3)
            }
        )
    }

    func chunkRowID(forChunkID chunkID: String) throws -> Int64? {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT rowid FROM chunks WHERE id = ?", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(chunkID, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return sqlite3_column_int64(stmt, 0)
    }

    func storedChunkDetails(chunkID: String, rowID: Int64) throws -> StoredChunkDetails? {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT content, tags, importance FROM chunks WHERE id = ? AND rowid = ?", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(chunkID, to: stmt, index: 1)
        sqlite3_bind_int64(stmt, 2, rowID)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }

        let content = columnText(stmt, 0) ?? ""
        let tagsJSON = columnText(stmt, 1) ?? "[]"
        let tagsData = Data(tagsJSON.utf8)
        let tags = (try? JSONDecoder().decode([String].self, from: tagsData)) ?? []
        let importance = Int(sqlite3_column_int(stmt, 2))
        return StoredChunkDetails(content: content, tags: tags, importance: importance)
    }

    func unreadCount(agentID: String, tags: [String]? = nil) throws -> Int {
        try ensureAuxiliarySchema()
        let ackFloor = try subscription(agentID: agentID)?.lastAckedSeq ?? 0
        guard let db else { throw DBError.notOpen }

        var conditions = ["rowid > ?"]
        let filterTags = (tags?.isEmpty == false) ? tags! : try subscription(agentID: agentID)?.tags ?? []
        if !filterTags.isEmpty {
            let tagTerms = Array(repeating: "tags LIKE ?", count: filterTags.count).joined(separator: " OR ")
            conditions.append("(\(tagTerms))")
        }

        let sql = "SELECT COUNT(*) FROM chunks WHERE \(conditions.joined(separator: " AND "))"
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var paramIdx: Int32 = 1
        sqlite3_bind_int64(stmt, paramIdx, ackFloor)
        paramIdx += 1
        for tag in filterTags {
            bindText("%\"\(tag)\"%", to: stmt, index: paramIdx)
            paramIdx += 1
        }

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    func dashboardStats(activityWindowMinutes: Int = 30, bucketCount: Int = 12) throws -> DashboardStats {
        let pendingStoreQueue = pendingStoreQueueSnapshot()
        let watcherHealth = watcherHealthSnapshot()
        return try withReadTransaction {
            let liveWindowMinutes = 1
            let now = Date()
            guard bucketCount > 0 else {
                return DashboardStats(
                    chunkCount: 0,
                    enrichedChunkCount: 0,
                    pendingEnrichmentCount: 0,
                    enrichmentPercent: 0,
                    enrichmentRatePerMinute: 0,
                    databaseSizeBytes: databaseSizeBytes(),
                    recentActivityBuckets: [],
                    recentEnrichmentBuckets: [],
                    recentWriteFiveMinuteCount: 0,
                    recentEnrichmentFiveMinuteCount: 0,
                    activityWindowMinutes: activityWindowMinutes,
                    bucketCount: bucketCount,
                    liveWindowMinutes: liveWindowMinutes,
                    watcherHealth: watcherHealth
                )
            }

            let counts = try dashboardCounts()
            let signalCounts = try dashboardSignalCoverageCounts()
            let lastEvents = try dashboardLastEvents(now: now)
            let chunkCount = counts.chunkCount
            let enrichedChunkCount = counts.enrichedChunkCount
            let pendingEnrichmentCount = counts.pendingEnrichmentCount
            let enrichmentPercent = chunkCount == 0 ? 0 : (Double(enrichedChunkCount) / Double(chunkCount)) * 100
            let enrichmentRatePerMinute = try recentEnrichmentRatePerMinute(windowMinutes: liveWindowMinutes, now: now)
            let recentActivityBuckets = try recentActivityBuckets(
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )
            let recentAgentWriteBuckets = try recentWriteBuckets(
                whereClause: Self.agentWriteSourceWhereClause,
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )
            let recentWatcherWriteBuckets = try recentWriteBuckets(
                whereClause: Self.watcherWriteSourceWhereClause,
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )
            let recentEnrichmentBuckets = try recentEnrichmentBuckets(
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )
            let recentWriteFiveMinuteCount = try recentActivityCount(windowSeconds: 300, now: now)
            let recentEnrichmentFiveMinuteCount = try recentEnrichmentCount(windowSeconds: 300, now: now)

            return DashboardStats(
                chunkCount: chunkCount,
                enrichedChunkCount: enrichedChunkCount,
                pendingEnrichmentCount: pendingEnrichmentCount,
                enrichmentPercent: enrichmentPercent,
                enrichmentRatePerMinute: enrichmentRatePerMinute,
                databaseSizeBytes: databaseSizeBytes(),
                recentActivityBuckets: recentActivityBuckets,
                recentAgentWriteBuckets: recentAgentWriteBuckets,
                recentWatcherWriteBuckets: recentWatcherWriteBuckets,
                recentEnrichmentBuckets: recentEnrichmentBuckets,
                recentWriteFiveMinuteCount: recentWriteFiveMinuteCount,
                recentEnrichmentFiveMinuteCount: recentEnrichmentFiveMinuteCount,
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                liveWindowMinutes: liveWindowMinutes,
                lastWriteAt: lastEvents.lastWriteAt,
                lastEnrichedAt: lastEvents.lastEnrichedAt,
                signalEligibleChunkCount: signalCounts.eligibleChunkCount,
                vectorIndexedChunkCount: signalCounts.vectorIndexedChunkCount,
                ftsIndexedChunkCount: signalCounts.ftsIndexedChunkCount,
                trigramIndexedChunkCount: signalCounts.trigramIndexedChunkCount,
                pendingStoreQueueDepth: pendingStoreQueue.depth,
                pendingStoreOldestQueuedAt: pendingStoreQueue.oldestQueuedAt,
                watcherHealth: watcherHealth
            )
        }
    }

    /// Windowed re-fetch of the three pipeline series for an ARBITRARY window
    /// (180 min for 3h, 1440 for 24h, …). This is what makes the shared
    /// Live/3h/24h selector show REAL historical data instead of relabeling the
    /// live buckets: the view calls this on timeframe change and feeds the result
    /// into the charts. Reuses the exact same per-source bucketing the live
    /// `dashboardStats` uses, so the wider windows are honest DB reads.
    func pipelineWindowBuckets(
        activityWindowMinutes: Int,
        bucketCount: Int = 12,
        now: Date = Date()
    ) throws -> PipelineWindowBuckets {
        try withReadTransaction {
            guard bucketCount > 0, activityWindowMinutes > 0 else {
                return PipelineWindowBuckets(
                    activityWindowMinutes: activityWindowMinutes,
                    bucketCount: bucketCount,
                    agentWriteBuckets: Array(repeating: 0, count: max(bucketCount, 0)),
                    watcherWriteBuckets: Array(repeating: 0, count: max(bucketCount, 0)),
                    enrichmentBuckets: Array(repeating: 0, count: max(bucketCount, 0))
                )
            }

            let agentWriteBuckets = try recentWriteBuckets(
                whereClause: Self.agentWriteSourceWhereClause,
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )
            let watcherWriteBuckets = try recentWriteBuckets(
                whereClause: Self.watcherWriteSourceWhereClause,
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )
            let enrichmentBuckets = try recentEnrichmentBuckets(
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                now: now
            )

            return PipelineWindowBuckets(
                activityWindowMinutes: activityWindowMinutes,
                bucketCount: bucketCount,
                agentWriteBuckets: agentWriteBuckets,
                watcherWriteBuckets: watcherWriteBuckets,
                enrichmentBuckets: enrichmentBuckets
            )
        }
    }

    private func watcherHealthSnapshot() -> DashboardStats.WatcherHealth? {
        let envPath = ProcessInfo.processInfo.environment["BRAINLAYER_WATCHER_HEALTH_PATH"]
        let healthPath = envPath ?? URL(fileURLWithPath: path)
            .deletingLastPathComponent()
            .appendingPathComponent("watcher-health.json")
            .path
        guard let data = FileManager.default.contents(atPath: healthPath),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return DashboardStats.WatcherHealth(
            alerting: payload["alerting"] as? Bool ?? false,
            filesTracked: payload["files_tracked"] as? Int ?? 0,
            maxOffsetLagBytes: payload["max_offset_lag_bytes"] as? Int ?? 0,
            activeEntriesPerMinute: payload["active_jsonl_entries_per_minute"] as? Double ?? 0,
            realtimeInsertsPerMinute: payload["db_realtime_inserts_per_minute"] as? Double ?? 0
        )
    }

    func dataVersion() throws -> Int {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA data_version", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    func exec(_ sql: String) {
        do {
            try execute(sql)
        } catch {
            NSLog("[BrainBar] SQL error: %@", String(describing: error))
        }
    }

    private func ensureAgentRow(agentID: String) throws {
        if try subscription(agentID: agentID) != nil {
            return
        }
        try executeUpdate(
            """
            INSERT OR IGNORE INTO brainbar_agents (
                agent_id, generation, connection_state, last_connected_at, disconnected_at, last_delivered_seq, last_acked_seq
            ) VALUES (?, 1, 'disconnected', NULL, NULL, 0, 0)
            """,
            binds: { stmt in bindText(agentID, to: stmt, index: 1) }
        )
    }

    private func scalarInt(_ sql: String) throws -> Int {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    private func dashboardCounts() throws -> (chunkCount: Int, enrichedChunkCount: Int, pendingEnrichmentCount: Int) {
        return (
            chunkCount: try scalarInt("SELECT COUNT(*) FROM chunks"),
            enrichedChunkCount: try scalarInt("SELECT COUNT(*) FROM chunks WHERE enrich_status IS NOT NULL"),
            pendingEnrichmentCount: try scalarInt("SELECT COUNT(*) FROM chunks WHERE enrich_status IS NULL AND enriched_at IS NULL")
        )
    }

    private func dashboardSignalCoverageCounts() throws -> (
        eligibleChunkCount: Int,
        vectorIndexedChunkCount: Int,
        ftsIndexedChunkCount: Int,
        trigramIndexedChunkCount: Int
    ) {
        let searchableWhereClause = try searchableChunkWhereClause(alias: "c")
        return (
            eligibleChunkCount: try scalarInt("SELECT COUNT(*) FROM chunks AS c WHERE \(searchableWhereClause)"),
            vectorIndexedChunkCount: try joinedCoverageCount(
                tableName: "chunk_vectors_rowids",
                idColumn: "id",
                searchableWhereClause: searchableWhereClause
            ),
            ftsIndexedChunkCount: try joinedCoverageCount(
                tableName: "chunks_fts",
                idColumn: "chunk_id",
                searchableWhereClause: searchableWhereClause
            ),
            trigramIndexedChunkCount: try joinedCoverageCount(
                tableName: "chunks_fts_trigram",
                idColumn: "chunk_id",
                searchableWhereClause: searchableWhereClause
            )
        )
    }

    private func joinedCoverageCount(tableName: String, idColumn: String, searchableWhereClause: String) throws -> Int {
        let allowedTables = [
            "chunk_vectors_rowids": "id",
            "chunks_fts": "chunk_id",
            "chunks_fts_trigram": "chunk_id",
        ]
        guard allowedTables[tableName] == idColumn else {
            throw DBError.exec(SQLITE_ERROR, "invalid coverage table")
        }
        guard let db else { throw DBError.notOpen }
        guard try tableExists(tableName) else { return 0 }
        guard try tableColumns(name: tableName, on: db).contains(idColumn) else { return 0 }
        return try scalarInt("""
            SELECT COUNT(DISTINCT signal.\(idColumn))
            FROM \(tableName) AS signal
            INNER JOIN chunks AS c ON c.id = signal.\(idColumn)
            WHERE \(searchableWhereClause)
        """)
    }

    private func searchableChunkWhereClause(alias: String) throws -> String {
        guard let db else { throw DBError.notOpen }
        let columns = try tableColumns(name: "chunks", on: db)
        var clauses: [String] = []
        if columns.contains("content") {
            clauses.append("\(alias).content IS NOT NULL")
            clauses.append("\(alias).content != ''")
        }
        if columns.contains("superseded_by") {
            clauses.append("\(alias).superseded_by IS NULL")
        }
        if columns.contains("aggregated_into") {
            clauses.append("\(alias).aggregated_into IS NULL")
        }
        if columns.contains("archived_at") {
            clauses.append("\(alias).archived_at IS NULL")
        }
        if columns.contains("archived") {
            clauses.append("COALESCE(\(alias).archived, 0) = 0")
        }
        if columns.contains("status") {
            clauses.append("COALESCE(\(alias).status, 'active') = 'active'")
        }
        return clauses.isEmpty ? "1 = 1" : clauses.joined(separator: " AND ")
    }

    private func dashboardLastEvents(now: Date) throws -> (lastWriteAt: Date?, lastEnrichedAt: Date?) {
        let recentWindowStart = now.addingTimeInterval(-Self.dashboardLatestEventLookbackSeconds)
        let lastWriteAt = try latestIndexedTimestampEpoch(
            column: "created_at",
            whereClause: nil,
            since: recentWindowStart
        )
        let lastEnrichedAt = try latestIndexedTimestampEpoch(
            column: "enriched_at",
            whereClause: "enriched_at IS NOT NULL AND enrich_status = 'success'",
            since: recentWindowStart
        )
        return (lastWriteAt: lastWriteAt, lastEnrichedAt: lastEnrichedAt)
    }

    private func recentActivityBuckets(activityWindowMinutes: Int, bucketCount: Int, now: Date) throws -> [Int] {
        guard activityWindowMinutes > 0 else { return Array(repeating: 0, count: bucketCount) }

        return try indexedTimestampBuckets(
            column: "created_at",
            whereClause: nil,
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: bucketCount,
            now: now
        )
    }

    private func recentWriteBuckets(
        whereClause: String,
        activityWindowMinutes: Int,
        bucketCount: Int,
        now: Date
    ) throws -> [Int] {
        guard activityWindowMinutes > 0 else { return Array(repeating: 0, count: bucketCount) }

        return try indexedTimestampBuckets(
            column: "created_at",
            whereClause: whereClause,
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: bucketCount,
            now: now
        )
    }

    private func recentEnrichmentBuckets(activityWindowMinutes: Int, bucketCount: Int, now: Date) throws -> [Int] {
        guard activityWindowMinutes > 0 else { return Array(repeating: 0, count: bucketCount) }

        return try indexedTimestampBuckets(
            column: "enriched_at",
            whereClause: "enriched_at IS NOT NULL AND enrich_status = 'success'",
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: bucketCount,
            now: now
        )
    }

    private func recentActivityCount(windowSeconds: TimeInterval, now: Date) throws -> Int {
        guard windowSeconds > 0 else { return 0 }
        return try indexedTimestampEpochs(
            column: "created_at",
            whereClause: nil,
            since: now.addingTimeInterval(-windowSeconds)
        ).count
    }

    private func recentEnrichmentCount(windowSeconds: TimeInterval, now: Date) throws -> Int {
        guard windowSeconds > 0 else { return 0 }
        return try indexedTimestampEpochs(
            column: "enriched_at",
            whereClause: "enriched_at IS NOT NULL AND enrich_status = 'success'",
            since: now.addingTimeInterval(-windowSeconds)
        ).count
    }

    private func recentEnrichmentRatePerMinute(windowMinutes: Int, now: Date) throws -> Double {
        guard windowMinutes > 0 else { return 0 }
        let count = Double(try indexedTimestampEpochs(
            column: "enriched_at",
            whereClause: "enriched_at IS NOT NULL AND enrich_status = 'success'",
            since: now.addingTimeInterval(Double(-windowMinutes * 60))
        ).count)
        return count / Double(windowMinutes)
    }

    private func indexedTimestampBuckets(
        column: String,
        whereClause: String?,
        activityWindowMinutes: Int,
        bucketCount: Int,
        now: Date
    ) throws -> [Int] {
        let windowDuration = Double(activityWindowMinutes * 60)
        let bucketWidthSeconds = max(1, windowDuration / Double(bucketCount))
        let windowStart = now.addingTimeInterval(-windowDuration)
        let epochs = try indexedTimestampEpochs(
            column: column,
            whereClause: whereClause,
            since: windowStart
        )
        var buckets = Array(repeating: 0, count: bucketCount)

        for epoch in epochs {
            let eventDate = Date(timeIntervalSince1970: TimeInterval(epoch))
            let offset = eventDate.timeIntervalSince(windowStart)
            if offset < 0 { continue }
            if offset > windowDuration { continue }

            let rawIndex = Int(offset / bucketWidthSeconds)
            let clampedIndex = min(max(rawIndex, 0), bucketCount - 1)
            buckets[clampedIndex] += 1
        }

        return buckets
    }

    private func indexedTimestampEpochs(
        column: String,
        whereClause: String?,
        since cutoffDate: Date
    ) throws -> [Int64] {
        guard let db else { throw DBError.notOpen }
        let epochSQL = Self.normalizedUnixEpochSQL(for: column)
        let indexLowerBound = cutoffDate.addingTimeInterval(-Self.dashboardTimestampIndexLookbackPaddingSeconds)
        let cutoffText = Self.dashboardTimestampIndexCutoffFormatter.string(from: indexLowerBound)
        var predicates = ["\(column) IS NOT NULL", "\(column) >= ?"]
        if let whereClause {
            predicates.append(whereClause)
        }
        let sql = """
            SELECT \(epochSQL) AS event_epoch
            FROM chunks
            WHERE \(predicates.joined(separator: " AND "))
            ORDER BY \(column) ASC
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(cutoffText, to: stmt, index: 1)

        let cutoffEpoch = Int64(cutoffDate.timeIntervalSince1970)
        var epochs: [Int64] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard sqlite3_column_type(stmt, 0) != SQLITE_NULL else { continue }
            let epoch = sqlite3_column_int64(stmt, 0)
            guard epoch >= cutoffEpoch else { continue }
            epochs.append(epoch)
        }
        return epochs
    }

    private func latestIndexedTimestampEpoch(
        column: String,
        whereClause: String?,
        since cutoffDate: Date
    ) throws -> Date? {
        let recentEpoch = try latestIndexedEpochSince(
            column: column,
            whereClause: whereClause,
            since: cutoffDate
        )
        if let recentEpoch {
            return Date(timeIntervalSince1970: TimeInterval(recentEpoch))
        }

        guard let db else { throw DBError.notOpen }
        let epochSQL = Self.normalizedUnixEpochSQL(for: column)
        var predicates = ["\(column) IS NOT NULL", "TRIM(\(column)) != ''"]
        if let whereClause {
            predicates.append(whereClause)
        }
        let sql = """
            SELECT MAX(event_epoch)
            FROM (
                SELECT \(epochSQL) AS event_epoch
                FROM chunks
                WHERE \(predicates.joined(separator: " AND "))
            )
            WHERE event_epoch IS NOT NULL
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        guard sqlite3_step(stmt) == SQLITE_ROW, sqlite3_column_type(stmt, 0) != SQLITE_NULL else {
            return nil
        }

        return Date(timeIntervalSince1970: TimeInterval(sqlite3_column_int64(stmt, 0)))
    }

    private func latestIndexedEpochSince(
        column: String,
        whereClause: String?,
        since cutoffDate: Date
    ) throws -> Int64? {
        guard let db else { throw DBError.notOpen }
        let epochSQL = Self.normalizedUnixEpochSQL(for: column)
        let indexLowerBound = cutoffDate.addingTimeInterval(-Self.dashboardTimestampIndexLookbackPaddingSeconds)
        let cutoffText = Self.dashboardTimestampIndexCutoffFormatter.string(from: indexLowerBound)
        var predicates = ["\(column) IS NOT NULL", "\(column) >= ?"]
        if let whereClause {
            predicates.append(whereClause)
        }
        let sql = """
            SELECT MAX(\(epochSQL)) AS event_epoch
            FROM chunks
            WHERE \(predicates.joined(separator: " AND "))
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(cutoffText, to: stmt, index: 1)

        let stepRc = sqlite3_step(stmt)
        guard stepRc == SQLITE_ROW else { throw DBError.step(stepRc) }
        guard sqlite3_column_type(stmt, 0) != SQLITE_NULL else { return nil }

        let epoch = sqlite3_column_int64(stmt, 0)
        let cutoffEpoch = Int64(cutoffDate.timeIntervalSince1970)
        return epoch >= cutoffEpoch ? epoch : nil
    }

    private static func normalizedUnixEpochSQL(for column: String) -> String {
        """
        CASE
            WHEN \(column) IS NULL OR TRIM(\(column)) = '' THEN NULL
            WHEN substr(\(column), -1) = 'Z' THEN unixepoch(\(column))
            WHEN substr(\(column), -6, 1) IN ('+', '-') THEN unixepoch(\(column))
            WHEN instr(\(column), 'T') > 0 THEN unixepoch(\(column), 'utc')
            ELSE unixepoch(\(column))
        END
        """
    }

    private static let dashboardTimestampIndexCutoffFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter
    }()

    private static let dashboardLatestEventLookbackSeconds: TimeInterval = 30 * 24 * 60 * 60
    private static let dashboardTimestampIndexLookbackPaddingSeconds: TimeInterval = 24 * 60 * 60

    private func databaseSizeBytes() -> Int64 {
        let candidates = [path, "\(path)-wal", "\(path)-shm"]
        return candidates.reduce(into: Int64(0)) { total, candidate in
            let attributes = try? FileManager.default.attributesOfItem(atPath: candidate)
            total += (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        }
    }

    private func subscriptionTags(agentID: String) throws -> [String] {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT tag FROM brainbar_subscriptions WHERE agent_id = ? ORDER BY tag"
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(agentID, to: stmt, index: 1)

        var tags: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let tag = columnText(stmt, 0) {
                tags.append(tag)
            }
        }
        return tags
    }

    private func withImmediateTransaction<T>(
        retries: Int = 3,
        shouldCancel: () -> Bool = { false },
        _ body: () throws -> T
    ) throws -> T {
        transactionLock.lock()
        defer { transactionLock.unlock() }
        guard !explicitTransactionIsOpen else {
            throw DBError.exec(SQLITE_MISUSE, "nested transactions are not supported")
        }
        if shouldCancel() {
            throw CancellationError()
        }
        try execute("BEGIN IMMEDIATE", retries: retries)
        explicitTransactionIsOpen = true
        do {
            let result = try body()
            try execute("COMMIT", retries: 3)
            explicitTransactionIsOpen = false
            return result
        } catch {
            try? execute("ROLLBACK")
            explicitTransactionIsOpen = false
            throw error
        }
    }

    private func withReadTransaction<T>(
        shouldCancel: () -> Bool = { false },
        _ body: () throws -> T
    ) throws -> T {
        transactionLock.lock()
        defer { transactionLock.unlock() }
        guard !explicitTransactionIsOpen else {
            throw DBError.exec(SQLITE_MISUSE, "nested transactions are not supported")
        }
        if shouldCancel() {
            throw CancellationError()
        }
        try execute("BEGIN", retries: 3)
        explicitTransactionIsOpen = true
        do {
            let result = try body()
            try execute("COMMIT", retries: 3)
            explicitTransactionIsOpen = false
            return result
        } catch {
            try? execute("ROLLBACK")
            explicitTransactionIsOpen = false
            throw error
        }
    }

    private func executeUpdate(
        _ sql: String,
        binds: (OpaquePointer) -> Void
    ) throws {
        guard let db else { throw DBError.notOpen }
        try runWriteStatement(on: db, sql: sql, retries: 3, bind: binds)
    }

    private func execute(_ sql: String, retries: Int = 0, retryDelayMillis: UInt32 = 250) throws {
        if Self.isMutationStatement(sql) {
            transactionLock.lock()
            defer { transactionLock.unlock() }
            try executeUnlocked(sql, retries: retries, retryDelayMillis: retryDelayMillis)
            return
        }

        try executeUnlocked(sql, retries: retries, retryDelayMillis: retryDelayMillis)
    }

    private func executeUnlocked(_ sql: String, retries: Int = 0, retryDelayMillis: UInt32 = 250) throws {
        guard let db else { throw DBError.notOpen }
        var attempts = 0
        while true {
            var errMsg: UnsafeMutablePointer<CChar>?
            let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
            if rc == SQLITE_OK {
                sqlite3_free(errMsg)
                if Self.isMutationStatement(sql) {
                    Self.postDashboardChangeNotification()
                }
                return
            }

            let message = errMsg.map { String(cString: $0) } ?? "unknown error"
            sqlite3_free(errMsg)

            if Self.isRetryableStoreErrorCode(rc), attempts < retries {
                attempts += 1
                usleep(retryDelayMillis * 1_000)
                continue
            }

            throw DBError.exec(rc, message)
        }
    }

    private func runWriteStatement(
        on handle: OpaquePointer?,
        sql: String,
        retries: Int = 0,
        retryDelayMillis: UInt32 = 250,
        bind: (OpaquePointer) -> Void
    ) throws {
        transactionLock.lock()
        defer { transactionLock.unlock() }
        try runWriteStatementLocked(
            on: handle,
            sql: sql,
            retries: retries,
            retryDelayMillis: retryDelayMillis,
            bind: bind
        )
    }

    private func runWriteStatementLocked(
        on handle: OpaquePointer?,
        sql: String,
        retries: Int = 0,
        retryDelayMillis: UInt32 = 250,
        bind: (OpaquePointer) -> Void
    ) throws {
        guard let handle else { throw DBError.notOpen }
        var attempts = 0

        while true {
            var stmt: OpaquePointer?
            let prepareRC = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
            guard prepareRC == SQLITE_OK, let stmt else {
                if Self.isRetryableStoreErrorCode(prepareRC), attempts < retries {
                    attempts += 1
                    usleep(retryDelayMillis * 1_000)
                    continue
                }
                throw DBError.prepare(prepareRC)
            }

            bind(stmt)
            let stepRC = sqlite3_step(stmt)
            sqlite3_finalize(stmt)

            if stepRC == SQLITE_DONE {
                Self.postDashboardChangeNotification()
                return
            }

            if Self.isRetryableStoreErrorCode(stepRC), attempts < retries {
                attempts += 1
                usleep(retryDelayMillis * 1_000)
                continue
            }

            throw DBError.step(stepRC)
        }
    }

    private func openConnection(path: String) throws -> OpaquePointer {
        var handle: OpaquePointer?
        let flags = openConfiguration.readOnly
            ? SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX
            : SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &handle, flags, nil)
        guard rc == SQLITE_OK, let handle else { throw DBError.open(path, rc) }
        NSLog("[BrainBar] Database opened %@", openConfiguration.readOnly ? "READONLY" : "READWRITE")
        return handle
    }

    private func configureConnection(_ handle: OpaquePointer?) throws {
        guard let handle else { throw DBError.notOpen }
        // AIDEV-NOTE: busy_timeout FIRST — 30s because watch agent holds locks during enrichment.
        try executeOnHandle(handle, sql: "PRAGMA busy_timeout = \(openConfiguration.busyTimeoutMillis)")
        if openConfiguration.readOnly {
            try executeOnHandle(handle, sql: "PRAGMA query_only = ON")
            try executeOnHandle(handle, sql: "PRAGMA cache_size = -64000")
            try executeOnHandle(handle, sql: "PRAGMA foreign_keys = ON")
            return
        }
        // AIDEV-NOTE: Skip journal_mode=WAL if already set — the PRAGMA itself needs a write lock
        // which blocks indefinitely when the watch agent is active. WAL is already set by Python.
        let currentMode = queryPragma(handle, name: "journal_mode")
        if currentMode?.lowercased() != "wal" {
            try executeOnHandle(handle, sql: "PRAGMA journal_mode = WAL")
        } else {
            NSLog("[BrainBar] journal_mode already WAL — skipping PRAGMA")
        }
        try executeOnHandle(handle, sql: "PRAGMA cache_size = -64000")
        try executeOnHandle(handle, sql: "PRAGMA synchronous = NORMAL")
        try executeOnHandle(handle, sql: "PRAGMA foreign_keys = ON")
    }

    private func queryPragma(_ handle: OpaquePointer, name: String) -> String? {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, "PRAGMA \(name)", -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else { return nil }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        guard let cStr = sqlite3_column_text(stmt, 0) else { return nil }
        return String(cString: cStr)
    }

    private func executeOnHandle(_ handle: OpaquePointer, sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &errMsg)
        if rc == SQLITE_OK {
            sqlite3_free(errMsg)
            return
        }
        let message = errMsg.map { String(cString: $0) } ?? "unknown error"
        sqlite3_free(errMsg)
        throw DBError.exec(rc, message)
    }

    private func encodeJSON(_ array: [String]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: array)
        return String(data: data, encoding: .utf8) ?? "[]"
    }

    private func columnText(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        guard let cStr = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: cStr)
    }

    private func bindText(_ value: String, to stmt: OpaquePointer?, index: Int32) {
        sqlite3_bind_text(stmt, index, value, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
    }

    private static func isMutationStatement(_ sql: String) -> Bool {
        let keyword = sql
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)
            .first?
            .lowercased()

        return [
            "insert", "update", "delete", "replace",
            "create", "drop", "alter"
        ].contains(keyword ?? "")
    }

    private static func postDashboardChangeNotification() {
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(),
            CFNotificationName(dashboardDidChangeNotification as CFString),
            nil,
            nil,
            true
        )
    }

    private func sanitizeFTS5Query(_ query: String) -> String {
        let tokens = query.split(separator: " ").compactMap { token -> String? in
            let cleaned = token
                .replacingOccurrences(of: "\"", with: "")
                .replacingOccurrences(of: "*", with: "")
                .trimmingCharacters(in: .whitespaces)
            guard !cleaned.isEmpty else { return nil }
            return "\"\(cleaned)\"*"
        }
        guard !tokens.isEmpty else { return "\"\"" }
        // Prefix search keeps typeahead fast while the reranker narrows the
        // bounded candidate set later in the search pipeline. Keep short
        // queries precise, but avoid over-constraining longer natural-language
        // queries into zero-result intersections.
        return tokens.joined(separator: tokens.count >= 4 ? " OR " : " ")
    }

    private func sanitizeTrigramFTS5Query(_ query: String) -> String {
        let cleaned = query
            .replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: "*", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return "" }
        return "\"\(cleaned)\""
    }

    private func normalizedSourceFilter(_ source: String?) -> String? {
        guard let source = source?.trimmingCharacters(in: .whitespacesAndNewlines), !source.isEmpty else {
            return nil
        }
        return source == "all" ? nil : source
    }

    private func expandedSearchQueries(_ query: String) -> [String] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var variants = [trimmed]
        let tokens = trimmed.split(separator: " ").map(String.init)
        for token in tokens {
            let replacements = lexicalDefenseReplacements(for: token) + kgAliasReplacements(for: token)
            for replacement in replacements where replacement != token {
                variants.append(replaceToken(token, with: replacement, in: trimmed))
            }
        }

        var seen = Set<String>()
        return variants.filter { seen.insert($0.lowercased()).inserted }
    }

    private func lexicalDefenseReplacements(for token: String) -> [String] {
        let normalized = token
            .trimmingCharacters(in: .punctuationCharacters.union(.whitespacesAndNewlines))
            .lowercased()
        return Self.lexicalDefenseReplacements[normalized] ?? []
    }

    private func kgAliasReplacements(for token: String) -> [String] {
        guard let db else { return [] }
        let normalized = normalizedAliasSurface(token)
        guard normalized.count >= 3 else { return [] }

        let sql = """
            SELECT name FROM kg_entities
            WHERE lower(replace(replace(replace(replace(name, '-', ''), '_', ''), '.', ''), ' ', '')) = ?
            UNION
            SELECT alias FROM kg_entity_aliases
            WHERE lower(replace(replace(replace(replace(alias, '-', ''), '_', ''), '.', ''), ' ', '')) = ?
            UNION
            SELECT e.name
            FROM kg_entity_aliases a
            JOIN kg_entities e ON e.id = a.entity_id
            WHERE lower(replace(replace(replace(replace(a.alias, '-', ''), '_', ''), '.', ''), ' ', '')) = ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }
        bindText(normalized, to: stmt, index: 1)
        bindText(normalized, to: stmt, index: 2)
        bindText(normalized, to: stmt, index: 3)

        var replacements: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let value = columnText(stmt, 0), !value.isEmpty {
                replacements.append(value)
            }
        }
        return replacements
    }

    private func normalizedAliasSurface(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: " ", with: "")
    }

    private func replaceToken(_ token: String, with replacement: String, in query: String) -> String {
        guard let range = query.range(of: token) else { return query }
        var copy = query
        copy.replaceSubrange(range, with: replacement)
        return copy
    }

    private func appendDeduped(
        _ rows: [[String: Any]],
        to results: inout [[String: Any]],
        seenChunkIDs: inout Set<String>,
        limit: Int
    ) {
        for row in rows {
            guard results.count < limit else { return }
            guard let chunkID = row["chunk_id"] as? String else { continue }
            if seenChunkIDs.insert(chunkID).inserted {
                results.append(row)
            }
        }
    }

    private func exactChunkIDSearchResult(
        query: String,
        limit: Int,
        project: String?,
        source: String?,
        tag: String?,
        importanceMin: Double?
    ) throws -> [[String: Any]]? {
        guard let db else { throw DBError.notOpen }
        guard limit > 0, !query.contains(where: { $0.isWhitespace }), query.contains("-") else {
            return nil
        }

        let sourceFilter = normalizedSourceFilter(source)
        var conditions = [
            "c.id = ?",
            "c.superseded_by IS NULL",
            "c.aggregated_into IS NULL",
            "c.archived_at IS NULL",
            "COALESCE(c.archived, 0) = 0",
            "COALESCE(c.status, 'active') = 'active'"
        ]
        if project != nil { conditions.append("(c.project = ? OR c.project IS NULL)") }
        if sourceFilter != nil { conditions.append("c.source = ?") }
        if tag != nil { conditions.append("c.tags LIKE ?") }
        if importanceMin != nil { conditions.append("c.importance >= ?") }

        let sql = """
            SELECT c.rowid, c.id, c.content, c.source_file, c.project, c.content_type, c.importance,
                   c.created_at, c.summary, c.tags, c.conversation_id, c.source
            FROM chunks c
            WHERE \(conditions.joined(separator: " AND "))
            LIMIT 1
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var paramIdx: Int32 = 1
        bindText(query, to: stmt, index: paramIdx)
        paramIdx += 1
        if let project {
            bindText(project, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let sourceFilter {
            bindText(sourceFilter, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let tag {
            bindText("%\"\(tag)\"%", to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let importanceMin {
            sqlite3_bind_double(stmt, paramIdx, importanceMin)
        }

        if sqlite3_step(stmt) == SQLITE_ROW {
            return [searchRow(from: stmt, score: 1.0)]
        }

        return try chunkExists(id: query) ? [] : nil
    }

    private func chunkExists(id: String) throws -> Bool {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT 1 FROM chunks WHERE id = ? LIMIT 1"
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(id, to: stmt, index: 1)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    private func searchRow(from stmt: OpaquePointer?, score: Double) -> [String: Any] {
        let rowID = sqlite3_column_int64(stmt, 0)
        return [
            "rowid": Int(rowID),
            "chunk_id": columnText(stmt, 1) as Any,
            "content": columnText(stmt, 2) as Any,
            "source_file": columnText(stmt, 3) as Any,
            "project": columnText(stmt, 4) as Any,
            "content_type": columnText(stmt, 5) as Any,
            "importance": sqlite3_column_double(stmt, 6),
            "created_at": columnText(stmt, 7) as Any,
            "summary": columnText(stmt, 8) as Any,
            "preview_text": Self.previewText(summary: columnText(stmt, 8), content: columnText(stmt, 2)),
            "tags": columnText(stmt, 9) as Any,
            "session_id": columnText(stmt, 10) as Any,
            "source": columnText(stmt, 11) as Any,
            "score": score
        ]
    }

    private func ensureInjectionEventColumns() throws {
        guard let db else { throw DBError.notOpen }
        guard (try? tableExists("injection_events")) == true else { return }
        let existingColumns = try tableColumns(name: "injection_events", on: db)
        if !existingColumns.contains("mode") {
            try execute("ALTER TABLE injection_events ADD COLUMN mode TEXT NOT NULL DEFAULT 'normal'")
        }
    }

    private func ensureChunkColumns() throws {
        guard let db else { throw DBError.notOpen }
        try execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL,
                details TEXT
            )
        """)
        let schemaMigrationColumns = try tableColumns(name: "schema_migrations", on: db)
        if !schemaMigrationColumns.contains("details") {
            try execute("ALTER TABLE schema_migrations ADD COLUMN details TEXT")
        }
        let existingColumns = try tableColumns(name: "chunks", on: db)
        let atomicColumns = ["brick_id", "source_uri", "status", "ingested_at", "topic_cluster"]
        let atomicMigrationApplied = try schemaMigrationApplied(name: "atomic_brick_chunks_v1", on: db)
        let needsAtomicBackfill = atomicColumns.contains { !existingColumns.contains($0) } || !atomicMigrationApplied
        if !existingColumns.contains("summary") {
            try execute("ALTER TABLE chunks ADD COLUMN summary TEXT")
        }
        if !existingColumns.contains("tags") {
            try execute("ALTER TABLE chunks ADD COLUMN tags TEXT DEFAULT '[]'")
        }
        if !existingColumns.contains("preview_text") {
            try execute("ALTER TABLE chunks ADD COLUMN preview_text TEXT")
        }
        if !existingColumns.contains("resolved_query") {
            try execute("ALTER TABLE chunks ADD COLUMN resolved_query TEXT")
        }
        if !existingColumns.contains("key_facts") {
            try execute("ALTER TABLE chunks ADD COLUMN key_facts TEXT")
        }
        if !existingColumns.contains("resolved_queries") {
            try execute("ALTER TABLE chunks ADD COLUMN resolved_queries TEXT")
        }
        if !existingColumns.contains("source") {
            try execute("ALTER TABLE chunks ADD COLUMN source TEXT DEFAULT 'claude_code'")
        }
        if !existingColumns.contains("source_file") {
            try execute("ALTER TABLE chunks ADD COLUMN source_file TEXT DEFAULT 'brainbar'")
        }
        if !existingColumns.contains("value_type") {
            try execute("ALTER TABLE chunks ADD COLUMN value_type TEXT")
        }
        if !existingColumns.contains("sender") {
            try execute("ALTER TABLE chunks ADD COLUMN sender TEXT")
        }
        if !existingColumns.contains("created_at") {
            try execute("ALTER TABLE chunks ADD COLUMN created_at TEXT")
            try execute("UPDATE chunks SET created_at = datetime('now') WHERE created_at IS NULL")
        }
        if !existingColumns.contains("enriched_at") {
            try execute("ALTER TABLE chunks ADD COLUMN enriched_at TEXT")
        }
        if !existingColumns.contains("enrich_status") {
            try execute("ALTER TABLE chunks ADD COLUMN enrich_status TEXT")
        }
        try execute("""
            UPDATE chunks
            SET enriched_at = NULL
            WHERE enriched_at IS NOT NULL
              AND TRIM(enriched_at) = ''
        """)
        try execute("""
            UPDATE chunks
            SET enrich_status = 'success'
            WHERE enriched_at IS NOT NULL
              AND TRIM(enriched_at) != ''
              AND enriched_at NOT LIKE 'skipped:%'
              AND enrich_status IS NULL
        """)
        try execute("""
            UPDATE chunks
            SET enrich_status = NULLIF(TRIM(substr(enriched_at, length('skipped:') + 1)), ''),
                enriched_at = NULL
            WHERE enriched_at LIKE 'skipped:%'
        """)
        if !existingColumns.contains("archived") {
            try execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
        }
        if !existingColumns.contains("superseded_by") {
            try execute("ALTER TABLE chunks ADD COLUMN superseded_by TEXT")
        }
        if !existingColumns.contains("archived_at") {
            try execute("ALTER TABLE chunks ADD COLUMN archived_at TEXT")
        }
        if !existingColumns.contains("aggregated_into") {
            try execute("ALTER TABLE chunks ADD COLUMN aggregated_into TEXT")
        }
        if !existingColumns.contains("brick_id") {
            try execute("ALTER TABLE chunks ADD COLUMN brick_id TEXT")
        }
        if !existingColumns.contains("source_uri") {
            try execute("ALTER TABLE chunks ADD COLUMN source_uri TEXT")
        }
        if !existingColumns.contains("status") {
            try execute("ALTER TABLE chunks ADD COLUMN status TEXT DEFAULT 'active'")
        }
        if !existingColumns.contains("ingested_at") {
            try execute("ALTER TABLE chunks ADD COLUMN ingested_at INTEGER")
        }
        if !existingColumns.contains("topic_cluster") {
            try execute("ALTER TABLE chunks ADD COLUMN topic_cluster TEXT")
        }
        if needsAtomicBackfill {
            try execute("UPDATE chunks SET brick_id = id WHERE brick_id IS NULL")
            if existingColumns.contains("source_file") {
                try execute("UPDATE chunks SET source_uri = source_file WHERE source_uri IS NULL AND source_file IS NOT NULL")
            } else {
                try execute("UPDATE chunks SET source_uri = 'brainbar' WHERE source_uri IS NULL")
            }
            try execute("""
                UPDATE chunks
                SET ingested_at = COALESCE(
                    CAST(strftime('%s', created_at) AS INTEGER),
                    CAST(strftime('%s', 'now') AS INTEGER)
                )
                WHERE ingested_at IS NULL
            """)
            try execute("""
                UPDATE chunks
                SET status = CASE
                    WHEN COALESCE(archived, 0) = 1
                      OR value_type = 'ARCHIVED'
                      OR archived_at IS NOT NULL
                        THEN 'archived'
                    WHEN superseded_by IS NOT NULL
                      OR aggregated_into IS NOT NULL
                        THEN 'superseded'
                    ELSE 'active'
                END
                WHERE status IS NULL
                   OR status NOT IN ('active', 'superseded', 'archived')
                   OR (
                        status = 'active'
                        AND (
                            COALESCE(archived, 0) = 1
                            OR value_type = 'ARCHIVED'
                            OR archived_at IS NOT NULL
                            OR superseded_by IS NOT NULL
                            OR aggregated_into IS NOT NULL
                        )
                    )
            """)
            try execute("""
                INSERT OR REPLACE INTO schema_migrations(name, applied_at)
                VALUES ('atomic_brick_chunks_v1', datetime('now'))
            """)
        }
    }

    private func ensureKGEntityColumns() throws {
        guard let db else { throw DBError.notOpen }
        try ensureKGEntityTable()
        let existingColumns = try tableColumns(name: "kg_entities", on: db)
        if !existingColumns.contains("description") {
            try execute("ALTER TABLE kg_entities ADD COLUMN description TEXT")
        }
        if !existingColumns.contains("importance") {
            try execute("ALTER TABLE kg_entities ADD COLUMN importance REAL DEFAULT 0.5")
        }
        if !existingColumns.contains("valid_until") {
            try execute("ALTER TABLE kg_entities ADD COLUMN valid_until TEXT")
        }
        if !existingColumns.contains("expired_at") {
            try execute("ALTER TABLE kg_entities ADD COLUMN expired_at TEXT")
        }
    }

    private func ensureKGEntityTable() throws {
        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                description TEXT,
                importance REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(entity_type, name)
            )
        """)
    }

    private func ensureKGRelationColumns() throws {
        guard let db else { throw DBError.notOpen }
        try ensureKGRelationTable()
        let existingColumns = try tableColumns(name: "kg_relations", on: db)
        if !existingColumns.contains("valid_from") {
            try execute("ALTER TABLE kg_relations ADD COLUMN valid_from TEXT")
        }
        if !existingColumns.contains("valid_until") {
            try execute("ALTER TABLE kg_relations ADD COLUMN valid_until TEXT")
        }
        if !existingColumns.contains("expired_at") {
            try execute("ALTER TABLE kg_relations ADD COLUMN expired_at TEXT")
        }
        try execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_validity ON kg_relations(valid_from, valid_until)")
        try execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_expired ON kg_relations(expired_at)")
    }

    private func ensureKGRelationTable() throws {
        try execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
    }

    private func ensureKGEntityAliasTable() throws {
        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_aliases (
                alias TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                alias_type TEXT DEFAULT 'name',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                valid_from TEXT,
                valid_to TEXT,
                PRIMARY KEY (alias, entity_id)
            )
        """)
        try execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_alias_lookup
            ON kg_entity_aliases(alias COLLATE NOCASE)
        """)
        try execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_alias_entity
            ON kg_entity_aliases(entity_id)
        """)
    }

    private func ensurePreviewTextTriggers() throws {
        try execute("DROP TRIGGER IF EXISTS chunks_preview_text_insert")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_preview_text_insert
            AFTER INSERT ON chunks
            WHEN new.preview_text IS NULL OR trim(new.preview_text) = ''
            BEGIN
                UPDATE chunks
                SET preview_text = \(Self.previewExpression)
                WHERE rowid = new.rowid;
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_preview_text_update")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_preview_text_update
            AFTER UPDATE OF content, summary ON chunks
            BEGIN
                UPDATE chunks
                SET preview_text = \(Self.previewExpression)
                WHERE rowid = new.rowid;
            END
        """)
    }

    private func rebuildFTSTableIfNeeded() throws {
        let needsRebuild = try ftsTableNeedsRebuild()
        if needsRebuild {
            try execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
            try execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
            try execute("DROP TRIGGER IF EXISTS chunks_fts_update")
            try execute("DROP TABLE IF EXISTS chunks_fts")
        }

        try execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                \(Self.ftsColumns),
                \(Self.ftsOptions)
            )
        """)
        let ftsCount = try countRows(in: "chunks_fts")
        let chunkCount = try countRows(in: "chunks")
        let needsBackfill = needsRebuild || ftsCount != chunkCount
        if needsBackfill {
            try execute("DELETE FROM chunks_fts")
            try execute("""
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
            """)
        }

        try execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_update")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update
            AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            END
        """)
    }

    private func rebuildTrigramFTSTable() throws {
        if try trigramFTSTableNeedsRebuild() {
            try execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_insert")
            try execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_delete")
            try execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_update")
            try execute("DROP TABLE IF EXISTS chunks_fts_trigram")
        }
        try ensureTrigramFTSSchemaAndTriggers()
        try execute("DELETE FROM chunks_fts_trigram")
        try backfillTrigramFTSTable()
    }

    private func ensureTrigramFTSSchemaAndTriggers() throws {
        try execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
                \(Self.ftsColumns),
                \(Self.trigramFTSOptions)
            )
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_insert")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_delete")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts_trigram WHERE chunk_id = old.id;
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_update")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_update
            AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
                DELETE FROM chunks_fts_trigram WHERE chunk_id = old.id;
                INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            END
        """)
    }

    private func backfillTrigramFTSTable() throws {
        try execute("""
            INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """)
    }

    func triggerTrigramRebuild(
        batchSize requestedBatchSize: Int = 1_000,
        shouldCancel: () -> Bool = { false },
        progress: (TrigramMaintenanceProgress) -> Void = { _ in }
    ) throws -> TrigramMaintenanceProgress {
        try ensureTrigramFTSSchemaAndTriggers()
        let batchSize = Self.normalizedTrigramMaintenanceBatchSize(requestedBatchSize)
        let total = try countRows(in: "chunks")
        let maxChunkRowID = try maxChunkRowID()
        var lastProcessedRowID: Int64 = 0
        var processed = 0
        let startedAt = Date()

        while processed < total,
              let batch = try nextChunkBatch(
                after: lastProcessedRowID,
                maxRowID: maxChunkRowID,
                batchSize: batchSize
              ) {
            if shouldCancel() {
                let cancelled = TrigramMaintenanceProgress(
                    state: .cancelled,
                    processed: processed,
                    total: total,
                    etaSeconds: nil
                )
                progress(cancelled)
                return cancelled
            }

            try withImmediateTransaction {
                try executeUpdate(
                    """
                    DELETE FROM chunks_fts_trigram
                    WHERE chunk_id IN (
                        SELECT id FROM chunks WHERE rowid > ? AND rowid <= ?
                    )
                    """,
                    binds: { stmt in
                        sqlite3_bind_int64(stmt, 1, lastProcessedRowID)
                        sqlite3_bind_int64(stmt, 2, batch.upperRowID)
                    }
                )
                try executeUpdate(
                    """
                    INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                    SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id
                    FROM chunks
                    WHERE rowid > ? AND rowid <= ?
                    ORDER BY rowid ASC
                    """,
                    binds: { stmt in
                        sqlite3_bind_int64(stmt, 1, lastProcessedRowID)
                        sqlite3_bind_int64(stmt, 2, batch.upperRowID)
                    }
                )
            }

            lastProcessedRowID = batch.upperRowID
            processed = min(processed + batch.rowCount, total)
            let running = TrigramMaintenanceProgress(
                state: .running,
                processed: processed,
                total: total,
                etaSeconds: Self.estimatedSecondsRemaining(
                    startedAt: startedAt,
                    processed: processed,
                    total: total
                )
            )
            progress(running)
        }

        try execute("ANALYZE chunks_fts_trigram", retries: 3)
        let done = TrigramMaintenanceProgress(
            state: .done,
            processed: total,
            total: total,
            etaSeconds: 0
        )
        progress(done)
        return done
    }

    private func maxChunkRowID() throws -> Int64 {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT COALESCE(MAX(rowid), 0) FROM chunks", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_ROW else { throw DBError.step(stepRC) }
        return sqlite3_column_int64(stmt, 0)
    }

    static func normalizedTrigramMaintenanceBatchSize(_ requestedBatchSize: Int) -> Int {
        min(max(1, requestedBatchSize), maximumTrigramMaintenanceBatchSize)
    }

    private func nextChunkBatch(after rowID: Int64, maxRowID: Int64, batchSize: Int) throws -> (upperRowID: Int64, rowCount: Int)? {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(
            db,
            """
            SELECT COALESCE(MAX(rowid), 0), COUNT(*)
            FROM (
                SELECT rowid
                FROM chunks
                WHERE rowid > ? AND rowid <= ?
                ORDER BY rowid ASC
                LIMIT ?
            )
            """,
            -1,
            &stmt,
            nil
        )
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, rowID)
        sqlite3_bind_int64(stmt, 2, maxRowID)
        sqlite3_bind_int(stmt, 3, Int32(batchSize))
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        let upperRowID = sqlite3_column_int64(stmt, 0)
        let rowCount = Int(sqlite3_column_int(stmt, 1))
        guard rowCount > 0 else { return nil }
        return (upperRowID, rowCount)
    }

    private static func estimatedSecondsRemaining(startedAt: Date, processed: Int, total: Int) -> Double? {
        guard processed > 0, total > processed else { return nil }
        let elapsed = Date().timeIntervalSince(startedAt)
        let rate = Double(processed) / max(elapsed, 0.001)
        return Double(total - processed) / rate
    }

    private func rebuildTrigramFTSTableIfNeeded() throws {
        let status = try trigramFTSStatus()
        let decision = Self.trigramStartupRepairDecision(
            tableExists: status.tableExists,
            schemaIsValid: status.schemaIsValid,
            chunkCount: status.chunkCount,
            trigramCount: status.trigramCount
        )

        switch decision {
        case .noRepairNeeded:
            try ensureTrigramFTSSchemaAndTriggers()
        case .rebuildSynchronously:
            try rebuildTrigramFTSTable()
        case .skipBackfill:
            try ensureTrigramFTSSchemaAndTriggers()
            NSLog(
                "[BrainBar] Skipping synchronous trigram FTS backfill at startup — chunks=%d trigram=%d. Run explicit maintenance to backfill.",
                status.chunkCount,
                status.trigramCount
            )
        }
    }

    static func trigramStartupRepairDecision(
        tableExists: Bool,
        schemaIsValid: Bool,
        chunkCount: Int,
        trigramCount: Int
    ) -> TrigramStartupRepairDecision {
        if !tableExists || !schemaIsValid {
            return .rebuildSynchronously
        }
        if chunkCount == trigramCount {
            return .noRepairNeeded
        }
        if chunkCount <= synchronousTrigramBackfillChunkLimit {
            return .rebuildSynchronously
        }
        return .skipBackfill
    }

    private func trigramFTSStatus() throws -> (tableExists: Bool, schemaIsValid: Bool, chunkCount: Int, trigramCount: Int) {
        guard let db else { throw DBError.notOpen }
        guard let sql = try sqliteMasterSQL(name: "chunks_fts_trigram", on: db) else {
            let chunkCount = try countRows(in: "chunks")
            return (false, false, chunkCount, 0)
        }
        let normalizedSQL = sql.lowercased()
        let schemaIsValid = normalizedSQL.contains("tokenize='trigram'") &&
            normalizedSQL.contains("summary") &&
            normalizedSQL.contains("resolved_query") &&
            normalizedSQL.contains("key_facts") &&
            normalizedSQL.contains("resolved_queries")

        let chunkCount = try countRows(in: "chunks")
        let trigramCount = try countRows(in: "chunks_fts_trigram")
        return (true, schemaIsValid, chunkCount, trigramCount)
    }

    private func countRows(in tableName: String) throws -> Int {
        guard let db else { throw DBError.notOpen }
        let allowedTables = ["chunks", "chunks_fts", "chunks_fts_trigram"]
        guard allowedTables.contains(tableName) else { throw DBError.exec(SQLITE_ERROR, "invalid count table") }

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM \(tableName)", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    private func ftsTableNeedsRebuild() throws -> Bool {
        guard let db else { throw DBError.notOpen }
        guard let sql = try sqliteMasterSQL(name: "chunks_fts", on: db) else { return true }
        let normalizedSQL = sql.lowercased()
        return !normalizedSQL.contains("prefix='2 3 4'") ||
            !normalizedSQL.contains("tokenize='unicode61 remove_diacritics 2'") ||
            !normalizedSQL.contains("summary") ||
            !normalizedSQL.contains("resolved_query") ||
            !normalizedSQL.contains("key_facts") ||
            !normalizedSQL.contains("resolved_queries")
    }

    private func trigramFTSTableNeedsRebuild() throws -> Bool {
        guard let db else { throw DBError.notOpen }
        guard let sql = try sqliteMasterSQL(name: "chunks_fts_trigram", on: db) else { return false }
        let normalizedSQL = sql.lowercased()
        return !normalizedSQL.contains("tokenize='trigram'") ||
            !normalizedSQL.contains("summary") ||
            !normalizedSQL.contains("resolved_query") ||
            !normalizedSQL.contains("key_facts") ||
            !normalizedSQL.contains("resolved_queries")
    }

    private func backfillPreviewText() throws {
        try execute("""
            UPDATE chunks
            SET preview_text = \(Self.previewExpression)
            WHERE preview_text IS NULL OR trim(preview_text) = ''
        """)
    }

    private func refreshSearchStatistics() throws {
        try execute("ANALYZE chunks")
        try execute("ANALYZE chunks_fts")
        try execute("ANALYZE chunks_fts_trigram")
    }

    private func refreshSearchStatisticsBestEffort() {
        do {
            try refreshSearchStatistics()
        } catch {
            NSLog("[BrainBar] Non-fatal search statistics refresh failure: %@", String(describing: error))
        }
    }

    private func ensurePendingStoreQueueIndex() throws {
        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_brainbar_queue_id
            ON chunks(json_extract(metadata, '$.brainbar_queue_id'))
            WHERE json_valid(metadata)
              AND json_extract(metadata, '$.brainbar_queue_id') IS NOT NULL
        """)
    }

    private func pendingStorePath() -> URL {
        Self.pendingStorePath(forDBPath: path)
    }

    private static func pendingStorePath(forDBPath dbPath: String) -> URL {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_PENDING_STORES_PATH"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath: dbPath).deletingLastPathComponent().appendingPathComponent("pending-stores.jsonl")
    }

    private static func appendPendingStoreLine(_ line: Data, to path: URL) throws {
        try Self.withPendingStoreProcessLock(for: path) {
            let fd = open(path.path, O_RDWR | O_CREAT | O_APPEND, 0o600)
            guard fd >= 0 else {
                let message = String(cString: strerror(errno))
                throw DBError.exec(SQLITE_CANTOPEN, "failed to open pending store queue for append: \(message)")
            }
            defer { Darwin.close(fd) }
            try Self.enforcePendingStoreFileMode(fd: fd, path: path)

            let existingLineCount = Self.pendingStoreLines(from: Self.readPendingStoreData(at: path) ?? Data()).count
            let maxLines = Self.pendingStoreMaxLines()
            guard existingLineCount < maxLines else {
                throw DBError.exec(
                    SQLITE_FULL,
                    "pending store queue is full (\(existingLineCount)/\(maxLines) lines); refusing to append another fallback item"
                )
            }

            try Self.writeAll(line, to: fd, context: "append pending store queue line")
        }
    }

    private static func readPendingStoreData(at path: URL) -> Data? {
        try? Data(contentsOf: path)
    }

    func pendingStoreQueueSnapshot() -> PendingStoreQueueSnapshot {
        let path = pendingStorePath()
        Self.pendingStoreFileLock.lock()
        defer { Self.pendingStoreFileLock.unlock() }

        do {
            return try Self.withPendingStoreProcessLock(for: path) {
                guard FileManager.default.fileExists(atPath: path.path),
                      let data = Self.readPendingStoreData(at: path) else {
                    return PendingStoreQueueSnapshot(depth: 0, oldestQueuedAt: nil)
                }

                let lines = Self.pendingStoreLines(from: data)
                guard !lines.isEmpty else {
                    return PendingStoreQueueSnapshot(depth: 0, oldestQueuedAt: nil)
                }

                let decoder = JSONDecoder()
                let oldest = lines.compactMap { line -> Date? in
                    guard let item = try? decoder.decode(PendingStoreItem.self, from: line),
                          let queuedAt = item.queuedAt else {
                        return nil
                    }
                    return Self.parsePendingStoreQueuedAt(queuedAt)
                }.min()

                return PendingStoreQueueSnapshot(depth: lines.count, oldestQueuedAt: oldest)
            }
        } catch {
            NSLog("[BrainBar] Failed to inspect pending stores queue: %@", String(describing: error))
            return PendingStoreQueueSnapshot(depth: 0, oldestQueuedAt: nil)
        }
    }

    private static func pendingStoreLines(from data: Data) -> [Data] {
        var lines: [Data] = []
        var start = data.startIndex

        func appendLine(_ range: Range<Data.Index>) {
            guard !range.isEmpty else { return }
            var line = Data(data[range])
            if line.last == 0x0D {
                line.removeLast()
            }
            guard !line.isEmpty else { return }
            lines.append(line)
        }

        for index in data.indices where data[index] == 0x0A {
            appendLine(start..<index)
            start = data.index(after: index)
        }

        if start < data.endIndex {
            appendLine(start..<data.endIndex)
        }

        return lines
    }

    private static func parsePendingStoreQueuedAt(_ rawValue: String) -> Date? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let date = ISO8601DateFormatter().date(from: trimmed) {
            return date
        }
        if let epoch = TimeInterval(trimmed) {
            return Date(timeIntervalSince1970: epoch)
        }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter.date(from: trimmed)
    }

    private static func pendingStoreMaxLines() -> Int {
        let rawValue = ProcessInfo.processInfo.environment[pendingStoreMaxLinesEnv]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let rawValue, let parsed = Int(rawValue), parsed > 0 else {
            return defaultPendingStoreMaxLines
        }
        return parsed
    }

    private static func pendingStoreLockPath(for path: URL) -> URL {
        path.deletingLastPathComponent().appendingPathComponent(".\(path.lastPathComponent).lock")
    }

    private static func withPendingStoreProcessLock<T>(for path: URL, _ body: () throws -> T) throws -> T {
        let lockPath = pendingStoreLockPath(for: path)
        try FileManager.default.createDirectory(
            at: lockPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let fd = open(lockPath.path, O_RDWR | O_CREAT, 0o600)
        guard fd >= 0 else {
            let message = String(cString: strerror(errno))
            throw DBError.exec(SQLITE_CANTOPEN, "failed to open pending store queue lock: \(message)")
        }
        defer { Darwin.close(fd) }
        try enforcePendingStoreFileMode(fd: fd, path: lockPath)

        guard flock(fd, LOCK_EX) == 0 else {
            let message = String(cString: strerror(errno))
            throw DBError.exec(SQLITE_IOERR, "failed to lock pending store queue: \(message)")
        }
        defer { flock(fd, LOCK_UN) }

        return try body()
    }

    private static func writeAll(_ data: Data, to fd: Int32, context: String) throws {
        try data.withUnsafeBytes { rawBuffer in
            guard var baseAddress = rawBuffer.baseAddress else { return }
            var remaining = rawBuffer.count

            while remaining > 0 {
                let written = Darwin.write(fd, baseAddress, remaining)
                if written < 0 {
                    if errno == EINTR {
                        continue
                    }
                    let message = String(cString: strerror(errno))
                    throw DBError.exec(SQLITE_IOERR, "failed to \(context): \(message)")
                }
                guard written > 0 else {
                    throw DBError.exec(SQLITE_IOERR, "failed to \(context): write returned 0")
                }

                remaining -= written
                baseAddress = baseAddress.advanced(by: written)
            }
        }
    }

    private static func writePrivateAtomic(_ data: Data, to path: URL) throws {
        let directory = path.deletingLastPathComponent()
        let temporaryPath = directory.appendingPathComponent(".\(path.lastPathComponent).\(UUID().uuidString).tmp")
        let fd = open(temporaryPath.path, O_WRONLY | O_CREAT | O_EXCL | O_TRUNC, 0o600)
        guard fd >= 0 else {
            let message = String(cString: strerror(errno))
            throw DBError.exec(SQLITE_CANTOPEN, "failed to create private pending store queue temp file: \(message)")
        }

        do {
            defer { Darwin.close(fd) }
            try enforcePendingStoreFileMode(fd: fd, path: temporaryPath)
            try writeAll(data, to: fd, context: "write pending store queue temp file")
            guard fsync(fd) == 0 else {
                let message = String(cString: strerror(errno))
                throw DBError.exec(SQLITE_IOERR, "failed to sync pending store queue temp file: \(message)")
            }
        } catch {
            try? FileManager.default.removeItem(at: temporaryPath)
            throw error
        }

        guard rename(temporaryPath.path, path.path) == 0 else {
            let message = String(cString: strerror(errno))
            try? FileManager.default.removeItem(at: temporaryPath)
            throw DBError.exec(SQLITE_IOERR, "failed to replace pending store queue atomically: \(message)")
        }
    }

    private static func enforcePendingStoreFileMode(fd: Int32? = nil, path: URL) throws {
        if let fd {
            guard fchmod(fd, 0o600) == 0 else {
                let message = String(cString: strerror(errno))
                throw DBError.exec(SQLITE_IOERR, "failed to set pending store queue permissions: \(message)")
            }
            return
        }

        do {
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        } catch {
            throw DBError.exec(SQLITE_IOERR, "failed to set pending store queue permissions: \(error)")
        }
    }

    private static func storeMetadataJSON(queueID: String?) -> String {
        guard let queueID = normalizedQueueID(queueID) else { return "{}" }
        let payload = ["brainbar_queue_id": queueID]
        guard let data = try? JSONEncoder().encode(payload),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    private static func normalizedQueueID(_ queueID: String?) -> String? {
        guard let queueID = queueID?.trimmingCharacters(in: .whitespacesAndNewlines),
              !queueID.isEmpty else {
            return nil
        }
        return queueID
    }

    private static func pendingStoreQueueID(for item: PendingStoreItem, lineIndex: Int) -> String {
        if let queueID = normalizedQueueID(item.queueID) {
            return queueID
        }
        return deterministicPendingStoreQueueID(
            content: item.content,
            tags: item.tags,
            importance: item.importance,
            source: item.source,
            lineIndex: lineIndex
        )
    }

    private static func pendingStoreReplayLine(for item: PendingStoreItem, queueID: String, chunkID: String) -> Data? {
        let replayItem = PendingStoreItem(
            content: item.content,
            tags: item.tags,
            importance: item.importance,
            source: item.source,
            project: item.project,
            queueID: queueID,
            queuedAt: item.queuedAt,
            createdAt: item.createdAt ?? item.queuedAt,
            chunkID: chunkID
        )
        return try? JSONEncoder().encode(replayItem)
    }

    private static func deterministicPendingStoreQueueID(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        lineIndex: Int
    ) -> String {
        var hash: UInt64 = 0xcbf29ce484222325

        func mix(_ byte: UInt8) {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }

        func mix(_ data: Data) {
            for byte in data {
                mix(byte)
            }
        }

        func mix(length: Int) {
            var value = UInt64(length).littleEndian
            withUnsafeBytes(of: &value) { bytes in
                mix(Data(bytes))
            }
        }

        func mix(_ string: String) {
            let data = Data(string.utf8)
            mix(length: data.count)
            mix(data)
        }

        mix(content)
        mix(length: tags.count)
        for tag in tags {
            mix(tag)
        }
        var encodedImportance = Int64(importance).littleEndian
        withUnsafeBytes(of: &encodedImportance) { bytes in
            mix(Data(bytes))
        }
        mix(source)
        var encodedLineIndex = Int64(lineIndex).littleEndian
        withUnsafeBytes(of: &encodedLineIndex) { bytes in
            mix(Data(bytes))
        }

        return String(format: "brainbar-pending-%016llx", hash)
    }

    private static func isRetryableQueueErrorCode(_ rc: Int32) -> Bool {
        isRetryableStoreErrorCode(rc)
    }

    private static func isRetryableStoreErrorCode(_ rc: Int32) -> Bool {
        let primaryCode = rc & 0xFF
        return primaryCode == SQLITE_BUSY || primaryCode == SQLITE_LOCKED || primaryCode == SQLITE_READONLY
    }

    private func hasStoredQueuedItem(queueID: String) throws -> Bool {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            SELECT 1
            FROM chunks
            WHERE json_valid(metadata)
              AND json_extract(metadata, '$.brainbar_queue_id') = ?
            LIMIT 1
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(queueID, to: stmt, index: 1)
        let stepRC = sqlite3_step(stmt)
        switch stepRC {
        case SQLITE_ROW:
            return true
        case SQLITE_DONE:
            return false
        default:
            throw DBError.step(stepRC)
        }
    }

    private func rewritePendingStoreFile(path: URL, snapshot: Data, remainingLines: [Data]) {
        do {
            let current = (try? Data(contentsOf: path)) ?? Data()
            let appendedLines: [Data]
            if current.count >= snapshot.count && current.prefix(snapshot.count).elementsEqual(snapshot) {
                appendedLines = Self.pendingStoreLines(from: Data(current.dropFirst(snapshot.count)))
            } else {
                NSLog("[BrainBar] Skipping pending store queue rewrite because the file changed outside the snapshot")
                return
            }

            let finalLines = remainingLines + appendedLines
            guard !finalLines.isEmpty else {
                try? FileManager.default.removeItem(at: path)
                return
            }

            var data = Data()
            for line in finalLines {
                data.append(line)
                data.append(0x0A)
            }
            try Self.writePrivateAtomic(data, to: path)
            try Self.enforcePendingStoreFileMode(path: path)
        } catch {
            NSLog("[BrainBar] Failed to rewrite pending stores queue: %@", String(describing: error))
        }
    }

    private func tableColumns(name: String, on db: OpaquePointer) throws -> Set<String> {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA table_info(\(name))", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var columns: Set<String> = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let columnName = columnText(stmt, 1) {
                columns.insert(columnName)
            }
        }
        return columns
    }

    private func schemaMigrationApplied(name: String, on db: OpaquePointer) throws -> Bool {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT 1 FROM schema_migrations WHERE name = ? LIMIT 1", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        bindText(name, to: stmt, index: 1)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    private func sqliteMasterSQL(name: String, on db: OpaquePointer) throws -> String? {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT sql FROM sqlite_master WHERE name = ?", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        bindText(name, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return columnText(stmt, 0)
    }

    private static func previewText(summary: String?, content: String?) -> String {
        let preferred = (summary?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false) ? summary! : (content ?? "")
        let flattened = preferred
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String(flattened.prefix(220))
    }

    private static func timestamp() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    // MARK: - brain_tags: list unique tags with counts

    func listTags(query: String? = nil, limit: Int = 50) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        
        // Use SQL aggregation for better performance with large datasets.
        // json_each extracts individual tags from the JSON array.
        let sql: String
        if query != nil {
            sql = """
                SELECT LOWER(TRIM(json_each.value)) AS tag, COUNT(*) AS count
                FROM chunks, json_each(chunks.tags)
                WHERE json_each.type = 'text'
                  AND LOWER(TRIM(json_each.value)) LIKE ?
                  AND TRIM(json_each.value) != ''
                GROUP BY LOWER(TRIM(json_each.value))
                ORDER BY count DESC
                LIMIT ?
            """
        } else {
            sql = """
                SELECT LOWER(TRIM(json_each.value)) AS tag, COUNT(*) AS count
                FROM chunks, json_each(chunks.tags)
                WHERE json_each.type = 'text'
                  AND TRIM(json_each.value) != ''
                GROUP BY LOWER(TRIM(json_each.value))
                ORDER BY count DESC
                LIMIT ?
            """
        }
        
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        
        var paramIdx: Int32 = 1
        if let query {
            bindText("%\(query.lowercased())%", to: stmt, index: paramIdx)
            paramIdx += 1
        }
        sqlite3_bind_int(stmt, paramIdx, Int32(limit))

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let tag = columnText(stmt, 0), !tag.isEmpty else { continue }
            let count = Int(sqlite3_column_int(stmt, 1))
            results.append(["tag": tag as Any, "count": count as Any])
        }
        return results
    }

    // MARK: - brain_update: update chunk importance/tags

    func updateChunk(id: String, importance: Int? = nil, tags: [String]? = nil) throws {
        guard db != nil else { throw DBError.notOpen }
        var rowsChanged = 0

        if let importance {
            let sql = "UPDATE chunks SET importance = ? WHERE id = ?"
            try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
                sqlite3_bind_int(stmt, 1, Int32(importance))
                bindText(id, to: stmt, index: 2)
            }
            rowsChanged += Int(sqlite3_changes(db))
        }

        if let tags {
            let tagsJSON = try encodeJSON(tags)
            let sql = "UPDATE chunks SET tags = ? WHERE id = ?"
            try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
                bindText(tagsJSON, to: stmt, index: 1)
                bindText(id, to: stmt, index: 2)
            }
            rowsChanged += Int(sqlite3_changes(db))
        }

        if rowsChanged == 0 {
            throw DBError.noResult
        }
    }

    // MARK: - brain_expand: get chunk + surrounding session context

    func expandChunk(id: String, before: Int = 3, after: Int = 3) throws -> [String: Any] {
        guard let db else { throw DBError.notOpen }
        let boundedBefore = min(max(before, 0), Self.maximumConversationContextPerSide)
        let boundedAfter = min(max(after, 0), Self.maximumConversationContextPerSide)
        let contentLimit = Int32(Self.maximumConversationContentCharacters)
        let chunkColumns = try tableColumns(name: "chunks", on: db)
        let senderSelect = chunkColumns.contains("sender") ? "sender" : "NULL AS sender"

        // Get the target chunk with its session_id and rowid
        let targetSQL = "SELECT rowid, id, substr(content, 1, ?), conversation_id, project, content_type, \(senderSelect), importance, created_at, summary, tags FROM chunks WHERE id = ?"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, targetSQL, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, contentLimit)
        bindText(id, to: stmt, index: 2)
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }

        let targetRowID = sqlite3_column_int64(stmt, 0)
        let sessionId = columnText(stmt, 3) ?? ""
        let target: [String: Any] = [
            "chunk_id": columnText(stmt, 1) as Any,
            "content": columnText(stmt, 2) as Any,
            "session_id": sessionId,
            "project": columnText(stmt, 4) as Any,
            "content_type": columnText(stmt, 5) as Any,
            "sender": columnText(stmt, 6) as Any,
            "importance": sqlite3_column_double(stmt, 7),
            "created_at": columnText(stmt, 8) as Any,
            "summary": columnText(stmt, 9) as Any,
            "tags": columnText(stmt, 10) as Any
        ]

        // Get surrounding chunks from same session using two separate queries
        var beforeContext: [[String: Any]] = []
        var afterContext: [[String: Any]] = []

        // Before chunks (reverse order, then flip)
        let beforeSQL = "SELECT id, substr(content, 1, ?), content_type, \(senderSelect), importance, created_at, summary FROM chunks WHERE conversation_id = ? AND rowid < ? ORDER BY rowid DESC LIMIT ?"
        var beforeStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, beforeSQL, -1, &beforeStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(beforeStmt) }
            sqlite3_bind_int(beforeStmt, 1, contentLimit)
            bindText(sessionId, to: beforeStmt, index: 2)
            sqlite3_bind_int64(beforeStmt, 3, targetRowID)
            sqlite3_bind_int(beforeStmt, 4, Int32(boundedBefore))
            var beforeChunks: [[String: Any]] = []
            while sqlite3_step(beforeStmt) == SQLITE_ROW {
                beforeChunks.append([
                    "chunk_id": columnText(beforeStmt, 0) as Any,
                    "content": columnText(beforeStmt, 1) as Any,
                    "content_type": columnText(beforeStmt, 2) as Any,
                    "sender": columnText(beforeStmt, 3) as Any,
                    "importance": sqlite3_column_double(beforeStmt, 4),
                    "created_at": columnText(beforeStmt, 5) as Any,
                    "summary": columnText(beforeStmt, 6) as Any
                ])
            }
            beforeContext = beforeChunks.reversed()
        }

        // After chunks
        let afterSQL = "SELECT id, substr(content, 1, ?), content_type, \(senderSelect), importance, created_at, summary FROM chunks WHERE conversation_id = ? AND rowid > ? ORDER BY rowid ASC LIMIT ?"
        var afterStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, afterSQL, -1, &afterStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(afterStmt) }
            sqlite3_bind_int(afterStmt, 1, contentLimit)
            bindText(sessionId, to: afterStmt, index: 2)
            sqlite3_bind_int64(afterStmt, 3, targetRowID)
            sqlite3_bind_int(afterStmt, 4, Int32(boundedAfter))
            while sqlite3_step(afterStmt) == SQLITE_ROW {
                afterContext.append([
                    "chunk_id": columnText(afterStmt, 0) as Any,
                    "content": columnText(afterStmt, 1) as Any,
                    "content_type": columnText(afterStmt, 2) as Any,
                    "sender": columnText(afterStmt, 3) as Any,
                    "importance": sqlite3_column_double(afterStmt, 4),
                    "created_at": columnText(afterStmt, 5) as Any,
                    "summary": columnText(afterStmt, 6) as Any
                ])
            }
        }

        return [
            "target": target,
            "before_context": beforeContext,
            "after_context": afterContext,
            "context": beforeContext + afterContext
        ]
    }

    func expandedConversation(id: String, before: Int = 3, after: Int = 3) throws -> ExpandedConversation {
        let payload = try expandChunk(id: id, before: before, after: after)
        guard let targetPayload = payload["target"] as? [String: Any] else {
            throw DBError.noResult
        }

        let target = ConversationChunk(
            chunkID: targetPayload["chunk_id"] as? String ?? "",
            content: targetPayload["content"] as? String ?? "",
            contentType: targetPayload["content_type"] as? String ?? "",
            sender: targetPayload["sender"] as? String ?? "",
            importance: targetPayload["importance"] as? Double ?? 0,
            createdAt: targetPayload["created_at"] as? String ?? "",
            summary: targetPayload["summary"] as? String ?? "",
            isTarget: true
        )

        let beforeContext = ((payload["before_context"] as? [[String: Any]]) ?? []).map { item in
            ConversationChunk(
                chunkID: item["chunk_id"] as? String ?? "",
                content: item["content"] as? String ?? "",
                contentType: item["content_type"] as? String ?? "",
                sender: item["sender"] as? String ?? "",
                importance: item["importance"] as? Double ?? 0,
                createdAt: item["created_at"] as? String ?? "",
                summary: item["summary"] as? String ?? "",
                isTarget: false
            )
        }
        let afterContext = ((payload["after_context"] as? [[String: Any]]) ?? []).map { item in
            ConversationChunk(
                chunkID: item["chunk_id"] as? String ?? "",
                content: item["content"] as? String ?? "",
                contentType: item["content_type"] as? String ?? "",
                sender: item["sender"] as? String ?? "",
                importance: item["importance"] as? Double ?? 0,
                createdAt: item["created_at"] as? String ?? "",
                summary: item["summary"] as? String ?? "",
                isTarget: false
            )
        }

        return ExpandedConversation(target: target, entries: beforeContext + [target] + afterContext)
    }

    // MARK: - brain_entity: insert + lookup entities

    func insertEntity(id: String, type: String, name: String, metadata: String = "{}") throws {
        guard let db else { throw DBError.notOpen }
        let sql = "INSERT OR REPLACE INTO kg_entities (id, entity_type, name, metadata) VALUES (?, ?, ?, ?)"
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(id, to: stmt, index: 1)
            bindText(type, to: stmt, index: 2)
            bindText(name, to: stmt, index: 3)
            bindText(metadata, to: stmt, index: 4)
        }
    }

    func insertRelation(sourceId: String, targetId: String, relationType: String) throws {
        guard let db else { throw DBError.notOpen }
        let relId = "\(sourceId)-\(relationType)-\(targetId)"
        let sql = "INSERT OR REPLACE INTO kg_relations (id, source_id, target_id, relation_type) VALUES (?, ?, ?, ?)"
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(relId, to: stmt, index: 1)
            bindText(sourceId, to: stmt, index: 2)
            bindText(targetId, to: stmt, index: 3)
            bindText(relationType, to: stmt, index: 4)
        }
    }

    func lookupEntity(query: String, entityType: String? = nil) throws -> [String: Any]? {
        guard let db else { throw DBError.notOpen }

        // First try exact name match
        let exactSQL = """
            SELECT id, entity_type, name, metadata, description, importance
            FROM kg_entities
            WHERE name = ?
              AND (? IS NULL OR entity_type = ?)
            LIMIT 1
        """
        var exactStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, exactSQL, -1, &exactStmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        bindText(query, to: exactStmt, index: 1)
        if let entityType {
            bindText(entityType, to: exactStmt, index: 2)
            bindText(entityType, to: exactStmt, index: 3)
        } else {
            sqlite3_bind_null(exactStmt, 2)
            sqlite3_bind_null(exactStmt, 3)
        }

        var entityId: String?
        var result: [String: Any]?

        if sqlite3_step(exactStmt) == SQLITE_ROW {
            entityId = columnText(exactStmt, 0)
            result = [
                "entity_id": entityId as Any,
                "entity_type": columnText(exactStmt, 1) as Any,
                "name": columnText(exactStmt, 2) as Any,
                "metadata": columnText(exactStmt, 3) as Any,
                "description": columnText(exactStmt, 4) as Any,
                "importance": sqlite3_column_double(exactStmt, 5)
            ]
        }
        sqlite3_finalize(exactStmt)

        // If no exact match, try LIKE
        if result == nil {
            let likeSQL = """
                SELECT id, entity_type, name, metadata, description, importance
                FROM kg_entities
                WHERE name LIKE ?
                  AND (? IS NULL OR entity_type = ?)
                LIMIT 1
            """
            var likeStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, likeSQL, -1, &likeStmt, nil) == SQLITE_OK else {
                throw DBError.prepare(sqlite3_errcode(db))
            }
            bindText("%\(query)%", to: likeStmt, index: 1)
            if let entityType {
                bindText(entityType, to: likeStmt, index: 2)
                bindText(entityType, to: likeStmt, index: 3)
            } else {
                sqlite3_bind_null(likeStmt, 2)
                sqlite3_bind_null(likeStmt, 3)
            }

            if sqlite3_step(likeStmt) == SQLITE_ROW {
                entityId = columnText(likeStmt, 0)
                result = [
                    "entity_id": entityId as Any,
                    "entity_type": columnText(likeStmt, 1) as Any,
                    "name": columnText(likeStmt, 2) as Any,
                    "metadata": columnText(likeStmt, 3) as Any,
                    "description": columnText(likeStmt, 4) as Any,
                    "importance": sqlite3_column_double(likeStmt, 5)
                ]
            }
            sqlite3_finalize(likeStmt)
        }

        // Get typed relations for found entity (excludes co_occurs_with noise)
        if let entityId, result != nil {
            let expirationSelects = try kgRelationExpirationSelects(alias: "r")
            let relSQL = """
                SELECT r.relation_type, e.name, r.target_id AS related_entity_id, 'outgoing' AS direction,
                       \(expirationSelects.validUntil), \(expirationSelects.expiredAt)
                FROM kg_relations r
                LEFT JOIN kg_entities e ON e.id = r.target_id
                WHERE r.source_id = ?
                  AND r.relation_type != 'co_occurs_with'
                UNION ALL
                SELECT r.relation_type, e.name, r.source_id AS related_entity_id, 'incoming' AS direction,
                       \(expirationSelects.validUntil), \(expirationSelects.expiredAt)
                FROM kg_relations r
                LEFT JOIN kg_entities e ON e.id = r.source_id
                WHERE r.target_id = ?
                  AND r.relation_type != 'co_occurs_with'
                ORDER BY 1
                LIMIT 20
            """
            var relStmt: OpaquePointer?
            if sqlite3_prepare_v2(db, relSQL, -1, &relStmt, nil) == SQLITE_OK {
                defer { sqlite3_finalize(relStmt) }
                bindText(entityId, to: relStmt, index: 1)
                bindText(entityId, to: relStmt, index: 2)
                var relations: [[String: Any]] = []
                while sqlite3_step(relStmt) == SQLITE_ROW {
                    let targetName = columnText(relStmt, 1) ?? ""
                    let direction = columnText(relStmt, 3) ?? "outgoing"
                    relations.append([
                        "relation_type": columnText(relStmt, 0) as Any,
                        "target_name": targetName as Any,
                        "target_entity_id": columnText(relStmt, 2) as Any,
                        "direction": direction as Any,
                        "valid_until": columnText(relStmt, 4) as Any,
                        "expired_at": columnText(relStmt, 5) as Any
                    ])
                }
                result?["relations"] = relations
            }
        }

        return result
    }

    private static func decodedJSONObject(_ raw: String?) -> [String: Any] {
        guard
            let raw,
            let data = raw.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return [:]
        }
        return object
    }

    // MARK: - KG fact lookup for search augmentation

    struct KGFact {
        let relatedEntity: String
        let relationType: String
        let entityType: String
        let direction: String  // "outgoing" or "incoming"
    }

    /// Pure SQL KG fact lookup — no embeddings needed.
    /// Returns typed relations for an entity, excluding co_occurs_with.
    func lookupEntityFacts(entityName: String, limit: Int = 20) throws -> [KGFact] {
        guard let db else { throw DBError.notOpen }

        let sql = """
            SELECT e2.name, r.relation_type, e2.entity_type, 'outgoing' AS direction
            FROM kg_relations_typed r
            JOIN kg_entities e1 ON r.source_id = e1.id
            JOIN kg_entities e2 ON r.target_id = e2.id
            WHERE LOWER(e1.name) = LOWER(?1)
            UNION ALL
            SELECT e1.name, r.relation_type, e1.entity_type, 'incoming'
            FROM kg_relations_typed r
            JOIN kg_entities e1 ON r.source_id = e1.id
            JOIN kg_entities e2 ON r.target_id = e2.id
            WHERE LOWER(e2.name) = LOWER(?1)
            ORDER BY 2
            LIMIT ?2
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(entityName, to: stmt, index: 1)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var facts: [KGFact] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            facts.append(KGFact(
                relatedEntity: columnText(stmt, 0) ?? "",
                relationType: columnText(stmt, 1) ?? "",
                entityType: columnText(stmt, 2) ?? "",
                direction: columnText(stmt, 3) ?? "outgoing"
            ))
        }
        return facts
    }

    /// Expose the raw db pointer for EntityCache loading.
    var dbHandle: OpaquePointer? { db }

    // MARK: - Knowledge Graph bulk queries

    struct KGEntityRow: Equatable, Sendable {
        let id: String
        let name: String
        let entityType: String
        let description: String?
        let importance: Double
        let linkedChunkCount: Int

        init(
            id: String,
            name: String,
            entityType: String,
            description: String?,
            importance: Double,
            linkedChunkCount: Int = 0
        ) {
            self.id = id
            self.name = name
            self.entityType = entityType
            self.description = description
            self.importance = importance
            self.linkedChunkCount = linkedChunkCount
        }
    }

    private struct KGEntityAliasSeed {
        let id: String
        let name: String
        let entityType: String
    }

    struct KGRelationRow: Equatable, Sendable {
        let id: String
        let sourceId: String
        let targetId: String
        let relationType: String
        let validUntil: Date?
        let expiredAt: Date?

        init(
            id: String,
            sourceId: String,
            targetId: String,
            relationType: String,
            validUntil: Date? = nil,
            expiredAt: Date? = nil
        ) {
            self.id = id
            self.sourceId = sourceId
            self.targetId = targetId
            self.relationType = relationType
            self.validUntil = validUntil
            self.expiredAt = expiredAt
        }
    }

    struct KGChunkRow: Equatable, Sendable {
        let chunkID: String
        let snippet: String
        let importance: Int
        let relevance: Double
    }

    struct ChunkCursor: Equatable, Sendable {
        let relevance: Double
        let createdAt: String
        let chunkID: String
    }

    struct ChunkPage: Equatable, Sendable {
        let rows: [KGChunkRow]
        let nextCursor: ChunkCursor?
    }

    struct SourceFileCursor: Equatable, Sendable {
        let topRelevance: Double
        let chunkCount: Int
        let sourceFile: String
    }

    struct SourceFileRow: Equatable, Sendable, Identifiable {
        let sourceFile: String
        let chunkCount: Int
        let topRelevance: Double

        var id: String { sourceFile }
    }

    struct SourceFilePage: Equatable, Sendable {
        let rows: [SourceFileRow]
        let nextCursor: SourceFileCursor?
    }

    struct EntitySidebarSnapshot: Equatable, Sendable {
        let chunkTotal: Int
        let fileTotal: Int
        let chunkPage: ChunkPage
        let filePage: SourceFilePage
    }

    struct EnrichmentStatsSummary: Equatable, Sendable {
        let totalChunks: Int
        let enriched: Int
        let unenrichedEligible: Int
        let skippedTooShort: Int
        let enrichedLast24Hours: Int

        var enrichedPercentText: String {
            guard totalChunks > 0 else { return "0.0%" }
            return String(format: "%.1f%%", (Double(enriched) / Double(totalChunks)) * 100.0)
        }
    }

    func fetchKGEntities(limit: Int = 500) throws -> [KGEntityRow] {
        guard let db else { throw DBError.notOpen }
        // Only return entities that participate in at least one semantic relation.
        // Without this, the graph is mostly disconnected noise.
        let sql = """
            SELECT e.id, e.name, e.entity_type, e.description,
                   COALESCE(e.importance, CAST(json_extract(e.metadata, '$.importance') AS REAL), 5.0) AS importance
            FROM kg_entities e
            WHERE EXISTS (
                SELECT 1 FROM kg_relations r
                WHERE r.relation_type != 'co_occurs_with'
                AND (r.source_id = e.id OR r.target_id = e.id)
            )
            ORDER BY importance DESC
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(limit))

        var seeds: [KGEntityAliasSeed] = []
        var descriptions: [String: String?] = [:]
        var importances: [String: Double] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = columnText(stmt, 0) ?? ""
            let name = columnText(stmt, 1) ?? ""
            let entityType = columnText(stmt, 2) ?? ""
            seeds.append(KGEntityAliasSeed(id: id, name: name, entityType: entityType))
            descriptions[id] = columnText(stmt, 3)
            importances[id] = sqlite3_column_double(stmt, 4)
        }

        let linkedChunkCounts = try fetchLinkedChunkCounts(for: seeds)
        let rows = seeds.map { seed in
            KGEntityRow(
                id: seed.id,
                name: seed.name,
                entityType: seed.entityType,
                description: descriptions[seed.id] ?? nil,
                importance: importances[seed.id] ?? 5.0,
                linkedChunkCount: linkedChunkCounts[seed.id, default: 0]
            )
        }
        return rows
    }

    func fetchKGRelations(limit: Int = 5_000) throws -> [KGRelationRow] {
        guard let db else { throw DBError.notOpen }
        let expirationSelects = try kgRelationExpirationSelects(alias: "kg_relations")
        // Exclude co_occurs_with — these are auto-generated from text proximity
        // and produce noisy, often incorrect edges in the graph view.
        let sql = """
            SELECT id, source_id, target_id, relation_type,
                   \(expirationSelects.validUntil), \(expirationSelects.expiredAt)
            FROM kg_relations
            WHERE relation_type != 'co_occurs_with'
            ORDER BY id
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, Int64(max(0, limit)))

        var rows: [KGRelationRow] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(KGRelationRow(
                id: columnText(stmt, 0) ?? "",
                sourceId: columnText(stmt, 1) ?? "",
                targetId: columnText(stmt, 2) ?? "",
                relationType: columnText(stmt, 3) ?? "",
                validUntil: KGTemporalDate.parse(columnText(stmt, 4)),
                expiredAt: KGTemporalDate.parse(columnText(stmt, 5))
            ))
        }
        return rows
    }

    private func kgRelationExpirationSelects(alias: String) throws -> (validUntil: String, expiredAt: String) {
        guard let db else { throw DBError.notOpen }
        let columns = try tableColumns(name: "kg_relations", on: db)
        let validUntil = columns.contains("valid_until")
            ? "\(alias).valid_until AS valid_until"
            : "NULL AS valid_until"
        let expiredAt = columns.contains("expired_at")
            ? "\(alias).expired_at AS expired_at"
            : "NULL AS expired_at"
        return (validUntil, expiredAt)
    }

    func linkEntityChunk(entityId: String, chunkId: String, relevance: Double = 1.0) throws {
        guard let db else { throw DBError.notOpen }
        let sql = "INSERT OR REPLACE INTO kg_entity_chunks (entity_id, chunk_id, relevance) VALUES (?, ?, ?)"
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(entityId, to: stmt, index: 1)
            bindText(chunkId, to: stmt, index: 2)
            sqlite3_bind_double(stmt, 3, relevance)
        }
    }

    func fetchEntityChunks(entityId: String, limit: Int = 20) throws -> [KGChunkRow] {
        try fetchEntityChunksPage(entityId: entityId, after: nil, limit: limit).rows
    }

    func fetchEntityChunkCount(entityId: String) throws -> Int {
        let entityIds = try entityAliasGroupIDs(entityId: entityId)
        let sql = """
            SELECT COUNT(DISTINCT ec.chunk_id)
            FROM kg_entity_chunks ec
            JOIN chunks c ON c.id = ec.chunk_id
            WHERE ec.entity_id IN (\(placeholders(count: entityIds.count)))
        """
        return try fetchCount(sql: sql, entityIds: entityIds)
    }

    func fetchEntitySourceFileCount(entityId: String) throws -> Int {
        guard let db else { throw DBError.notOpen }
        let entityIds = try entityAliasGroupIDs(entityId: entityId)
        let sql = """
            SELECT COUNT(DISTINCT c.source_file)
            FROM kg_entity_chunks ec
            JOIN chunks c ON c.id = ec.chunk_id
            WHERE ec.entity_id IN (\(placeholders(count: entityIds.count)))
              AND NULLIF(c.source_file, '') IS NOT NULL
        """
        return try fetchCount(sql: sql, entityIds: entityIds, db: db)
    }

    func fetchEntityChunksPage(entityId: String, after: ChunkCursor?, limit: Int) throws -> ChunkPage {
        guard let db else { throw DBError.notOpen }
        let pageLimit = max(0, limit)
        guard pageLimit > 0 else { return ChunkPage(rows: [], nextCursor: nil) }
        let entityIds = try entityAliasGroupIDs(entityId: entityId)

        let cursorPredicate: String
        if after == nil {
            cursorPredicate = ""
        } else {
            cursorPredicate = """
                WHERE (
                    relevance < ?
                    OR (relevance = ? AND created_at < ?)
                    OR (relevance = ? AND created_at = ? AND chunk_id < ?)
                )
            """
        }
        let sql = """
            SELECT chunk_id, snippet, importance, relevance, created_at
            FROM (
                SELECT c.id AS chunk_id,
                       COALESCE(NULLIF(c.summary, ''), substr(c.content, 1, 200)) AS snippet,
                       c.importance AS importance,
                       MAX(ec.relevance) AS relevance,
                       COALESCE(c.created_at, '') AS created_at
                FROM kg_entity_chunks ec
                JOIN chunks c ON c.id = ec.chunk_id
                WHERE ec.entity_id IN (\(placeholders(count: entityIds.count)))
                GROUP BY c.id
            )
            \(cursorPredicate)
            ORDER BY relevance DESC, created_at DESC, chunk_id DESC
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        var bindIndex = bindEntityIDs(entityIds, to: stmt, startingAt: 1)
        if let after {
            sqlite3_bind_double(stmt, bindIndex, after.relevance)
            bindIndex += 1
            sqlite3_bind_double(stmt, bindIndex, after.relevance)
            bindIndex += 1
            bindText(after.createdAt, to: stmt, index: bindIndex)
            bindIndex += 1
            sqlite3_bind_double(stmt, bindIndex, after.relevance)
            bindIndex += 1
            bindText(after.createdAt, to: stmt, index: bindIndex)
            bindIndex += 1
            bindText(after.chunkID, to: stmt, index: bindIndex)
            bindIndex += 1
        }
        sqlite3_bind_int(stmt, bindIndex, Int32(pageLimit + 1))

        var rows: [KGChunkRow] = []
        var cursors: [ChunkCursor] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(KGChunkRow(
                chunkID: columnText(stmt, 0) ?? "",
                snippet: columnText(stmt, 1) ?? "",
                importance: Int(sqlite3_column_int(stmt, 2)),
                relevance: sqlite3_column_double(stmt, 3)
            ))
            cursors.append(ChunkCursor(
                relevance: sqlite3_column_double(stmt, 3),
                createdAt: columnText(stmt, 4) ?? "",
                chunkID: columnText(stmt, 0) ?? ""
            ))
        }

        let hasMore = rows.count > pageLimit
        if hasMore {
            rows.removeLast()
            cursors.removeLast()
        }
        return ChunkPage(rows: rows, nextCursor: hasMore ? cursors.last : nil)
    }

    func fetchEntitySourceFiles(entityId: String, limit: Int, after: SourceFileCursor?) throws -> SourceFilePage {
        guard let db else { throw DBError.notOpen }
        let pageLimit = max(0, limit)
        guard pageLimit > 0 else { return SourceFilePage(rows: [], nextCursor: nil) }
        let entityIds = try entityAliasGroupIDs(entityId: entityId)

        let cursorPredicate: String
        if after == nil {
            cursorPredicate = ""
        } else {
            cursorPredicate = """
                WHERE (
                    top_relevance < ?
                    OR (top_relevance = ? AND chunk_count < ?)
                    OR (top_relevance = ? AND chunk_count = ? AND source_file > ?)
                )
            """
        }
        let sql = """
            SELECT source_file, chunk_count, top_relevance
            FROM (
                SELECT c.source_file AS source_file,
                       COUNT(DISTINCT c.id) AS chunk_count,
                       MAX(ec.relevance) AS top_relevance
                FROM kg_entity_chunks ec
                JOIN chunks c ON c.id = ec.chunk_id
                WHERE ec.entity_id IN (\(placeholders(count: entityIds.count)))
                  AND NULLIF(c.source_file, '') IS NOT NULL
                GROUP BY c.source_file
            )
            \(cursorPredicate)
            ORDER BY top_relevance DESC, chunk_count DESC, source_file ASC
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        var bindIndex = bindEntityIDs(entityIds, to: stmt, startingAt: 1)
        if let after {
            sqlite3_bind_double(stmt, bindIndex, after.topRelevance)
            bindIndex += 1
            sqlite3_bind_double(stmt, bindIndex, after.topRelevance)
            bindIndex += 1
            sqlite3_bind_int(stmt, bindIndex, Int32(after.chunkCount))
            bindIndex += 1
            sqlite3_bind_double(stmt, bindIndex, after.topRelevance)
            bindIndex += 1
            sqlite3_bind_int(stmt, bindIndex, Int32(after.chunkCount))
            bindIndex += 1
            bindText(after.sourceFile, to: stmt, index: bindIndex)
            bindIndex += 1
        }
        sqlite3_bind_int(stmt, bindIndex, Int32(pageLimit + 1))

        var rows: [SourceFileRow] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(SourceFileRow(
                sourceFile: columnText(stmt, 0) ?? "",
                chunkCount: Int(sqlite3_column_int(stmt, 1)),
                topRelevance: sqlite3_column_double(stmt, 2)
            ))
        }

        let hasMore = rows.count > pageLimit
        if hasMore {
            rows.removeLast()
        }
        let cursor = rows.last.map {
            SourceFileCursor(topRelevance: $0.topRelevance, chunkCount: $0.chunkCount, sourceFile: $0.sourceFile)
        }
        return SourceFilePage(rows: rows, nextCursor: hasMore ? cursor : nil)
    }

    func fetchEntitySidebarSnapshot(
        entityId: String,
        chunkLimit: Int,
        fileLimit: Int,
        shouldCancel: () -> Bool = { false }
    ) throws -> EntitySidebarSnapshot {
        try withReadTransaction(shouldCancel: shouldCancel) {
            try EntitySidebarSnapshot(
                chunkTotal: fetchEntityChunkCount(entityId: entityId),
                fileTotal: fetchEntitySourceFileCount(entityId: entityId),
                chunkPage: fetchEntityChunksPage(entityId: entityId, after: nil, limit: chunkLimit),
                filePage: fetchEntitySourceFiles(entityId: entityId, limit: fileLimit, after: nil)
            )
        }
    }

    private func fetchCount(sql: String, entityId: String, db providedDB: OpaquePointer? = nil) throws -> Int {
        try fetchCount(sql: sql, entityIds: [entityId], db: providedDB)
    }

    private func fetchCount(sql: String, entityIds: [String], db providedDB: OpaquePointer? = nil) throws -> Int {
        guard let db = providedDB ?? self.db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        _ = bindEntityIDs(entityIds, to: stmt, startingAt: 1)
        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_ROW else {
            throw DBError.step(stepRC)
        }
        return Int(sqlite3_column_int(stmt, 0))
    }

    private func entityAliasGroupIDs(entityId: String) throws -> [String] {
        guard let db else { throw DBError.notOpen }
        let entitySQL = "SELECT name, entity_type FROM kg_entities WHERE id = ?"
        var entityStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, entitySQL, -1, &entityStmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(entityStmt) }
        bindText(entityId, to: entityStmt, index: 1)
        guard sqlite3_step(entityStmt) == SQLITE_ROW else {
            return [entityId]
        }
        var orderedIds = [entityId]
        var seen = Set(orderedIds)
        func append(_ id: String) {
            guard !id.isEmpty, !seen.contains(id) else { return }
            seen.insert(id)
            orderedIds.append(id)
        }

        let aliasEntitySQL = """
            SELECT DISTINCT aliasEntity.id
            FROM kg_entity_aliases a
            JOIN kg_entities canonical ON canonical.id = a.entity_id
            JOIN kg_entities aliasEntity ON lower(aliasEntity.name) = lower(a.alias)
            WHERE a.entity_id IN (\(placeholders(count: orderedIds.count)))
              AND aliasEntity.entity_type = canonical.entity_type
            ORDER BY aliasEntity.id
        """
        var aliasStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, aliasEntitySQL, -1, &aliasStmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(aliasStmt) }
        _ = bindEntityIDs(orderedIds, to: aliasStmt, startingAt: 1)
        while sqlite3_step(aliasStmt) == SQLITE_ROW {
            append(columnText(aliasStmt, 0) ?? "")
        }

        return orderedIds
    }

    private func placeholders(count: Int) -> String {
        Array(repeating: "?", count: max(1, count)).joined(separator: ", ")
    }

    private func fetchLinkedChunkCounts(for entities: [KGEntityAliasSeed]) throws -> [String: Int] {
        guard let db else { throw DBError.notOpen }
        guard !entities.isEmpty else { return [:] }

        let visibleIds = entities.map(\.id)
        var groupMembers = Dictionary(uniqueKeysWithValues: entities.map { ($0.id, [$0.id]) })
        let canonicalIds = visibleIds
        let aliasMembers = try aliasEntityIDsByCanonicalID(canonicalIds: canonicalIds, db: db)
        for entity in entities {
            if let aliases = aliasMembers[entity.id] {
                groupMembers[entity.id, default: [entity.id]].append(contentsOf: aliases)
            }
        }

        for (groupId, members) in groupMembers {
            var seen: Set<String> = []
            groupMembers[groupId] = members.filter { seen.insert($0).inserted }
        }
        return try fetchDistinctChunkCounts(groupMembers: groupMembers, db: db)
    }

    private func aliasEntityIDsByCanonicalID(canonicalIds: [String], db: OpaquePointer) throws -> [String: [String]] {
        guard !canonicalIds.isEmpty else { return [:] }
        let sql = """
            SELECT a.entity_id, aliasEntity.id
            FROM kg_entity_aliases a
            JOIN kg_entities canonical ON canonical.id = a.entity_id
            JOIN kg_entities aliasEntity ON lower(aliasEntity.name) = lower(a.alias)
            WHERE a.entity_id IN (\(placeholders(count: canonicalIds.count)))
              AND aliasEntity.entity_type = canonical.entity_type
            ORDER BY a.entity_id, aliasEntity.id
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        _ = bindEntityIDs(canonicalIds, to: stmt, startingAt: 1)
        var aliases: [String: [String]] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let canonicalId = columnText(stmt, 0) ?? ""
            let aliasId = columnText(stmt, 1) ?? ""
            guard !canonicalId.isEmpty, !aliasId.isEmpty else { continue }
            aliases[canonicalId, default: []].append(aliasId)
        }
        return aliases
    }

    private func fetchDistinctChunkCounts(groupMembers: [String: [String]], db: OpaquePointer) throws -> [String: Int] {
        let pairs = groupMembers.flatMap { groupId, entityIds in entityIds.map { (groupId, $0) } }
        guard !pairs.isEmpty else { return [:] }
        let values = Array(repeating: "(?, ?)", count: pairs.count).joined(separator: ", ")
        let sql = """
            WITH group_members(group_id, entity_id) AS (VALUES \(values))
            SELECT gm.group_id, COUNT(DISTINCT ec.chunk_id)
            FROM group_members gm
            JOIN kg_entity_chunks ec ON ec.entity_id = gm.entity_id
            JOIN chunks c ON c.id = ec.chunk_id
            GROUP BY gm.group_id
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        var bindIndex: Int32 = 1
        for (groupId, entityId) in pairs {
            bindText(groupId, to: stmt, index: bindIndex)
            bindIndex += 1
            bindText(entityId, to: stmt, index: bindIndex)
            bindIndex += 1
        }
        var counts: [String: Int] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            counts[columnText(stmt, 0) ?? ""] = Int(sqlite3_column_int(stmt, 1))
        }
        return counts
    }

    @discardableResult
    private func bindEntityIDs(_ entityIds: [String], to stmt: OpaquePointer?, startingAt index: Int32) -> Int32 {
        var bindIndex = index
        for entityId in entityIds {
            bindText(entityId, to: stmt, index: bindIndex)
            bindIndex += 1
        }
        return bindIndex
    }

    func getChunk(id: String) throws -> [String: Any]? {
        guard let db else { throw DBError.notOpen }
        let sql = """
            SELECT id, content, content_type, source, summary, created_at, archived_at, superseded_by, tags, source_file
            FROM chunks
            WHERE id = ?
            LIMIT 1
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(id, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return [
            "id": columnText(stmt, 0) as Any,
            "content": columnText(stmt, 1) as Any,
            "content_type": columnText(stmt, 2) as Any,
            "source": columnText(stmt, 3) as Any,
            "summary": columnText(stmt, 4) as Any,
            "created_at": columnText(stmt, 5) as Any,
            "archived_at": columnText(stmt, 6) as Any,
            "superseded_by": columnText(stmt, 7) as Any,
            "tags": columnText(stmt, 8) as Any,
            "source_file": columnText(stmt, 9) as Any,
        ]
    }

    func archiveChunk(id: String, reason _: String? = nil) throws -> Bool {
        // Keep the reason parameter for Python MCP contract parity; BrainBar
        // does not persist archive reasons yet.
        guard try getChunk(id: id) != nil else { return false }
        try executeUpdate(
            """
            UPDATE chunks
            SET archived = 1,
                archived_at = ?,
                status = 'archived'
            WHERE id = ?
            """
        ) { stmt in
            bindText(Self.timestamp(), to: stmt, index: 1)
            bindText(id, to: stmt, index: 2)
        }
        refreshSearchStatisticsBestEffort()
        return true
    }

    func supersedeChunk(oldChunkID: String, newChunkID: String) throws -> Bool {
        guard try getChunk(id: oldChunkID) != nil, try getChunk(id: newChunkID) != nil else {
            return false
        }
        try executeUpdate(
            "UPDATE chunks SET superseded_by = ?, status = 'superseded' WHERE id = ?"
        ) { stmt in
            bindText(newChunkID, to: stmt, index: 1)
            bindText(oldChunkID, to: stmt, index: 2)
        }
        refreshSearchStatisticsBestEffort()
        return true
    }

    func getPersonContext(name: String, context: String?, numMemories: Int) throws -> [String: Any]? {
        guard let entity = try lookupEntity(query: name, entityType: "person") else {
            return nil
        }
        guard let entityID = entity["entity_id"] as? String else {
            return nil
        }

        let profile = Self.decodedJSONObject(entity["metadata"] as? String)
        let relations = ((entity["relations"] as? [[String: Any]]) ?? []).map { relation in
            [
                "relation_type": relation["relation_type"] as Any,
                "target": (relation["target_name"] as? String) ?? "",
                "direction": relation["direction"] as Any,
            ]
        }
        let memories = try fetchEntityMemories(entityID: entityID, context: context, limit: numMemories)

        return [
            "entity_id": entityID,
            "name": entity["name"] as Any,
            "entity_type": entity["entity_type"] as Any,
            "profile": profile,
            "hard_constraints": profile["hard_constraints"] as? [String: Any] ?? [:],
            "preferences": profile["preferences"] as? [String: Any] ?? [:],
            "contact_info": profile["contact_info"] as? [String: Any] ?? [:],
            "relations": relations,
            "memories": memories,
            "memory_count": memories.count,
        ]
    }

    private func fetchEntityMemories(entityID: String, context: String?, limit: Int) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        let hasContext = context?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        let sql: String
        if hasContext {
            sql = """
                SELECT c.content, c.content_type, c.created_at, c.summary, ec.relevance
                FROM kg_entity_chunks ec
                JOIN chunks c ON c.id = ec.chunk_id
                WHERE ec.entity_id = ?
                  AND c.superseded_by IS NULL
                  AND c.archived_at IS NULL
                  AND (c.content LIKE ? OR COALESCE(c.summary, '') LIKE ?)
                ORDER BY ec.relevance DESC, c.created_at DESC
                LIMIT ?
            """
        } else {
            sql = """
                SELECT c.content, c.content_type, c.created_at, c.summary, ec.relevance
                FROM kg_entity_chunks ec
                JOIN chunks c ON c.id = ec.chunk_id
                WHERE ec.entity_id = ?
                  AND c.superseded_by IS NULL
                  AND c.archived_at IS NULL
                ORDER BY ec.relevance DESC, c.created_at DESC
                LIMIT ?
            """
        }

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(entityID, to: stmt, index: 1)
        var bindIndex: Int32 = 2
        if let context, hasContext {
            let pattern = "%\(context)%"
            bindText(pattern, to: stmt, index: bindIndex)
            bindIndex += 1
            bindText(pattern, to: stmt, index: bindIndex)
            bindIndex += 1
        }
        sqlite3_bind_int(stmt, bindIndex, Int32(limit))

        var memories: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            memories.append([
                "content": String((columnText(stmt, 0) ?? "").prefix(500)),
                "type": columnText(stmt, 1) ?? "unknown",
                "date": String((columnText(stmt, 2) ?? "").prefix(10)),
                "summary": columnText(stmt, 3) as Any,
                "relevance": sqlite3_column_double(stmt, 4),
            ])
        }
        return memories
    }

    func enrichmentStats() throws -> EnrichmentStatsSummary {
        guard let db else { throw DBError.notOpen }
        let enrichedAtEpochSQL = Self.normalizedUnixEpochSQL(for: "enriched_at")
        let terminalCoveragePredicate = "enrich_status IS NOT NULL"

        func queryInt(_ sql: String) throws -> Int {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DBError.prepare(sqlite3_errcode(db))
            }
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(stmt, 0))
        }

        return EnrichmentStatsSummary(
            totalChunks: try queryInt("SELECT COUNT(*) FROM chunks"),
            enriched: try queryInt("SELECT COUNT(*) FROM chunks WHERE \(terminalCoveragePredicate)"),
            unenrichedEligible: try queryInt("SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL AND enrich_status IS NULL AND char_count >= 50"),
            skippedTooShort: try queryInt("SELECT COUNT(*) FROM chunks WHERE enrich_status IS NULL AND enriched_at IS NULL AND char_count < 50"),
            enrichedLast24Hours: try queryInt("""
                SELECT COUNT(*)
                FROM (
                    SELECT \(enrichedAtEpochSQL) AS enriched_epoch, enrich_status
                    FROM chunks
                )
                WHERE enrich_status = 'success'
                  AND enriched_epoch IS NOT NULL
                  AND enriched_epoch >= unixepoch('now', '-24 hours')
            """)
        )
    }

    func enrichChunks(
        mode: String,
        limit: Int,
        sinceHours: Int,
        phase _: String,
        chunkIDs: [String]?
    ) throws -> [String: Any] {
        guard let db else { throw DBError.notOpen }
        if let chunkIDs, chunkIDs.isEmpty {
            return [
                "mode": mode,
                "attempted": 0,
                "enriched": 0,
                "skipped": 0,
                "failed": 0,
            ]
        }

        var conditions = [
            "superseded_by IS NULL",
            "aggregated_into IS NULL",
            "archived_at IS NULL",
            "COALESCE(archived, 0) = 0",
            "COALESCE(status, 'active') = 'active'"
        ]
        if chunkIDs == nil {
            conditions.append("enriched_at IS NULL")
            conditions.append("enrich_status IS NULL")
            conditions.append("char_count >= 50")
        }
        if mode == "realtime", chunkIDs == nil {
            conditions.append("created_at >= datetime('now', ?)")
        }
        if let chunkIDs, !chunkIDs.isEmpty {
            let placeholders = Array(repeating: "?", count: chunkIDs.count).joined(separator: ", ")
            conditions.append("id IN (\(placeholders))")
        }

        let sql = """
            SELECT id, content, summary
            FROM chunks
            WHERE \(conditions.joined(separator: " AND "))
            ORDER BY created_at DESC
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }

        var bindIndex: Int32 = 1
        if mode == "realtime", chunkIDs == nil {
            bindText("-\(sinceHours) hours", to: stmt, index: bindIndex)
            bindIndex += 1
        }
        if let chunkIDs, !chunkIDs.isEmpty {
            for chunkID in chunkIDs {
                bindText(chunkID, to: stmt, index: bindIndex)
                bindIndex += 1
            }
        }
        sqlite3_bind_int(stmt, bindIndex, Int32(limit))

        var targets: [(id: String, content: String)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            targets.append((columnText(stmt, 0) ?? "", columnText(stmt, 1) ?? ""))
        }

        var enriched = 0
        var failed = 0
        for target in targets {
            let summary = Self.previewText(summary: nil, content: target.content)
            do {
                try executeUpdate(
                    "UPDATE chunks SET summary = ?, enriched_at = ?, enrich_status = 'success' WHERE id = ?"
                ) { stmt in
                    bindText(summary, to: stmt, index: 1)
                    bindText(Self.timestamp(), to: stmt, index: 2)
                    bindText(target.id, to: stmt, index: 3)
                }
                enriched += 1
            } catch {
                failed += 1
                NSLog(
                    "[BrainBar] Enrichment summary backfill failed for chunk %@: %@",
                    target.id,
                    String(describing: error)
                )
            }
        }
        refreshSearchStatisticsBestEffort()

        return [
            "mode": mode,
            "attempted": targets.count,
            "enriched": enriched,
            "skipped": max(0, targets.count - enriched - failed),
            "failed": failed,
        ]
    }

    // MARK: - brain_recall

    func recallSession(sessionId: String, limit: Int = 20) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        let sql = """
            SELECT id, content, project, content_type, importance, created_at, summary, tags
            FROM chunks WHERE conversation_id = ? ORDER BY rowid DESC LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(sessionId, to: stmt, index: 1)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append([
                "chunk_id": columnText(stmt, 0) as Any,
                "content": columnText(stmt, 1) as Any,
                "project": columnText(stmt, 2) as Any,
                "content_type": columnText(stmt, 3) as Any,
                "importance": sqlite3_column_double(stmt, 4),
                "created_at": columnText(stmt, 5) as Any,
                "summary": columnText(stmt, 6) as Any,
                "tags": columnText(stmt, 7) as Any,
                "score": 1.0  // session recall has no relevance scoring
            ])
        }
        return results
    }

    func recallStats() throws -> [String: Any] {
        guard let db else { throw DBError.notOpen }

        func queryInt(_ sql: String) throws -> Int {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DBError.prepare(sqlite3_errcode(db))
            }
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(stmt, 0))
        }

        let totalChunks = try queryInt("SELECT COUNT(*) FROM chunks")
        let totalEntities = try queryInt("SELECT COUNT(*) FROM kg_entities")
        let totalRelations = try queryInt("SELECT COUNT(*) FROM kg_relations")
        // Pending enrichment is represented by NULL; any persisted status is terminal coverage.
        let enrichedChunks = try queryInt("SELECT COUNT(*) FROM chunks WHERE enrich_status IS NOT NULL")
        let totalProjects = try queryInt("SELECT COUNT(DISTINCT project) FROM chunks")
        let projects = try queryStrings("SELECT DISTINCT project FROM chunks WHERE project IS NOT NULL AND project != '' ORDER BY project ASC LIMIT 12")
        let contentTypes = try queryStrings("SELECT DISTINCT content_type FROM chunks WHERE content_type IS NOT NULL AND content_type != '' ORDER BY content_type ASC")

        return [
            "total_chunks": totalChunks,
            "total_entities": totalEntities,
            "total_relations": totalRelations,
            "enriched_chunks": enrichedChunks,
            "total_projects": totalProjects,
            "enrichment_pct": totalChunks > 0 ? Double(enrichedChunks) / Double(totalChunks) * 100.0 : 0.0,
            "projects": projects,
            "content_types": contentTypes
        ]
    }

    private func queryStrings(_ sql: String) throws -> [String] {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }

        var values: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let value = columnText(stmt, 0), !value.isEmpty {
                values.append(value)
            }
        }
        return values
    }

    // MARK: - Injection events

    @discardableResult
    func recordInjectionEvent(sessionID: String, query: String, chunkIDs: [String], tokenCount: Int, timestamp: String? = nil) -> InjectionEvent {
        guard let db else {
            return InjectionEvent(id: 0, sessionID: sessionID, timestamp: "", query: query, chunkIDs: chunkIDs, tokenCount: tokenCount)
        }
        let chunkJSON = (try? JSONSerialization.data(withJSONObject: chunkIDs)).flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let ts = timestamp ?? Self.sqliteDateFormatter.string(from: Date())
        let sql = "INSERT INTO injection_events (session_id, timestamp, query, chunk_ids, token_count) VALUES (?, ?, ?, ?, ?)"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            return InjectionEvent(id: 0, sessionID: sessionID, timestamp: ts, query: query, chunkIDs: chunkIDs, tokenCount: tokenCount)
        }
        defer { sqlite3_finalize(stmt) }
        bindText(sessionID, to: stmt, index: 1)
        bindText(ts, to: stmt, index: 2)
        bindText(query, to: stmt, index: 3)
        bindText(chunkJSON, to: stmt, index: 4)
        sqlite3_bind_int(stmt, 5, Int32(tokenCount))
        let stepRC = sqlite3_step(stmt)
        let rowID = sqlite3_last_insert_rowid(db)
        if stepRC == SQLITE_DONE {
            Self.postDashboardChangeNotification()
        }
        return InjectionEvent(id: rowID, sessionID: sessionID, timestamp: ts, query: query, chunkIDs: chunkIDs, tokenCount: tokenCount)
    }

    // MARK: - Injection event listing

    func listInjectionEvents(sessionID: String? = nil, limit: Int = 20) throws -> [InjectionEvent] {
        guard let db else { throw DBError.notOpen }
        let hasModeColumn = try tableColumns(name: "injection_events", on: db).contains("mode")
        let modeSelect = hasModeColumn ? "mode" : "'normal'"
        var sql = "SELECT id, session_id, timestamp, query, chunk_ids, token_count, \(modeSelect) AS mode FROM injection_events"
        var conditions: [String] = []
        if sessionID != nil { conditions.append("session_id = ?") }
        conditions.append(Self.liveInjectionEventSQLPredicate(eventTable: "injection_events"))
        sql += " WHERE \(conditions.joined(separator: " AND "))"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        var idx: Int32 = 1
        if let sessionID {
            bindText(sessionID, to: stmt, index: idx)
            idx += 1
        }
        sqlite3_bind_int(stmt, idx, Int32(limit))

        var events: [InjectionEvent] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let row: [String: Any] = [
                "id": Int64(sqlite3_column_int64(stmt, 0)),
                "session_id": columnText(stmt, 1) as Any,
                "timestamp": columnText(stmt, 2) as Any,
                "query": columnText(stmt, 3) as Any,
                "chunk_ids": columnText(stmt, 4) as Any,
                "token_count": Int(sqlite3_column_int(stmt, 5)),
                "mode": columnText(stmt, 6) as Any
            ]
            if let event = try? InjectionEvent(row: row) {
                let details = (try? injectionChunkDetails(ids: event.chunkIDs)) ?? []
                let resumableID = Self.resumableClaudeConversationID(event: event, chunks: details)
                guard let scopedEvent = injectionFeedScopedEvent(event, chunks: details) else {
                    continue
                }
                events.append(
                    InjectionEvent(
                        id: scopedEvent.id,
                        sessionID: scopedEvent.sessionID,
                        timestamp: scopedEvent.timestamp,
                        query: scopedEvent.query,
                        chunkIDs: scopedEvent.chunkIDs,
                        tokenCount: scopedEvent.tokenCount,
                        mode: scopedEvent.mode,
                        chunks: scopedEvent.chunks,
                        claudeConversationID: resumableID
                    )
                )
            }
        }
        return events
    }

    private static func resumableClaudeConversationID(event: InjectionEvent, chunks: [InjectionChunk]) -> String {
        if isUUID(event.claudeConversationID) {
            return event.claudeConversationID
        }
        for chunk in chunks where isUUID(chunk.claudeConversationID) {
            return chunk.claudeConversationID
        }
        for chunk in chunks {
            if let id = extractClaudeConversationID(fromSourceFile: chunk.sourceFile) {
                return id
            }
        }
        return ""
    }

    private static func extractClaudeConversationID(fromSourceFile sourceFile: String) -> String? {
        let parts = sourceFile.split(separator: "/").map(String.init)
        if let last = parts.last {
            let stem = URL(fileURLWithPath: last).deletingPathExtension().lastPathComponent
            if isUUID(stem) {
                return stem
            }
        }
        return parts.reversed().first(where: isUUID)
    }

    private static func isUUID(_ value: String) -> Bool {
        UUID(uuidString: value) != nil
    }

    private static func liveInjectionEventSQLPredicate(eventTable: String) -> String {
        let chunkIDRows = "json_each(\(eventTable).chunk_ids)"
        return """
            (
                NOT EXISTS (
                    SELECT 1
                    FROM \(chunkIDRows) AS chunk_id
                    WHERE TRIM(CAST(chunk_id.value AS TEXT)) != ''
                )
                OR EXISTS (
                    SELECT 1
                    FROM \(chunkIDRows) AS chunk_id
                    JOIN chunks AS feed_chunk
                        ON feed_chunk.id = CAST(chunk_id.value AS TEXT)
                    WHERE TRIM(CAST(chunk_id.value AS TEXT)) != ''
                      AND (
                        COALESCE(LOWER(TRIM(feed_chunk.source)), '') != 'youtube'
                        AND COALESCE(LOWER(TRIM(feed_chunk.source_file)), '') NOT LIKE 'youtube:%'
                        AND COALESCE(LOWER(TRIM(feed_chunk.content_type)), '') NOT IN ('transcript', 'podcast', 'seed')
                        AND NOT EXISTS (
                            SELECT 1
                            FROM json_each(
                                CASE
                                    WHEN json_valid(feed_chunk.tags) THEN feed_chunk.tags
                                    ELSE '[]'
                                END
                            ) AS feed_tag
                            WHERE LOWER(TRIM(CAST(feed_tag.value AS TEXT))) IN (
                                'manual_seed',
                                'seed',
                                'imported',
                                'podcast',
                                'youtube'
                            )
                        )
                      )
                )
                OR NOT EXISTS (
                    SELECT 1
                    FROM \(chunkIDRows) AS chunk_id
                    JOIN chunks AS feed_chunk
                        ON feed_chunk.id = CAST(chunk_id.value AS TEXT)
                    WHERE TRIM(CAST(chunk_id.value AS TEXT)) != ''
                )
            )
        """
    }

    private func injectionFeedScopedEvent(_ event: InjectionEvent, chunks: [InjectionChunk]) -> InjectionEvent? {
        guard !chunks.isEmpty else {
            return event
        }

        // Mixed events are kept, but only the live hook chunks remain visible
        // in the feed; imported seed chunks stay out of counts, previews, and
        // expansion targets.
        let liveChunks = chunks.filter(Self.isLiveInjectionFeedChunk)
        guard !liveChunks.isEmpty else {
            return nil
        }

        let knownChunkIDs = Set(chunks.map(\.id))
        let liveChunkIDs = event.chunkIDs.filter { chunkID in
            liveChunks.contains { $0.id == chunkID } || !knownChunkIDs.contains(chunkID)
        }
        let scopedChunkIDs = liveChunkIDs.isEmpty ? liveChunks.map(\.id) : liveChunkIDs
        return InjectionEvent(
            id: event.id,
            sessionID: event.sessionID,
            timestamp: event.timestamp,
            query: event.query,
            chunkIDs: scopedChunkIDs,
            tokenCount: event.tokenCount,
            mode: event.mode,
            chunks: liveChunks,
            claudeConversationID: event.claudeConversationID
        )
    }

    private static func isLiveInjectionFeedChunk(_ chunk: InjectionChunk) -> Bool {
        let source = chunk.source.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let sourceFile = chunk.sourceFile.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let contentType = chunk.contentType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let tags = chunk.tags.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }

        if source == "youtube" || sourceFile.hasPrefix("youtube:") {
            return false
        }
        if contentType == "transcript" || contentType == "podcast" || contentType == "seed" {
            return false
        }
        if tags.contains(where: { tag in
            tag == "manual_seed" ||
                tag == "seed" ||
                tag == "imported" ||
                tag == "podcast" ||
                tag == "youtube"
        }) {
            return false
        }
        return true
    }

    private func injectionChunkDetails(ids: [String]) throws -> [InjectionChunk] {
        guard let db else { throw DBError.notOpen }
        let orderedIDs = ids.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !orderedIDs.isEmpty else { return [] }

        let placeholders = Array(repeating: "?", count: orderedIDs.count).joined(separator: ",")
        let sql = """
            SELECT id, content, summary, source, source_file, tags, content_type,
                   CASE
                       WHEN json_valid(metadata) THEN COALESCE(json_extract(metadata, '$.claude_conversation_id'), '')
                       ELSE ''
                   END AS claude_conversation_id
            FROM chunks
            WHERE id IN (\(placeholders))
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }

        for (offset, id) in orderedIDs.enumerated() {
            bindText(id, to: stmt, index: Int32(offset + 1))
        }

        var detailsByID: [String: InjectionChunk] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let row: [String: Any] = [
                "id": columnText(stmt, 0) as Any,
                "content": columnText(stmt, 1) as Any,
                "summary": columnText(stmt, 2) as Any,
                "source": columnText(stmt, 3) as Any,
                "source_file": columnText(stmt, 4) as Any,
                "tags": columnText(stmt, 5) as Any,
                "content_type": columnText(stmt, 6) as Any,
                "claude_conversation_id": columnText(stmt, 7) as Any,
            ]
            let detail = InjectionChunk(row: row)
            detailsByID[detail.id] = detail
        }
        return orderedIDs.compactMap { detailsByID[$0] }
    }

    // MARK: - brain_digest: rule-based entity extraction

    /// Build a deterministic, KG-resolvable entity ID from an entity name.
    /// Format mirrors the existing `<type>-<slug>` convention so repeated digests
    /// upsert the same entity (via INSERT OR REPLACE) instead of duplicating.
    static func digestEntityID(name: String) -> String {
        let slug = name
            .lowercased()
            .map { $0.isLetter || $0.isNumber ? $0 : "-" }
            .reduce(into: "") { acc, ch in
                if ch == "-" && acc.hasSuffix("-") { return }
                acc.append(ch)
            }
        let trimmed = slug.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return "digest-entity-\(trimmed.isEmpty ? "unknown" : trimmed)"
    }

    private func entityIDForDigestEntity(name: String) throws -> String? {
        guard let db else { throw DBError.notOpen }

        let existingSQL = """
            SELECT id
            FROM kg_entities
            WHERE name = ?
            ORDER BY CASE entity_type
                WHEN 'person' THEN 0
                WHEN 'project' THEN 1
                WHEN 'company' THEN 2
                WHEN 'tool' THEN 3
                WHEN 'concept' THEN 4
                ELSE 5
            END, id
            LIMIT 1
        """
        var existingStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, existingSQL, -1, &existingStmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        bindText(name, to: existingStmt, index: 1)
        if sqlite3_step(existingStmt) == SQLITE_ROW {
            let existingID = columnText(existingStmt, 0)
            sqlite3_finalize(existingStmt)
            return existingID
        }
        sqlite3_finalize(existingStmt)

        let entityID = Self.digestEntityID(name: name)
        let insertSQL = "INSERT OR IGNORE INTO kg_entities (id, entity_type, name, metadata) VALUES (?, 'concept', ?, '{}')"
        try runWriteStatement(on: db, sql: insertSQL, retries: 3) { stmt in
            bindText(entityID, to: stmt, index: 1)
            bindText(name, to: stmt, index: 2)
        }

        var insertedStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, existingSQL, -1, &insertedStmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(insertedStmt) }
        bindText(name, to: insertedStmt, index: 1)
        guard sqlite3_step(insertedStmt) == SQLITE_ROW else { return nil }
        return columnText(insertedStmt, 0)
    }

    func digest(content: String, project: String? = nil, title: String? = nil) throws -> [String: Any] {
        guard db != nil else { throw DBError.notOpen }

        // Rule-based entity extraction
        var entities: [String] = []
        var urls: [String] = []
        var codeIds: [String] = []

        // Extract capitalized multi-word names (2-3 words, each capitalized)
        let namePattern = try NSRegularExpression(pattern: "\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+){1,2})\\b")
        let nsContent = content as NSString
        let nameMatches = namePattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in nameMatches {
            let name = nsContent.substring(with: match.range)
            // Filter common non-entity phrases
            let skip = ["The", "This", "That", "These", "Those", "Here", "There", "When", "What", "Which", "Where", "How"]
            if !skip.contains(where: { name.hasPrefix($0 + " ") }) {
                entities.append(name)
            }
        }

        // Extract PascalCase identifiers (code names like BrainLayer, MCPRouter)
        let pascalPattern = try NSRegularExpression(pattern: "\\b([A-Z][a-z]+[A-Z][a-zA-Z]+)\\b")
        let pascalMatches = pascalPattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in pascalMatches {
            entities.append(nsContent.substring(with: match.range))
        }

        // Extract URLs
        let urlPattern = try NSRegularExpression(pattern: "https?://[^\\s,)]+")
        let urlMatches = urlPattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in urlMatches {
            urls.append(nsContent.substring(with: match.range))
        }

        // Extract code identifiers (snake_case, dotted paths)
        let codePattern = try NSRegularExpression(pattern: "\\b([a-z][a-z_]+\\.[a-z_]+)\\b")
        let codeMatches = codePattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in codeMatches {
            codeIds.append(nsContent.substring(with: match.range))
        }

        // Deduplicate
        entities = Array(Set(entities))
        urls = Array(Set(urls))
        codeIds = Array(Set(codeIds))

        // Store the digest as a chunk
        let digestSummary = "Digest: \(entities.count) entities, \(urls.count) URLs, \(codeIds.count) code refs"
        let titledContent: String = {
            let body = String(content.prefix(500)) + (content.count > 500 ? "..." : "")
            if let title, !title.isEmpty { return "\(title)\n\n\(body)" }
            return body
        }()

        do {
            let stored = try store(
                content: titledContent,
                tags: ["digest"] + entities.prefix(5).map { $0 },
                importance: 5,
                source: "digest",
                project: project
            )

            // Persist extracted entities to the knowledge graph so they are
            // immediately resolvable via brain_entity (lookupEntity), and link
            // each to the digested chunk so brain_entity surfaces its memories.
            var entitiesPersisted = 0
            for name in entities {
                do {
                    guard let entityID = try entityIDForDigestEntity(name: name) else { continue }
                    try linkEntityChunk(entityId: entityID, chunkId: stored.chunkID, relevance: 1.0)
                    entitiesPersisted += 1
                } catch {
                    // Best-effort: a single failed entity write must not fail the digest.
                }
            }

            return [
                "mode": "digest",
                "entities": entities,
                "entities_created": entitiesPersisted,
                "urls": urls,
                "code_identifiers": codeIds,
                "chunks_created": 1,
                "relations_created": 0,
                "chunk_id": stored.chunkID,
                "summary": digestSummary
            ]
        } catch {
            return [
                "mode": "digest",
                "entities": entities,
                "entities_created": 0,
                "urls": urls,
                "code_identifiers": codeIds,
                "chunks_created": 0,
                "relations_created": 0,
                "error": "Storage failed: \(error.localizedDescription)",
                "summary": "\(digestSummary) (storage failed)"
            ]
        }
    }

    private static let sqliteDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter
    }()

    enum DBError: LocalizedError {
        case notOpen
        case open(String, Int32)
        case prepare(Int32)
        case step(Int32)
        case exec(Int32, String)
        case noResult
        case invalidPragma(String)

        var errorDescription: String? {
            switch self {
            case .notOpen: return "Database not open"
            case .open(let path, let rc): return "SQLite open failed at \(path): \(rc)"
            case .prepare(let rc): return "SQLite prepare failed: \(rc)"
            case .step(let rc): return "SQLite step failed: \(rc)"
            case .exec(let rc, let message): return "SQLite exec failed: \(rc) (\(message))"
            case .noResult: return "No result"
            case .invalidPragma(let name): return "PRAGMA '\(name)' not in allowlist"
            }
        }
    }

    deinit {
        close()
    }
}
