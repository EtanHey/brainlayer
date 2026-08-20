// MCPRouter.swift — MCP JSON-RPC method router.
//
// Routes the 3 core MCP methods:
// - initialize: handshake, return capabilities
// - tools/list: enumerate all 8 BrainLayer tools with schemas
// - tools/call: dispatch to tool handler by name
//
// Also handles notifications (no response) and unknown methods (error).

import Foundation

final class MCPRouter: @unchecked Sendable {
    private enum ToolProfile {
        case core
        case full
    }

    private static let profileEnvironmentKey = "BRAINLAYER_MCP_PROFILE"
    private let backupWriterStartedAtUnix: TimeInterval
    private static let coreToolNames: Set<String> = [
        "brain_search",
        "brain_store",
        "brain_recall",
        "brain_expand",
    ]
    // BrainBar's Unix socket is owner-only (chmod 0600), which is the trust
    // boundary for local callers including scheduled backup. Keep backup
    // callable without connection-local palette state, but out of the
    // advertised core inventory while the stateless redesign is owned elsewhere.
    private static let profileIndependentCallToolNames: Set<String> = [
        "brain_backup_vacuum_into",
    ]
    private static let coreToolDescriptions: [String: String] = [
        "brain_search": "Search memory.",
        // The core palette is token-terse, but it is also the ONLY description a
        // default agent sees (the full text needs expand_palette), so it has to
        // carry the outcome contract Etan asked for on 2026-08-19 -- compressed to
        // the six words plus the one rule that stops the re-store loop.
        //
        // This costs ~100 bytes of the core tools/list boot payload, which sat at
        // 1499/1500 with no headroom. Etan RATIFIED the guard move to 1600 on
        // 2026-08-19: the bytes buy outcome disambiguation, because agents were
        // re-storing on ambiguous responses. The guard is now 1600 EXACTLY and is
        // not headroom to spend -- see testCoreToolsListStaysWithinBootBudget.
        "brain_store": "Store memory. status: STORED|DUPLICATE|MERGED|DEFERRED all = success, "
            + "do NOT re-store; REJECTED|ERROR = nothing stored, no chunk_id.",
        "brain_recall": "Recall context.",
        "brain_expand": "Expand chunk.",
    ]
    nonisolated(unsafe) private static let expandPaletteToolDefinition: [String: Any] = [
        "name": "expand_palette",
        "description": "Expose all tools.",
        "inputSchema": [
            "type": "object",
        ] as [String: Any],
    ]

    // Keep contested MCP writes interactive: make one short attempt, then queue
    // for replay so brain_store returns well under the 1s prompt-queue budget.
    // Longer contention handling belongs in the deferred single-writer/backpressure
    // path, not in the synchronous MCP response.
    //
    // Budget math: a contended BEGIN IMMEDIATE blocks up to the busy timeout, and
    // each extra retry adds another busy-timeout block plus a retry sleep. To stay
    // under the 200ms queue budget the synchronous path makes exactly one short
    // attempt (retries == 0) with a sub-budget busy timeout, then queues on busy.
    private static let mcpStoreBusyTimeoutMillis: Int32 = 25
    private static let mcpStoreRetries = 0
    private static let pendingStoreDrainInitialDelay: TimeInterval = 0.25
    private static let pendingStoreDrainMaxDelay: TimeInterval = 30.0

    final class PaletteSession: @unchecked Sendable {
        private let lock = NSLock()
        private var expanded = false
        let conversationID: String

        init(conversationID: String = "mcp-\(UUID().uuidString.lowercased())") {
            self.conversationID = conversationID
        }

        func isExpanded() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return expanded
        }

        func expand() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !expanded else { return false }
            expanded = true
            return true
        }
    }

    private struct ToolOutput {
        let text: String
        let metadata: [String: Any]

        init(text: String, metadata: [String: Any] = [:]) {
            self.text = text
            self.metadata = metadata
        }
    }

    private struct BackupVacuumResult: Encodable {
        let status: String
        let targetPath: String
        let bytes: Int64

        enum CodingKeys: String, CodingKey {
            case status
            case targetPath = "target_path"
            case bytes
        }
    }

    private struct HybridSearchArgumentBox: @unchecked Sendable {
        let arguments: [String: Any]
    }

    private final class HybridSearchResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<HybridSearchResponse, Error>?

        func set(_ result: Result<HybridSearchResponse, Error>) {
            lock.lock()
            self.result = result
            lock.unlock()
        }

        func get() -> Result<HybridSearchResponse, Error>? {
            lock.lock()
            defer { lock.unlock() }
            return result
        }
    }

    private final class PendingStoreDrainRegistry: @unchecked Sendable {
        private let lock = NSLock()
        private var activeKeys = Set<String>()

        func insert(_ key: String) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return activeKeys.insert(key).inserted
        }

        func remove(_ key: String) {
            lock.lock()
            activeKeys.remove(key)
            lock.unlock()
        }
    }

    private final class PendingStoreDrainScheduler: @unchecked Sendable {
        let queue: DispatchQueue
        let registry = PendingStoreDrainRegistry()

        init(queue: DispatchQueue) {
            self.queue = queue
        }
    }

    private var database: BrainDatabase?
    private var readDatabase: BrainDatabase?
    private let hybridSearchClient: HybridSearchClientProtocol?
    private let dbPath: String?
    private let hybridSearchBudget: TimeInterval
    private let toolProfile: ToolProfile
    private let pendingStoreDrainScheduler: PendingStoreDrainScheduler
    private let defaultPaletteSession = PaletteSession()
    let entityCache = EntityCache()
    private static let defaultStringMaxLength = 256
    private static let defaultStringArrayMaxItems = 100
    private static let stringMaxLengths: [String: Int] = [
        "action": 64,
        "agent_id": 128,
        "chunk_id": 128,
        "content": 200_000,
        "detail": 16,
        "mode": 32,
        "name": 256,
        "new_chunk_id": 128,
        "old_chunk_id": 128,
        "project": 256,
        "query": 4_096,
        "reason": 1_024,
        "session_id": 128,
        "source": 32,
        "tag": 128,
        "target_path": 4_096
    ]
    private static let stringArrayLimits: [String: (maxItems: Int, itemMaxLength: Int)] = [
        "chunk_ids": (maxItems: 500, itemMaxLength: 128),
        "tags": (maxItems: 100, itemMaxLength: 128)
    ]

    init(
        profile: String? = nil,
        hybridSearchClient: HybridSearchClientProtocol? = nil,
        hybridSearchBudget: TimeInterval = 0.8,
        dbPath: String? = nil,
        pendingStoreDrainQueue: DispatchQueue? = nil,
        backupWriterStartedAtUnix: TimeInterval = Date().timeIntervalSince1970
    ) {
        self.toolProfile = Self.resolveToolProfile(profile)
        self.hybridSearchClient = hybridSearchClient
        self.hybridSearchBudget = max(0.001, hybridSearchBudget)
        self.dbPath = dbPath
        self.pendingStoreDrainScheduler = PendingStoreDrainScheduler(
            queue: pendingStoreDrainQueue ?? DispatchQueue(
                label: "com.brainlayer.brainbar.pending-store-drain",
                qos: .utility
            )
        )
        self.backupWriterStartedAtUnix = backupWriterStartedAtUnix
    }

    private static func resolveToolProfile(_ explicitProfile: String?) -> ToolProfile {
        let environmentProfile = ProcessInfo.processInfo.environment[profileEnvironmentKey]
        let rawProfile = explicitProfile ?? environmentProfile
        let normalized = rawProfile?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""

        switch normalized {
        case "full", "operator":
            return .full
        case "", "core":
            return .core
        default:
            if explicitProfile == nil, let rawProfile = environmentProfile {
                let message = "BrainBar: unknown \(profileEnvironmentKey)=\(rawProfile); using core profile\n"
                FileHandle.standardError.write(Data(message.utf8))
            }
            return .core
        }
    }

    func makePaletteSession() -> PaletteSession {
        PaletteSession()
    }

    private func exposedToolDefinitions(for session: PaletteSession) -> [[String: Any]] {
        if toolProfile == .full || session.isExpanded() {
            return Self.toolDefinitions
        }

        let coreDefinitions = Self.toolDefinitions.filter { definition in
            guard let name = definition["name"] as? String else { return false }
            return Self.coreToolNames.contains(name)
        }.map(Self.compactCoreToolDefinition)
        return coreDefinitions + [Self.expandPaletteToolDefinition]
    }

    func isToolExposed(_ name: String, session: PaletteSession) -> Bool {
        exposedToolDefinitions(for: session).contains { ($0["name"] as? String) == name }
    }

    static func compactCoreToolDefinition(_ definition: [String: Any]) -> [String: Any] {
        guard let name = definition["name"] as? String else { return definition }
        guard let description = coreToolDescriptions[name]
            ?? (definition["description"] as? String),
              !description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            preconditionFailure("Core tool '\(name)' has no usable description")
        }

        var compact: [String: Any] = [
            "name": name,
            "description": description,
        ]
        if let inputSchema = definition["inputSchema"] as? [String: Any] {
            compact["inputSchema"] = removingDescriptions(from: inputSchema)
        }
        return compact
    }

    private static func removingDescriptions(from value: Any) -> Any {
        if let dictionary = value as? [String: Any] {
            return dictionary.reduce(into: [String: Any]()) { result, entry in
                guard entry.key != "description" else { return }
                result[entry.key] = removingDescriptions(from: entry.value)
            }
        }
        if let array = value as? [Any] {
            return array.map(removingDescriptions(from:))
        }
        return value
    }

    private func expandPalette(session: PaletteSession) -> ToolOutput {
        guard session.expand() else {
            return ToolOutput(
                text: "BrainLayer tool palette is already expanded.",
                metadata: [
                    "expanded": false,
                    "already_expanded": true,
                    "registered_tools": [String](),
                ]
            )
        }

        let deferredNames = Self.toolDefinitions.compactMap { definition -> String? in
            guard let name = definition["name"] as? String else { return nil }
            return Self.coreToolNames.contains(name) ? nil : name
        }
        return ToolOutput(
            text: "Expanded BrainLayer tool palette.",
            metadata: [
                "expanded": true,
                "already_expanded": false,
                "registered_tools": deferredNames,
            ]
        )
    }

    /// Inject database for tool handlers + load entity cache.
    func setDatabase(_ db: BrainDatabase) {
        setDatabases(write: db, read: db)
    }

    /// Inject separate write and read handles. Read tools use the read handle so
    /// WAL readers do not inherit write-handle busy_timeout or transaction state.
    func setDatabases(write writeDB: BrainDatabase, read readDB: BrainDatabase) {
        database = writeDB
        readDatabase = readDB
        entityCache.load(from: readDB.dbHandle)
        entityCache.startRefreshTimer(db: readDB.dbHandle)
        scheduleDrainForExistingPendingStores(db: writeDB)
    }

    /// Stores acknowledged before the database was injected (DB_NOT_OPEN queue path)
    /// have no drain scheduled; without this, they persist only if a later store
    /// happens to run. Schedule their drains as soon as a write handle exists.
    private func scheduleDrainForExistingPendingStores(db: BrainDatabase) {
        // The snapshot takes the cross-process queue LOCK_EX; a concurrent MCP
        // replay can hold it across DB writes, so never block the caller
        // (setDatabases runs on BrainBarServer.queue during initialization).
        let scheduler = pendingStoreDrainScheduler
        Self.scheduleExistingPendingStoreScan(scheduler: scheduler, db: db, delay: 0)
    }

    private static func scheduleExistingPendingStoreScan(
        scheduler: PendingStoreDrainScheduler,
        db: BrainDatabase,
        delay: TimeInterval
    ) {
        scheduler.queue.asyncAfter(deadline: .now() + delay) { [weak scheduler, weak db] in
            guard let scheduler, let db, db.isOpen else { return }
            guard let snapshot = db.pendingStoreQueueSnapshotIfReadable() else {
                scheduleExistingPendingStoreScan(
                    scheduler: scheduler,
                    db: db,
                    delay: min(
                        pendingStoreDrainMaxDelay,
                        max(pendingStoreDrainInitialDelay, delay * 2)
                    )
                )
                return
            }
            var scheduledAny = false
            for identity in snapshot.identityKeys where identity.hasPrefix("chunk:") {
                scheduledAny = true
                Self.schedulePendingStoreDrain(
                    scheduler: scheduler,
                    db: db,
                    chunkID: String(identity.dropFirst("chunk:".count)),
                    delay: Self.pendingStoreDrainInitialDelay
                )
            }
            if !scheduledAny && snapshot.depth > 0 {
                Self.scheduleIdentitylessLegacyFlush(
                    scheduler: scheduler,
                    db: db,
                    delay: Self.pendingStoreDrainInitialDelay
                )
            }
        }
    }

    /// Legacy entries written by older builds may lack chunk_id and thus have no
    /// identity for the normal per-chunk drain; flush with capped-backoff retries
    /// until the queue empties so a busy DB at startup cannot strand them.
    private static func scheduleIdentitylessLegacyFlush(
        scheduler: PendingStoreDrainScheduler,
        db: BrainDatabase,
        delay: TimeInterval
    ) {
        scheduler.queue.asyncAfter(deadline: .now() + delay) { [weak scheduler, weak db] in
            guard let scheduler, let db, db.isOpen else { return }
            _ = db.flushPendingStores(
                busyTimeoutMillis: mcpStoreBusyTimeoutMillis,
                retries: mcpStoreRetries
            )
            guard let after = db.pendingStoreQueueSnapshotIfReadable() else {
                scheduleIdentitylessLegacyFlush(
                    scheduler: scheduler,
                    db: db,
                    delay: min(delay * 2, 60)
                )
                return
            }
            if after.depth > 0 {
                scheduleIdentitylessLegacyFlush(
                    scheduler: scheduler,
                    db: db,
                    delay: min(delay * 2, 60)
                )
            }
        }
    }

    private func readDB() throws -> BrainDatabase {
        guard let db = readDatabase ?? database else {
            throw ToolError.noDatabase
        }
        return db
    }

    private func writeDB() throws -> BrainDatabase {
        guard let db = database else {
            throw ToolError.noDatabase
        }
        return db
    }

    /// Handle a parsed JSON-RPC request and return a response.
    /// Returns empty dict for notifications (no id).
    func handle(_ request: [String: Any], session: PaletteSession? = nil) -> [String: Any] {
        let paletteSession = session ?? defaultPaletteSession
        guard let method = request["method"] as? String else {
            return jsonRPCError(id: request["id"], code: -32600, message: "Invalid request: missing method")
        }

        // Notifications have no id — don't respond.
        // Check both missing key AND explicit JSON null (NSNull from JSONSerialization).
        let rawID = request["id"]
        let isNotification = (rawID == nil || rawID is NSNull)
        if isNotification {
            return [:]
        }
        let id = rawID!

        switch method {
        case "initialize":
            return handleInitialize(id: id, params: request["params"] as? [String: Any] ?? [:])
        case "notifications/initialized":
            // If a client sends this with an id, ack it so it doesn't hang.
            return jsonRPCResult(id: id, result: [:] as [String: Any])
        case "tools/list":
            return handleToolsList(id: id, session: paletteSession)
        case "tools/call":
            return handleToolsCall(
                id: id,
                params: request["params"] as? [String: Any] ?? [:],
                session: paletteSession
            )
        case "resources/list":
            return handleResourcesList(id: id)
        case "prompts/list":
            return jsonRPCResult(id: id, result: ["prompts": [Any]()])
        case "ping":
            return jsonRPCResult(id: id, result: [:] as [String: Any])
        default:
            return jsonRPCError(id: id, code: -32601, message: "Method not found: \(method)")
        }
    }

    // MARK: - initialize

    private func handleInitialize(id: Any, params: [String: Any]) -> [String: Any] {
        return [
            "jsonrpc": "2.0",
            "id": id,
            "result": [
                "protocolVersion": "2024-11-05",
                "capabilities": [
                    "tools": ["listChanged": true],
                    "experimental": [
                        "claude/channel": [:] as [String: Any]
                    ]
                ],
                "serverInfo": [
                    "name": "brainbar",
                    "version": "1.0.0",
                    "backupWriterStartedAtUnix": backupWriterStartedAtUnix,
                ]
            ] as [String: Any]
        ]
    }

    // MARK: - tools/list

    private func handleToolsList(id: Any, session: PaletteSession) -> [String: Any] {
        return [
            "jsonrpc": "2.0",
            "id": id,
            "result": [
                "tools": exposedToolDefinitions(for: session)
            ]
        ]
    }

    private func handleResourcesList(id: Any) -> [String: Any] {
        // Tags are available on-demand via brain_tags; do not preload them into session context.
        return jsonRPCResult(id: id, result: ["resources": [Any]()])
    }

    // MARK: - tools/call

    private func handleToolsCall(id: Any, params: [String: Any], session: PaletteSession) -> [String: Any] {
        guard let toolName = params["name"] as? String else {
            return jsonRPCError(id: id, code: -32602, message: "Missing tool name")
        }

        let arguments = params["arguments"] as? [String: Any] ?? [:]

        if toolName == "expand_palette" {
            guard toolProfile == .core else {
                return jsonRPCError(id: id, code: -32601, message: "Unknown tool: \(toolName)")
            }
            return toolCallResult(id: id, output: expandPalette(session: session))
        }

        guard Self.toolDefinitions.contains(where: { ($0["name"] as? String) == toolName }) else {
            return jsonRPCError(id: id, code: -32601, message: "Unknown tool: \(toolName)")
        }

        guard isToolExposed(toolName, session: session)
                || Self.profileIndependentCallToolNames.contains(toolName) else {
            let message = "Tool '\(toolName)' is gated by the core MCP profile. "
                + "Call expand_palette, or set BRAINLAYER_MCP_PROFILE=full on the MCP server."
            return jsonRPCError(id: id, code: -32601, message: message)
        }

        // Dispatch to handler
        do {
            try Self.validate(arguments: arguments, for: toolName)
            let output = try dispatchTool(name: toolName, arguments: arguments, session: session)
            return toolCallResult(id: id, output: output)
        } catch {
            return [
                "jsonrpc": "2.0",
                "id": id,
                "result": [
                    "content": [
                        ["type": "text", "text": Self.failureText(for: toolName, error: error)]
                    ],
                    "isError": true
                ] as [String: Any]
            ]
        }
    }

    /// brain_store failures speak the store outcome vocabulary, so an agent can
    /// tell "nothing was stored, and nothing is queued either" from a DEFERRED
    /// receipt. A bad request is REJECTED (resending it cannot succeed); anything
    /// else is ERROR (worth one retry). Other tools keep the generic wording.
    private static func failureText(for toolName: String, error: Error) -> String {
        guard toolName == "brain_store" else {
            return "Error: \(error.localizedDescription)"
        }
        let outcome: StoreOutcome
        switch error as? ToolError {
        case .missingParameter, .schemaValidation, .unknownTool:
            outcome = .rejected
        default:
            outcome = .error
        }
        return Formatters.formatStoreFailure(
            outcome: outcome,
            reason: error.localizedDescription,
            useColor: false
        )
    }

    private func toolCallResult(id: Any, output: ToolOutput) -> [String: Any] {
        var result: [String: Any] = [
            "content": [
                ["type": "text", "text": output.text]
            ]
        ]
        for (key, value) in output.metadata {
            result[key] = value
        }
        return [
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        ]
    }

    private func dispatchTool(
        name: String,
        arguments: [String: Any],
        session: PaletteSession
    ) throws -> ToolOutput {
        switch name {
        case "brain_search":
            return try handleBrainSearch(arguments)
        case "brain_store":
            return try handleBrainStore(arguments, session: session)
        case "brain_get_person":
            return try handleBrainGetPerson(arguments)
        case "brain_recall":
            return try handleBrainRecall(arguments)
        case "brain_entity":
            return try handleBrainEntity(arguments)
        case "brain_digest":
            return try handleBrainDigest(arguments)
        case "brain_update":
            return try handleBrainUpdate(arguments)
        case "brain_expand":
            return try handleBrainExpand(arguments)
        case "brain_tags":
            return try handleBrainTags(arguments)
        case "brain_supersede":
            return try handleBrainSupersede(arguments)
        case "brain_archive":
            return try handleBrainArchive(arguments)
        case "brain_enrich":
            return try handleBrainEnrich(arguments)
        case "brain_subscribe":
            return try handleBrainSubscribe(arguments)
        case "brain_unsubscribe":
            return try handleBrainUnsubscribe(arguments)
        case "brain_ack":
            return try handleBrainAck(arguments)
        case "brain_maintenance_rebuild_trigram":
            return try handleBrainMaintenanceRebuildTrigram(arguments)
        case "brain_backup_vacuum_into":
            return try handleBrainBackupVacuumInto(arguments)
        default:
            throw ToolError.unknownTool(name)
        }
    }

    // MARK: - Tool Handlers

    private func handleBrainSearch(_ args: [String: Any]) throws -> ToolOutput {
        let profileStartedAt = SearchProfileLogger.now()
        let profileQueryID = (args["_profile_query_id"] as? String)
            ?? (SearchProfileLogger.isEnabled ? SearchProfileLogger.newQueryID() : nil)
        guard let query = args["query"] as? String else {
            throw ToolError.missingParameter("query")
        }
        let limit = min(args["num_results"] as? Int ?? 5, 100)
        let project = args["project"] as? String
        let source = args["source"] as? String
        let tag = args["tag"] as? String
        let subscriberID = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String)
        let unreadOnly = args["unread_only"] as? Bool ?? false
        let sourceCountsAsFilter: Bool
        if let source {
            let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
            sourceCountsAsFilter = !trimmed.isEmpty && trimmed != "all"
        } else {
            sourceCountsAsFilter = false
        }
        // importance_min may arrive as Int or Double from JSON
        let importanceMin: Double? = if let d = args["importance_min"] as? Double { d }
            else if let i = args["importance_min"] as? Int { Double(i) }
            else { nil }
        if unreadOnly && subscriberID == nil {
            throw ToolError.missingParameter("agent_id")
        }
        // Unread search marks messages delivered (a write), so it must run on the
        // writable connection. Plain reads stay on the read-only connection to
        // avoid contending for the single-writer lock.
        let db = unreadOnly ? try writeDB() : try readDB()
        SearchProfileLogger.log(
            scope: "search.brainbar",
            step: "router_dispatch",
            queryID: profileQueryID,
            durMS: SearchProfileLogger.durationMS(since: profileStartedAt)
        )

        func localKGSection() -> String {
            let hasActiveFilters = project != nil || sourceCountsAsFilter || tag != nil || subscriberID != nil || importanceMin != nil
            if hasActiveFilters {
                return ""
            }
            let detected = entityCache.detectEntities(in: query)
            guard let first = detected.first else {
                return ""
            }
            let facts = (try? db.lookupEntityFacts(entityName: first.name)) ?? []
            if facts.isEmpty {
                return ""
            }
            return TextFormatter.formatKGFacts(entity: first.name, facts: facts)
        }

        func searchViaBrainBarDatabase() throws -> (text: String, metadata: [String: Any]) {
            let results = try db.search(
                query: query,
                limit: limit,
                project: project,
                source: source,
                tag: tag,
                importanceMin: importanceMin,
                subscriberID: subscriberID,
                unreadOnly: unreadOnly
            )
            let typedResults = results.map(SearchResult.init(payload:))
            return (
                TextFormatter.formatSearchResults(
                    query: query,
                    results: typedResults,
                    total: typedResults.count,
                    detail: args["detail"] as? String ?? "compact"
                ),
                [:]
            )
        }

        let textSection: String
        let metadata: [String: Any]
        let kgSection: String
        if hybridSearchClient != nil, subscriberID == nil, !unreadOnly {
            do {
                let response = try hybridSearchWithinBudget(arguments: hybridSearchArguments(
                    query: query,
                    limit: limit,
                    project: project,
                    source: source,
                    tag: tag,
                    importanceMin: importanceMin,
                    detail: args["detail"] as? String,
                    profileQueryID: profileQueryID
                ))
                if let response {
                    if hybridSearchResponseIsEmpty(response) {
                        NSLog("[BrainBar] Hybrid search helper returned empty results, falling back to BrainBar database search")
                        let fallback = try searchViaBrainBarDatabase()
                        textSection = fallback.text
                        metadata = fallback.metadata
                        kgSection = localKGSection()
                    } else {
                        textSection = response.text
                        metadata = sanitizedHybridMetadata(response.metadata)
                        kgSection = ""
                    }
                } else {
                    NSLog("[BrainBar] Hybrid search helper exceeded %.3fs budget, falling back to BrainBar database search", hybridSearchBudget)
                    let fallback = try searchViaBrainBarDatabase()
                    textSection = fallback.text
                    metadata = fallback.metadata
                    kgSection = localKGSection()
                }
            } catch {
                NSLog("[BrainBar] Hybrid search helper failed, falling back to BrainBar database search: %@", String(describing: error))
                let fallback = try searchViaBrainBarDatabase()
                textSection = fallback.text
                metadata = fallback.metadata
                kgSection = localKGSection()
            }
        } else {
            let fallback = try searchViaBrainBarDatabase()
            textSection = fallback.text
            metadata = fallback.metadata
            kgSection = localKGSection()
        }

        // KG section goes before the <brain_search> envelope
        if kgSection.isEmpty {
            return ToolOutput(text: textSection, metadata: metadata)
        }
        return ToolOutput(text: kgSection + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + textSection, metadata: metadata)
    }

    private func hybridSearchResponseIsEmpty(_ response: HybridSearchResponse) -> Bool {
        if let structured = response.metadata["structuredContent"] as? [String: Any] {
            if let total = structured["total"] as? Int {
                return total == 0
            }
            if let total = structured["total"] as? NSNumber {
                return total.intValue == 0
            }
            if let results = structured["results"] as? [Any] {
                return results.isEmpty
            }
        }

        return response.text.contains(" - 0 of 0 shown")
    }

    private func hybridSearchWithinBudget(arguments: [String: Any]) throws -> HybridSearchResponse? {
        guard let hybridSearchClient else { return nil }
        if let readinessProvider = hybridSearchClient as? HybridSearchReadinessProviding,
           !readinessProvider.isReady {
            readinessProvider.startWarming()
            return nil
        }

        let group = DispatchGroup()
        let argumentBox = HybridSearchArgumentBox(arguments: arguments)
        let resultBox = HybridSearchResultBox()
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            let outcome = Result { try hybridSearchClient.search(arguments: argumentBox.arguments) }
            resultBox.set(outcome)
            group.leave()
        }

        let timeout = DispatchTime.now() + .nanoseconds(Int(hybridSearchBudget * 1_000_000_000))
        guard group.wait(timeout: timeout) == .success else {
            return nil
        }

        return try resultBox.get()?.get()
    }

    private func sanitizedHybridMetadata(_ metadata: [String: Any]) -> [String: Any] {
        var allowed: [String: Any] = [:]
        if let structuredContent = metadata["structuredContent"] {
            allowed["structuredContent"] = structuredContent
        }
        return allowed
    }

    private func hybridSearchArguments(
        query: String,
        limit: Int,
        project: String?,
        source: String?,
        tag: String?,
        importanceMin: Double?,
        detail: String?,
        profileQueryID: String?
    ) -> [String: Any] {
        var arguments: [String: Any] = [
            "query": query,
            "num_results": limit,
            "source": source ?? "all",
            "detail": detail ?? "compact"
        ]
        if let project {
            arguments["project"] = project
        }
        if let tag {
            arguments["tag"] = tag
        }
        if let importanceMin {
            arguments["importance_min"] = importanceMin
        }
        if let profileQueryID {
            arguments["_profile_query_id"] = profileQueryID
        }
        return arguments
    }

    private func handleBrainStore(_ args: [String: Any], session: PaletteSession) throws -> ToolOutput {
        guard let content = args["content"] as? String else {
            throw ToolError.missingParameter("content")
        }
        let tags = args["tags"] as? [String] ?? []
        let importance = args["importance"] as? Int ?? 5
        let project = args["project"] as? String
        guard let db = database else {
            return try queueBrainStore(
                content: content,
                tags: tags,
                importance: importance,
                source: "mcp",
                project: project,
                conversationID: session.conversationID,
                reason: "DB_NOT_OPEN"
            )
        }
        let hadPendingStoresBeforeAttempt = db.pendingStoreQueueSnapshot().depth > 0
        do {
            switch try db.storeOrQueueWithinBudget(
                content: content,
                tags: tags,
                importance: importance,
                source: "mcp",
                project: project,
                conversationID: session.conversationID,
                busyTimeoutMillis: Self.mcpStoreBusyTimeoutMillis,
                retries: Self.mcpStoreRetries
            ) {
            case .stored(let stored):
                let flushedStores = db.flushPendingStores()
                return ToolOutput(
                    text: Formatters.formatStoreResult(
                        chunkId: stored.chunkID,
                        tags: tags,
                        outcome: stored.outcome,
                        useColor: false
                    ),
                    metadata: [
                        "queued": false,
                        // STORED vs DUPLICATE: both resolve to chunk_id and both are
                        // success, but only one of them wrote a row. An agent that
                        // cannot tell them apart re-stores.
                        "status": stored.outcome.status,
                        "stored_new": stored.outcome.storedNew,
                        "chunk_id": stored.chunkID,
                        "flushed_count": flushedStores.count,
                        "_brainbarStoredChunk": [
                            "chunk_id": stored.chunkID,
                            "rowid": stored.rowID
                        ],
                        "_brainbarFlushedQueuedChunks": Self.flushedQueuedChunkReceipts(flushedStores)
                    ]
                )
            case .queued(let queueID, let queuedAt, let chunkID):
                // Mirror the .stored path: even when the current store queues under a
                // busy DB, drain the existing backlog so prior writes are not stranded.
                // Exclude the just-queued chunk so it is not immediately re-replayed /
                // double-stored before its own deferred drain runs.
                let flushedStores = hadPendingStoresBeforeAttempt
                    ? db.flushPendingStores(
                        excludingChunkIDs: [chunkID],
                        busyTimeoutMillis: Self.mcpStoreBusyTimeoutMillis,
                        retries: Self.mcpStoreRetries
                    )
                    : []
                schedulePendingStoreDrain(db: db, chunkID: chunkID)
                return queuedBrainStoreOutput(
                    queueID: queueID,
                    queuedAt: queuedAt,
                    chunkID: chunkID,
                    queuePath: db.pendingStoreQueuePathForReceipt(),
                    flushedStores: flushedStores
                )
            }
        } catch BrainDatabase.DBError.notOpen {
            return try queueBrainStore(
                content: content,
                tags: tags,
                importance: importance,
                source: "mcp",
                project: project,
                conversationID: session.conversationID,
                reason: "DB_NOT_OPEN"
            )
        }
    }

    private func schedulePendingStoreDrain(db: BrainDatabase, chunkID: String) {
        Self.schedulePendingStoreDrain(
            scheduler: pendingStoreDrainScheduler,
            db: db,
            chunkID: chunkID,
            delay: Self.pendingStoreDrainInitialDelay
        )
    }

    private static func schedulePendingStoreDrain(
        scheduler: PendingStoreDrainScheduler,
        db: BrainDatabase,
        chunkID: String,
        delay: TimeInterval
    ) {
        let drainKey = "\(ObjectIdentifier(db).hashValue):\(chunkID)"
        guard scheduler.registry.insert(drainKey) else { return }

        scheduler.queue.asyncAfter(deadline: .now() + delay) { [weak scheduler, weak db] in
            guard let scheduler, let db else { return }
            drainPendingStoreTarget(
                scheduler: scheduler,
                db: db,
                chunkID: chunkID,
                drainKey: drainKey,
                delay: delay
            )
        }
    }

    private static func drainPendingStoreTarget(
        scheduler: PendingStoreDrainScheduler,
        db: BrainDatabase,
        chunkID: String,
        drainKey: String,
        delay: TimeInterval
    ) {
        guard db.isOpen else {
            finishPendingStoreDrain(scheduler: scheduler, drainKey: drainKey)
            return
        }

        let targetIdentity = "chunk:\(chunkID)"
        guard let before = db.pendingStoreQueueSnapshotIfReadable() else {
            reschedulePendingStoreDrain(
                scheduler: scheduler,
                db: db,
                chunkID: chunkID,
                drainKey: drainKey,
                delay: delay,
                flushedAny: false
            )
            return
        }
        guard before.identityKeys.contains(targetIdentity) else {
            finishPendingStoreDrain(scheduler: scheduler, drainKey: drainKey)
            return
        }

        let flushedStores = db.flushPendingStores(
            busyTimeoutMillis: mcpStoreBusyTimeoutMillis,
            retries: mcpStoreRetries
        )
        guard let after = db.pendingStoreQueueSnapshotIfReadable() else {
            reschedulePendingStoreDrain(
                scheduler: scheduler,
                db: db,
                chunkID: chunkID,
                drainKey: drainKey,
                delay: delay,
                flushedAny: !flushedStores.isEmpty
            )
            return
        }
        guard after.identityKeys.contains(targetIdentity) else {
            finishPendingStoreDrain(scheduler: scheduler, drainKey: drainKey)
            return
        }

        reschedulePendingStoreDrain(
            scheduler: scheduler,
            db: db,
            chunkID: chunkID,
            drainKey: drainKey,
            delay: delay,
            flushedAny: !flushedStores.isEmpty
        )
    }

    private static func reschedulePendingStoreDrain(
        scheduler: PendingStoreDrainScheduler,
        db: BrainDatabase,
        chunkID: String,
        drainKey: String,
        delay: TimeInterval,
        flushedAny: Bool
    ) {
        let nextDelay = flushedAny
            ? pendingStoreDrainInitialDelay
            : min(pendingStoreDrainMaxDelay, max(pendingStoreDrainInitialDelay, delay * 2.0))
        scheduler.queue.asyncAfter(deadline: .now() + nextDelay) { [weak scheduler, weak db] in
            guard let scheduler, let db else { return }
            drainPendingStoreTarget(
                scheduler: scheduler,
                db: db,
                chunkID: chunkID,
                drainKey: drainKey,
                delay: nextDelay
            )
        }
    }

    private static func finishPendingStoreDrain(
        scheduler: PendingStoreDrainScheduler,
        drainKey: String
    ) {
        scheduler.registry.remove(drainKey)
    }

    private func queueBrainStore(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        project: String?,
        conversationID: String,
        reason: String
    ) throws -> ToolOutput {
        guard let dbPath else {
            throw ToolError.noDatabase
        }
        let queued = try BrainDatabase.queuePendingStore(
            dbPath: dbPath,
            content: content,
            tags: tags,
            importance: importance,
            source: source,
            project: project,
            conversationID: conversationID
        )
        return queuedBrainStoreOutput(
            queueID: queued.queueID,
            queuedAt: queued.queuedAt,
            chunkID: queued.chunkID,
            queuePath: BrainDatabase.pendingStoreQueuePathForReceipt(dbPath: dbPath),
            reason: reason
        )
    }

    private func queuedBrainStoreOutput(
        queueID: String,
        queuedAt: String,
        chunkID: String,
        queuePath: URL,
        reason: String = "DB_BUSY",
        flushedStores: [BrainDatabase.FlushedPendingStore] = []
    ) -> ToolOutput {
        let action = Self.deferredQueueAction(for: queuePath)
        return ToolOutput(
            text: Formatters.formatStoreResult(chunkId: chunkID, queued: true, queuedReason: reason, useColor: false),
            metadata: [
                "queued": true,
                "status": "DEFERRED",
                "queue_id": queueID,
                "queued_at": queuedAt,
                "chunk_id": chunkID,
                "related": [] as [String],
                "deferred": [
                    "status": "DEFERRED",
                    "reason": reason,
                    "chunk_id": chunkID,
                    "queue_id": queueID,
                    "queued_at": queuedAt,
                    "queue_path": queuePath.path,
                    "action": action
                ] as [String: Any],
                "flushed_count": flushedStores.count,
                "_brainbarFlushedQueuedChunks": Self.flushedQueuedChunkReceipts(flushedStores)
            ]
        )
    }

    private static func deferredQueueAction(for queuePath: URL) -> String {
        queuePath.lastPathComponent == "pending-stores.jsonl" ? "queued_for_replay" : "queued_for_drain"
    }

    private static func flushedQueuedChunkReceipts(_ flushedStores: [BrainDatabase.FlushedPendingStore]) -> [[String: Any]] {
        flushedStores.map { flushed in
            [
                "chunk_id": flushed.storedChunk.chunkID,
                "rowid": flushed.storedChunk.rowID
            ] as [String: Any]
        }
    }

    private func handleBrainGetPerson(_ args: [String: Any]) throws -> ToolOutput {
        guard let name = args["name"] as? String else {
            throw ToolError.missingParameter("name")
        }
        let context = args["context"] as? String
        let numMemories = min(args["num_memories"] as? Int ?? 10, 50)
        let db = try readDB()
        guard let person = try db.getPersonContext(name: name, context: context, numMemories: numMemories) else {
            return ToolOutput(text: "No person entity found matching '\(name)'.")
        }
        return ToolOutput(text: Formatters.formatEntityCard(entity: person, useColor: false))
    }

    private func handleBrainRecall(_ args: [String: Any]) throws -> ToolOutput {
        let db = try readDB()
        let mode = args["mode"] as? String ?? "stats"
        if mode == "injections" {
            let sessionId = args["session_id"] as? String
            let events = try db.listInjectionEvents(sessionID: sessionId, limit: 20)
            if events.isEmpty {
                return ToolOutput(text: "│ No injection events found")
            }
            var lines: [String] = []
            lines.append("┌─ brain_recall injections (\(events.count) events)")
            for event in events {
                lines.append("│  [\(event.timestamp.prefix(19))] \(event.sessionID)")
                lines.append("│    \(event.summaryLine)")
                if !event.chunkIDs.isEmpty {
                    lines.append("│    " + event.chunkIDs.joined(separator: ", "))
                }
            }
            lines.append("└─")
            return ToolOutput(text: lines.joined(separator: "\n"))
        }
        if mode == "context" {
            let sessionId = args["session_id"] as? String ?? ""
            if sessionId.isEmpty {
                let stats = try db.recallStats()
                return ToolOutput(text: TextFormatter.formatStats(StatsResult(payload: stats)))
            }
            let results = try db.recallSession(sessionId: sessionId, limit: 20)
            let typedResults = results.map(SearchResult.init(payload:))
            return ToolOutput(text: TextFormatter.formatRecalledContext(query: "session:\(sessionId)", results: typedResults))
        }
        let stats = try db.recallStats()
        return ToolOutput(text: TextFormatter.formatStats(StatsResult(payload: stats)))
    }

    private func handleBrainEntity(_ args: [String: Any]) throws -> ToolOutput {
        guard let query = args["query"] as? String else {
            throw ToolError.missingParameter("query")
        }
        let db = try readDB()
        guard let entity = try db.lookupEntity(query: query) else {
            return ToolOutput(text: "\u{2502} No entity found for \"\(query)\"")
        }
        return ToolOutput(text: TextFormatter.formatEntitySimple(EntityCard(lookupPayload: entity)))
    }

    private func handleBrainDigest(_ args: [String: Any]) throws -> ToolOutput {
        guard let content = args["content"] as? String else {
            throw ToolError.missingParameter("content")
        }
        let project = (args["project"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let title = (args["title"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let db = try writeDB()
        let result = try db.digest(content: content, project: project, title: title)
        if let error = result["error"] as? String {
            throw ToolError.operationFailed(error)
        }
        let integrityKeys = [
            "content_integrity",
            "expected_characters",
            "stored_characters",
            "expected_bytes",
            "stored_bytes"
        ]
        var metadata: [String: Any] = [:]
        for key in integrityKeys {
            metadata[key] = result[key]
        }
        return ToolOutput(
            text: TextFormatter.formatDigestResult(DigestResult(payload: result)),
            metadata: metadata
        )
    }

    private func handleBrainUpdate(_ args: [String: Any]) throws -> ToolOutput {
        let db = try writeDB()
        let chunkId = args["chunk_id"] as? String ?? ""
        if chunkId.isEmpty {
            throw ToolError.missingParameter("chunk_id")
        }
        let importance = args["importance"] as? Int
        let tags = args["tags"] as? [String]
        if importance == nil && tags == nil {
            throw ToolError.missingParameter("importance or tags")
        }
        try db.updateChunk(id: chunkId, importance: importance, tags: tags)
        return ToolOutput(text: "\u{2714} Updated \(chunkId)" + (importance != nil ? " imp:\(importance!)" : "") + (tags != nil ? " tags:\(tags!.joined(separator: ","))" : ""))
    }

    private func handleBrainExpand(_ args: [String: Any]) throws -> ToolOutput {
        guard let chunkId = args["chunk_id"] as? String else {
            throw ToolError.missingParameter("chunk_id")
        }
        let db = try readDB()
        let before = args["before"] as? Int ?? 3
        let after = args["after"] as? Int ?? 3
        let expanded = try db.expandChunk(
            id: chunkId,
            before: before,
            after: after,
            includeFullTargetContent: true
        )
        let target = expanded["target"] as? [String: Any] ?? [:]
        let context = expanded["context"] as? [[String: Any]] ?? []
        var lines: [String] = []
        lines.append("\u{250c}\u{2500} brain_expand: \(chunkId)")
        let targetContent = (target["content"] as? String) ?? ""
        if !targetContent.isEmpty {
            lines.append("\u{251c}\u{2500} Target")
            lines.append("\u{2502} \(targetContent)")
        }
        if !context.isEmpty {
            lines.append("\u{251c}\u{2500} Context (\(context.count) chunks)")
            for c in context {
                let cid = (c["chunk_id"] as? String ?? "").prefix(12)
                let snippet = String(((c["content"] as? String) ?? "").prefix(80))
                lines.append("\u{2502}  [\(cid)] \(snippet)")
            }
        }
        lines.append("\u{2514}\u{2500}")
        return ToolOutput(text: lines.joined(separator: "\n"))
    }

    private func handleBrainTags(_ args: [String: Any]) throws -> ToolOutput {
        let db = try readDB()
        let query = args["query"] as? String
        let limit = args["limit"] as? Int ?? 50
        let tags = try db.listTags(query: query, limit: limit)
        if tags.isEmpty {
            return ToolOutput(text: "\u{2502} No tags found" + (query != nil ? " matching \"\(query!)\"" : ""))
        }
        var lines: [String] = []
        lines.append("\u{250c}\u{2500} brain_tags (\(tags.count) tags)")
        for t in tags {
            let name = t["tag"] as? String ?? ""
            let count = t["count"] as? Int ?? 0
            lines.append("\u{2502}  \(name) (\(count))")
        }
        lines.append("\u{2514}\u{2500}")
        return ToolOutput(text: lines.joined(separator: "\n"))
    }

    private func handleBrainSupersede(_ args: [String: Any]) throws -> ToolOutput {
        guard let oldChunkID = args["old_chunk_id"] as? String else {
            throw ToolError.missingParameter("old_chunk_id")
        }
        guard let newChunkID = args["new_chunk_id"] as? String else {
            throw ToolError.missingParameter("new_chunk_id")
        }
        let safetyCheck = args["safety_check"] as? String ?? "auto"
        let confirm = args["confirm"] as? Bool ?? false
        let db = try writeDB()

        guard let oldChunk = try db.getChunk(id: oldChunkID) else {
            throw ToolError.notFound("Old chunk not found: \(oldChunkID)")
        }
        guard try db.getChunk(id: newChunkID) != nil else {
            throw ToolError.notFound("New chunk not found: \(newChunkID)")
        }

        if safetyCheck == "auto", requiresSupersedeConfirmation(chunk: oldChunk) {
            return ToolOutput(text: jsonEncode([
                "action": "confirm_required",
                "reason": "Old chunk contains personal data — requires safety_check='confirm' and confirm=true",
                "old_chunk_id": oldChunkID,
                "old_preview": String((oldChunk["content"] as? String ?? "").prefix(200)),
                "new_chunk_id": newChunkID,
            ] as [String: String]))
        }

        if safetyCheck == "confirm", !confirm {
            return ToolOutput(text: jsonEncode([
                "action": "confirm_required",
                "old_chunk_id": oldChunkID,
                "old_preview": String((oldChunk["content"] as? String ?? "").prefix(200)),
                "new_chunk_id": newChunkID,
                "instruction": "Re-call with confirm=true to proceed",
            ] as [String: String]))
        }

        guard try db.supersedeChunk(oldChunkID: oldChunkID, newChunkID: newChunkID) else {
            throw ToolError.notFound("Supersede failed for: \(oldChunkID)")
        }
        return ToolOutput(text: jsonEncode([
            "action": "superseded",
            "old_chunk_id": oldChunkID,
            "new_chunk_id": newChunkID,
        ] as [String: String]))
    }

    private func handleBrainArchive(_ args: [String: Any]) throws -> ToolOutput {
        guard let chunkID = args["chunk_id"] as? String else {
            throw ToolError.missingParameter("chunk_id")
        }
        let reason = args["reason"] as? String
        let db = try writeDB()
        guard try db.archiveChunk(id: chunkID, reason: reason) else {
            throw ToolError.notFound("Chunk not found: \(chunkID)")
        }
        var payload: [String: String] = [
            "action": "archived",
            "chunk_id": chunkID,
        ]
        if let reason {
            payload["reason"] = reason
        }
        return ToolOutput(text: jsonEncode(payload))
    }

    private func handleBrainEnrich(_ args: [String: Any]) throws -> ToolOutput {
        let mode = args["mode"] as? String ?? "realtime"
        let limit = max(1, min(args["limit"] as? Int ?? 25, 5_000))
        let sinceHours = args["since_hours"] as? Int ?? 8_760
        let phase = args["phase"] as? String ?? "run"
        let chunkIDs = args["chunk_ids"] as? [String]
        let stats = args["stats"] as? Bool ?? false
        let db = try writeDB()

        if stats {
            let summary = try db.enrichmentStats()
            let lines = [
                "\u{250c}\u{2500} Enrichment Stats",
                "\u{2502} Total: \(summary.totalChunks)  Enriched: \(summary.enriched) (\(summary.enrichedPercentText))  Remaining: \(summary.unenrichedEligible)  Skipped: \(summary.skippedTooShort)",
                "\u{2502} Last 24h: \(summary.enrichedLast24Hours) enriched",
                "\u{2514}\u{2500}",
            ]
            return ToolOutput(text: lines.joined(separator: "\n"))
        }

        let result = try db.enrichChunks(
            mode: mode,
            limit: limit,
            sinceHours: sinceHours,
            phase: phase,
            chunkIDs: chunkIDs
        )
        return ToolOutput(text: Formatters.formatDigestResult(result: result, useColor: false))
    }

    private func handleBrainSubscribe(_ args: [String: Any]) throws -> ToolOutput {
        guard let _ = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String) else {
            throw ToolError.missingParameter("agent_id")
        }
        guard let _ = args["tags"] as? [String] else {
            throw ToolError.missingParameter("tags")
        }
        throw ToolError.notImplemented("brain_subscribe")
    }

    private func handleBrainUnsubscribe(_ args: [String: Any]) throws -> ToolOutput {
        guard let _ = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String) else {
            throw ToolError.missingParameter("agent_id")
        }
        throw ToolError.notImplemented("brain_unsubscribe")
    }

    private func handleBrainAck(_ args: [String: Any]) throws -> ToolOutput {
        guard let _ = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String) else {
            throw ToolError.missingParameter("agent_id")
        }
        guard args["seq"] is Int || args["seq"] is Int64 else {
            throw ToolError.missingParameter("seq")
        }
        throw ToolError.notImplemented("brain_ack")
    }

    private func requiresSupersedeConfirmation(chunk: [String: Any]) -> Bool {
        let personalTypes = Set(["journal", "note", "bookmark"])
        let personalKeywords = ["health", "family", "relationship", "finance", "personal", "therapy", "medical"]
        let contentType = (chunk["content_type"] as? String ?? "").lowercased()
        if personalTypes.contains(contentType) {
            return true
        }
        let content = (chunk["content"] as? String ?? "").lowercased()
        return personalKeywords.contains(where: { content.contains($0) })
    }

    private func handleBrainMaintenanceRebuildTrigram(_ args: [String: Any]) throws -> ToolOutput {
        let cancelRequested = args["cancel"] as? Bool ?? false
        if cancelRequested {
            let final = BrainDatabase.TrigramMaintenanceProgress(
                state: .cancelled,
                processed: 0,
                total: 0,
                etaSeconds: nil
            )
            return ToolOutput(
                text: "brain_maintenance_rebuild_trigram: cancelled 0/0",
                metadata: [
                    "progress": final.metadata,
                    "events": [final.metadata]
                ]
            )
        }

        let db = try writeDB()
        let batchSize = args["batch_size"] as? Int ?? 1_000
        let maxReturnedEvents = 25
        var events: [BrainDatabase.TrigramMaintenanceProgress] = []
        let final = try db.triggerTrigramRebuild(
            batchSize: batchSize,
            // Preflight-only: cancellation never reaches the inner rebuild loop here.
            shouldCancel: { false },
            progress: { event in
                events.append(event)
                if events.count > maxReturnedEvents {
                    events.removeFirst(events.count - maxReturnedEvents)
                }
            }
        )
        let text = "brain_maintenance_rebuild_trigram: \(final.state.rawValue) \(final.processed)/\(final.total)"
        return ToolOutput(
            text: text,
            metadata: [
                "progress": final.metadata,
                "events": events.map(\.metadata)
            ]
        )
    }

    private func handleBrainBackupVacuumInto(_ args: [String: Any]) throws -> ToolOutput {
        guard let targetPath = args["target_path"] as? String else {
            throw ToolError.missingParameter("target_path")
        }
        let db = try writeDB()
        let bytes = try db.vacuumInto(targetPath: targetPath)
        let payload = BackupVacuumResult(status: "ok", targetPath: targetPath, bytes: bytes)
        return ToolOutput(
            text: jsonEncode(payload),
            metadata: [
                "target_path": targetPath,
                "bytes": bytes,
            ]
        )
    }

    /// Safe JSON encoding — never use string interpolation with user data.
    private func jsonEncode<T: Encodable>(_ value: T) -> String {
        guard let data = try? JSONEncoder().encode(value),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }

    // MARK: - Error helpers

    private func jsonRPCResult(id: Any, result: [String: Any]) -> [String: Any] {
        return [
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        ]
    }

    private func jsonRPCError(id: Any?, code: Int, message: String) -> [String: Any] {
        var response: [String: Any] = [
            "jsonrpc": "2.0",
            "error": [
                "code": code,
                "message": message
            ]
        ]
        if let id { response["id"] = id }
        return response
    }

    private static func limitedInputSchema(_ schema: [String: Any]) -> [String: Any] {
        return applyInputLimits(to: schema, fieldName: nil)
    }

    private static func applyInputLimits(to schema: [String: Any], fieldName: String?) -> [String: Any] {
        var schema = schema
        guard let type = schema["type"] as? String else {
            return schema
        }

        switch type {
        case "string":
            if schema["maxLength"] == nil {
                schema["maxLength"] = stringMaxLengths[fieldName ?? ""] ?? defaultStringMaxLength
            }
        case "array":
            if var items = schema["items"] as? [String: Any] {
                if (items["type"] as? String) == "string" {
                    let limits = stringArrayLimits[fieldName ?? ""] ??
                        (maxItems: defaultStringArrayMaxItems, itemMaxLength: defaultStringMaxLength)
                    if schema["maxItems"] == nil {
                        schema["maxItems"] = limits.maxItems
                    }
                    if items["maxLength"] == nil {
                        items["maxLength"] = limits.itemMaxLength
                    }
                }
                schema["items"] = applyInputLimits(to: items, fieldName: fieldName)
            }
        case "object":
            if var properties = schema["properties"] as? [String: Any] {
                for (propertyName, value) in properties {
                    guard let propertySchema = value as? [String: Any] else { continue }
                    properties[propertyName] = applyInputLimits(to: propertySchema, fieldName: propertyName)
                }
                schema["properties"] = properties
            }
        default:
            break
        }

        return schema
    }

    private static func validate(arguments: [String: Any], for toolName: String) throws {
        guard
            let tool = toolDefinitions.first(where: { ($0["name"] as? String) == toolName }),
            let schema = tool["inputSchema"] as? [String: Any]
        else {
            return
        }

        try validate(value: arguments, against: schema, fieldPath: "arguments")
    }

    private static func validate(value: Any, against schema: [String: Any], fieldPath: String) throws {
        guard let type = schema["type"] as? String else {
            return
        }

        switch type {
        case "object":
            guard let object = value as? [String: Any] else {
                throw ToolError.schemaValidation("\(fieldPath) must be an object")
            }
            let required = schema["required"] as? [String] ?? []
            for key in required where object[key] == nil || object[key] is NSNull {
                throw ToolError.schemaValidation("\(key) is required")
            }
            if let properties = schema["properties"] as? [String: Any] {
                for (propertyName, propertyValue) in object {
                    guard
                        !(propertyValue is NSNull),
                        let propertySchema = properties[propertyName] as? [String: Any]
                    else { continue }
                    try validate(value: propertyValue, against: propertySchema, fieldPath: propertyName)
                }
            }
        case "string":
            guard let stringValue = value as? String else {
                throw ToolError.schemaValidation("\(fieldPath) must be a string")
            }
            if let enumValues = schema["enum"] as? [String], !enumValues.contains(stringValue) {
                throw ToolError.schemaValidation("\(fieldPath) must be one of \(enumValues.joined(separator: ", "))")
            }
            if let maxLength = schema["maxLength"] as? Int, stringValue.count > maxLength {
                throw ToolError.schemaValidation(
                    "\(fieldPath) length \(stringValue.count) exceeds maxLength \(maxLength)"
                )
            }
        case "array":
            guard let arrayValue = value as? [Any] else {
                throw ToolError.schemaValidation("\(fieldPath) must be an array")
            }
            if let maxItems = schema["maxItems"] as? Int, arrayValue.count > maxItems {
                throw ToolError.schemaValidation("\(fieldPath) item count \(arrayValue.count) exceeds maxItems \(maxItems)")
            }
            if let itemSchema = schema["items"] as? [String: Any] {
                for (index, item) in arrayValue.enumerated() {
                    try validate(value: item, against: itemSchema, fieldPath: "\(fieldPath)[\(index)]")
                }
            }
        default:
            break
        }
    }

    enum ToolError: LocalizedError {
        case unknownTool(String)
        case missingParameter(String)
        case noDatabase
        case notFound(String)
        case notImplemented(String)
        case schemaValidation(String)
        case operationFailed(String)

        var errorDescription: String? {
            switch self {
            case .unknownTool(let name): return "Unknown tool: \(name)"
            case .missingParameter(let param): return "Missing required parameter: \(param)"
            case .noDatabase: return "Database not available"
            case .notFound(let message): return message
            case .notImplemented(let tool): return "\(tool) not yet implemented in BrainBar (use Python MCP server)"
            case .schemaValidation(let message): return "Schema validation error: \(message)"
            case .operationFailed(let message): return message
            }
        }
    }

    // MARK: - Tool Definitions

    private static func toolAnnotations(
        readOnly: Bool,
        destructive: Bool,
        idempotent: Bool,
        openWorld: Bool = false
    ) -> [String: Any] {
        [
            "readOnlyHint": readOnly,
            "destructiveHint": destructive,
            "idempotentHint": idempotent,
            "openWorldHint": openWorld,
        ]
    }

    nonisolated(unsafe) static let readOnlyAnnotations = toolAnnotations(
        readOnly: true,
        destructive: false,
        idempotent: true
    )

    nonisolated(unsafe) static let recallAnnotations: [String: Any] = {
        var annotations = MCPRouter.readOnlyAnnotations
        annotations["anthropic/maxResultSizeChars"] = 250_000
        return annotations
    }()

    nonisolated(unsafe) static let expandAnnotations: [String: Any] = {
        var annotations = MCPRouter.readOnlyAnnotations
        annotations["anthropic/maxResultSizeChars"] = 250_000
        return annotations
    }()

    nonisolated(unsafe) static let writeAnnotations = toolAnnotations(
        readOnly: false,
        destructive: false,
        idempotent: false
    )

    nonisolated(unsafe) static let writeIdempotentAnnotations = toolAnnotations(
        readOnly: false,
        destructive: false,
        idempotent: true
    )

    nonisolated(unsafe) static let toolDefinitions: [[String: Any]] = [
        [
            "name": "brain_search",
            "description": "Search memory for past decisions, bugs, notes, and project history by topic. Use when asked \"have we done this\", \"what did we decide\", \"how did I implement X\". For session context, not topics, use brain_recall.",
            "annotations": MCPRouter.readOnlyAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Natural language search query"],
                    "num_results": ["type": "integer", "description": "Number of results (default: 5, max: 100)"],
                    "project": ["type": "string", "description": "Filter by project name"],
                    "source": ["type": "string", "enum": ["claude_code", "whatsapp", "youtube", "mcp", "all"], "description": "Filter by data source. Omit or use 'all' to search everything."],
                    "tag": ["type": "string", "description": "Filter by tag"],
                    "importance_min": ["type": "number", "description": "Minimum importance score (1-10)"],
                    "agent_id": ["type": "string", "description": "Optional stable agent id for unread filtering"],
                    "unread_only": ["type": "boolean", "description": "Return only chunks not yet acknowledged by agent_id"],
                    "detail": ["type": "string", "enum": ["compact", "full"], "description": "Result detail level"],
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any])
        ],
        [
            "name": "brain_store",
            "description": "Store a decision, correction, bug cause, or learning so future sessions find it. status STORED|DUPLICATE|MERGED|DEFERRED all = success, do NOT re-store (DEFERRED is queued and will be stored); REJECTED|ERROR store nothing and return no `status` or chunk_id. For long raw text use brain_digest.",
            "annotations": MCPRouter.writeAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "content": ["type": "string", "description": "Content to store"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Tags for categorization"],
                    "importance": ["type": "integer", "description": "Importance score (1-10)"],
                    "project": ["type": "string", "description": "Project context for the stored memory"],
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any])
        ],
        [
            "name": "brain_recall",
            "description": "Get session-level context: what you are working on now, recent sessions, one session's detail, or knowledge-base stats. For topic lookup use brain_search.",
            "annotations": MCPRouter.recallAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "mode": ["type": "string", "enum": ["context", "sessions", "operations", "plan", "summary", "stats", "injections"], "description": "Recall mode"],
                    "session_id": ["type": "string", "description": "Session ID for operations/summary mode"],
                ] as [String: Any],
            ] as [String: Any])
        ],
        [
            "name": "brain_entity",
            "description": "Look up a named person, project, company, or tool in the knowledge graph and return its relations. For fuzzy topical recall use brain_search.",
            "annotations": MCPRouter.readOnlyAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Entity name to look up"],
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any])
        ],
        [
            "name": "brain_get_person",
            "description": "Get one person's profile, relations, and linked memories in a single call.",
            "annotations": MCPRouter.readOnlyAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "name": ["type": "string", "description": "Person name to look up"],
                    "context": ["type": "string", "description": "Optional context to rank memories by relevance"],
                    "num_memories": ["type": "integer", "description": "Number of memory chunks to return (default: 10, max: 50)"],
                ] as [String: Any],
                "required": ["name"]
            ] as [String: Any])
        ],
        [
            "name": "brain_digest",
            "description": "Digest a large block of raw text (transcript, doc, article) into one searchable memory chunk: it enriches the text and connects the entities it extracts into the knowledge graph. For a short note use brain_store.",
            "annotations": MCPRouter.writeAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "content": ["type": "string", "description": "Raw text to digest into memory"],
                    "project": ["type": "string", "description": "Project context for the digested chunk (enables brain_search project scoping)"],
                    "title": ["type": "string", "description": "Optional short title/label for the digest"],
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any])
        ],
        [
            "name": "brain_update",
            "description": "Change an existing chunk's importance or tags. Does not edit content: store new content with brain_store, hide a chunk with brain_archive.",
            "annotations": MCPRouter.writeIdempotentAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "chunk_id": ["type": "string", "description": "ID of the chunk to update (from a brain_search or brain_store result)"],
                    "importance": ["type": "integer", "description": "New importance score, 1-10"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "New tag list — replaces the chunk's existing tags"],
                ] as [String: Any],
                "required": ["chunk_id"]
            ] as [String: Any])
        ],
        [
            "name": "brain_expand",
            "description": "Open one search result in full, with the chunks around it.",
            "annotations": MCPRouter.expandAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "chunk_id": ["type": "string", "description": "Chunk ID to expand"],
                    "before": ["type": "integer", "description": "Context chunks before (default: 3)"],
                    "after": ["type": "integer", "description": "Context chunks after (default: 3)"],
                ] as [String: Any],
                "required": ["chunk_id"]
            ] as [String: Any])
        ],
        [
            "name": "brain_tags",
            "description": "List the tags in use with their counts; filter by substring.",
            "annotations": MCPRouter.readOnlyAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Optional search query to filter tags"],
                ] as [String: Any],
            ] as [String: Any])
        ],
        [
            "name": "brain_supersede",
            "description": "Mark an old chunk as replaced by a newer one and hide it from search. To hide with no replacement use brain_archive.",
            "annotations": MCPRouter.toolAnnotations(readOnly: false, destructive: true, idempotent: false),
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "old_chunk_id": ["type": "string", "description": "The chunk ID to mark as superseded"],
                    "new_chunk_id": ["type": "string", "description": "The chunk ID that replaces the old one"],
                    "safety_check": ["type": "string", "enum": ["auto", "confirm"], "description": "Safety mode: auto or confirm"],
                    "confirm": ["type": "boolean", "description": "Confirm superseding personal data when required"],
                ] as [String: Any],
                "required": ["old_chunk_id", "new_chunk_id"]
            ] as [String: Any])
        ],
        [
            "name": "brain_archive",
            "description": "Hide a chunk from default search, recoverably. To point at a replacement instead use brain_supersede.",
            "annotations": MCPRouter.toolAnnotations(readOnly: false, destructive: true, idempotent: false),
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "chunk_id": ["type": "string", "description": "The chunk ID to archive"],
                    "reason": ["type": "string", "description": "Optional reason for archiving"],
                ] as [String: Any],
                "required": ["chunk_id"]
            ] as [String: Any])
        ],
        [
            "name": "brain_enrich",
            "description": "Backfill summaries and enrichment metadata on existing chunks.",
            "annotations": MCPRouter.writeAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "mode": ["type": "string", "enum": ["realtime", "batch"], "description": "Enrichment mode"],
                    "limit": ["type": "integer", "description": "Maximum number of chunks to process"],
                    "since_hours": ["type": "integer", "description": "Only enrich chunks from the last N hours in realtime mode"],
                    "phase": ["type": "string", "enum": ["submit", "poll", "import", "run"], "description": "Batch phase"],
                    "chunk_ids": ["type": "array", "items": ["type": "string"], "description": "Optional explicit chunk IDs to enrich"],
                    "stats": ["type": "boolean", "description": "Return progress statistics only"],
                ] as [String: Any],
            ] as [String: Any])
        ],
        [
            "name": "brain_subscribe",
            "description": "Subscribe an agent to live notifications for the given tags.",
            "annotations": MCPRouter.writeAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "agent_id": ["type": "string", "description": "Stable agent identifier"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Tags to receive live notifications for"],
                ] as [String: Any],
                "required": ["agent_id", "tags"]
            ] as [String: Any])
        ],
        [
            "name": "brain_unsubscribe",
            "description": "Remove some or all of an agent's tag subscriptions.",
            "annotations": MCPRouter.writeIdempotentAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "agent_id": ["type": "string", "description": "Stable agent identifier"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Optional subset of tags to remove"],
                ] as [String: Any],
                "required": ["agent_id"]
            ] as [String: Any])
        ],
        [
            "name": "brain_ack",
            "description": "Acknowledge that an agent processed messages up to a chunk rowid.",
            "annotations": MCPRouter.writeIdempotentAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "agent_id": ["type": "string", "description": "Stable agent identifier"],
                    "seq": ["type": "integer", "description": "Highest chunk rowid acknowledged by the agent"],
                ] as [String: Any],
                "required": ["agent_id", "seq"]
            ] as [String: Any])
        ],
        [
            "name": "brain_backup_vacuum_into",
            "description": "Write a SQLite backup snapshot (VACUUM INTO) to a target path.",
            "annotations": MCPRouter.writeIdempotentAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "target_path": ["type": "string", "description": "Absolute path for the new SQLite backup file"],
                ] as [String: Any],
                "required": ["target_path"]
            ] as [String: Any])
        ],
        [
            "name": "brain_maintenance_rebuild_trigram",
            "description": "Operator-triggered maintenance command to rebuild the trigram FTS table in lock-aware batches.",
            "annotations": MCPRouter.writeIdempotentAnnotations,
            "inputSchema": [
                "type": "object",
                "properties": [
                    "batch_size": ["type": "integer", "description": "Rows to backfill per write transaction (default: 1000)"],
                    "cancel": ["type": "boolean", "description": "Return a cancelled state before starting work"],
                ] as [String: Any]
            ] as [String: Any]
        ],
    ]
}
