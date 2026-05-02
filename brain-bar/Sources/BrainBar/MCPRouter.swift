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
    private struct ToolOutput {
        let text: String
        let metadata: [String: Any]

        init(text: String, metadata: [String: Any] = [:]) {
            self.text = text
            self.metadata = metadata
        }
    }

    private var database: BrainDatabase?
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
        "tag": 128
    ]
    private static let stringArrayLimits: [String: (maxItems: Int, itemMaxLength: Int)] = [
        "chunk_ids": (maxItems: 500, itemMaxLength: 128),
        "tags": (maxItems: 100, itemMaxLength: 128)
    ]

    /// Inject database for tool handlers + load entity cache.
    func setDatabase(_ db: BrainDatabase) {
        self.database = db
        entityCache.load(from: db.dbHandle)
        entityCache.startRefreshTimer(db: db.dbHandle)
    }

    /// Handle a parsed JSON-RPC request and return a response.
    /// Returns empty dict for notifications (no id).
    func handle(_ request: [String: Any]) -> [String: Any] {
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
            return handleToolsList(id: id)
        case "tools/call":
            return handleToolsCall(id: id, params: request["params"] as? [String: Any] ?? [:])
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
                    "tools": ["listChanged": false],
                    "experimental": [
                        "claude/channel": [:] as [String: Any]
                    ]
                ],
                "serverInfo": [
                    "name": "brainbar",
                    "version": "1.0.0"
                ]
            ] as [String: Any]
        ]
    }

    // MARK: - tools/list

    private func handleToolsList(id: Any) -> [String: Any] {
        return [
            "jsonrpc": "2.0",
            "id": id,
            "result": [
                "tools": Self.toolDefinitions
            ]
        ]
    }

    private func handleResourcesList(id: Any) -> [String: Any] {
        // Tags are available on-demand via brain_tags; do not preload them into session context.
        return jsonRPCResult(id: id, result: ["resources": [Any]()])
    }

    // MARK: - tools/call

    private func handleToolsCall(id: Any, params: [String: Any]) -> [String: Any] {
        guard let toolName = params["name"] as? String else {
            return jsonRPCError(id: id, code: -32602, message: "Missing tool name")
        }

        let arguments = params["arguments"] as? [String: Any] ?? [:]

        // Check tool exists
        guard Self.toolDefinitions.contains(where: { ($0["name"] as? String) == toolName }) else {
            return jsonRPCError(id: id, code: -32601, message: "Unknown tool: \(toolName)")
        }

        // Dispatch to handler
        do {
            try Self.validate(arguments: arguments, for: toolName)
            let output = try dispatchTool(name: toolName, arguments: arguments)
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
                "result": result
            ]
        } catch {
            return [
                "jsonrpc": "2.0",
                "id": id,
                "result": [
                    "content": [
                        ["type": "text", "text": "Error: \(error.localizedDescription)"]
                    ],
                    "isError": true
                ] as [String: Any]
            ]
        }
    }

    private func dispatchTool(name: String, arguments: [String: Any]) throws -> ToolOutput {
        switch name {
        case "brain_search":
            return try handleBrainSearch(arguments)
        case "brain_store":
            return try handleBrainStore(arguments)
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
        default:
            throw ToolError.unknownTool(name)
        }
    }

    // MARK: - Tool Handlers

    private func handleBrainSearch(_ args: [String: Any]) throws -> ToolOutput {
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
        guard let db = database else {
            throw ToolError.noDatabase
        }

        // Entity detection → KG fact lookup
        var kgSection = ""
        let hasActiveFilters = project != nil || sourceCountsAsFilter || tag != nil || subscriberID != nil || importanceMin != nil
        if !hasActiveFilters {
            let detected = entityCache.detectEntities(in: query)
            if let first = detected.first {
                let facts = (try? db.lookupEntityFacts(entityName: first.name)) ?? []
                if !facts.isEmpty {
                    kgSection = TextFormatter.formatKGFacts(entity: first.name, facts: facts)
                }
            }
        }

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
        let textSection = TextFormatter.formatSearchResults(query: query, results: typedResults, total: typedResults.count)

        // KG section goes before the <brain_search> envelope
        if kgSection.isEmpty {
            return ToolOutput(text: textSection)
        }
        return ToolOutput(text: kgSection + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + textSection)
    }

    private func handleBrainStore(_ args: [String: Any]) throws -> ToolOutput {
        guard let content = args["content"] as? String else {
            throw ToolError.missingParameter("content")
        }
        let tags = args["tags"] as? [String] ?? []
        let importance = args["importance"] as? Int ?? 5
        guard let db = database else {
            throw ToolError.noDatabase
        }
        do {
            let stored = try db.store(content: content, tags: tags, importance: importance, source: "mcp")
            let flushedStores = db.flushPendingStores()
            return ToolOutput(
                text: Formatters.formatStoreResult(chunkId: stored.chunkID),
                metadata: [
                    "queued": false,
                    "flushed_count": flushedStores.count,
                    "_brainbarStoredChunk": [
                        "chunk_id": stored.chunkID,
                        "rowid": stored.rowID
                    ],
                    "_brainbarFlushedQueuedChunks": flushedStores.map { flushed in
                        [
                            "chunk_id": flushed.storedChunk.chunkID,
                            "rowid": flushed.storedChunk.rowID,
                            "content": flushed.content,
                            "tags": flushed.tags,
                            "importance": flushed.importance
                        ] as [String: Any]
                    }
                ]
            )
        } catch {
            guard db.shouldQueueStoreError(error) else {
                throw error
            }
            try db.queuePendingStore(content: content, tags: tags, importance: importance, source: "mcp")
            return ToolOutput(
                text: Formatters.formatStoreResult(chunkId: "", queued: true),
                metadata: ["queued": true]
            )
        }
    }

    private func handleBrainGetPerson(_ args: [String: Any]) throws -> ToolOutput {
        guard let name = args["name"] as? String else {
            throw ToolError.missingParameter("name")
        }
        let context = args["context"] as? String
        let numMemories = min(args["num_memories"] as? Int ?? 10, 50)
        guard let db = database else { throw ToolError.noDatabase }
        guard let person = try db.getPersonContext(name: name, context: context, numMemories: numMemories) else {
            return ToolOutput(text: "No person entity found matching '\(name)'.")
        }
        return ToolOutput(text: Formatters.formatEntityCard(entity: person, useColor: false))
    }

    private func handleBrainRecall(_ args: [String: Any]) throws -> ToolOutput {
        guard let db = database else { throw ToolError.noDatabase }
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
            return ToolOutput(text: TextFormatter.formatSearchResults(query: "session:\(sessionId)", results: typedResults, total: typedResults.count))
        }
        let stats = try db.recallStats()
        return ToolOutput(text: TextFormatter.formatStats(StatsResult(payload: stats)))
    }

    private func handleBrainEntity(_ args: [String: Any]) throws -> ToolOutput {
        guard let query = args["query"] as? String else {
            throw ToolError.missingParameter("query")
        }
        guard let db = database else { throw ToolError.noDatabase }
        guard let entity = try db.lookupEntity(query: query) else {
            return ToolOutput(text: "\u{2502} No entity found for \"\(query)\"")
        }
        return ToolOutput(text: TextFormatter.formatEntitySimple(EntityCard(lookupPayload: entity)))
    }

    private func handleBrainDigest(_ args: [String: Any]) throws -> ToolOutput {
        guard let content = args["content"] as? String else {
            throw ToolError.missingParameter("content")
        }
        guard let db = database else { throw ToolError.noDatabase }
        let result = try db.digest(content: content)
        return ToolOutput(text: TextFormatter.formatDigestResult(DigestResult(payload: result)))
    }

    private func handleBrainUpdate(_ args: [String: Any]) throws -> ToolOutput {
        guard let db = database else { throw ToolError.noDatabase }
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
        guard let db = database else { throw ToolError.noDatabase }
        let before = args["before"] as? Int ?? 3
        let after = args["after"] as? Int ?? 3
        let expanded = try db.expandChunk(id: chunkId, before: before, after: after)
        let target = expanded["target"] as? [String: Any] ?? [:]
        let context = expanded["context"] as? [[String: Any]] ?? []
        var lines: [String] = []
        lines.append("\u{250c}\u{2500} brain_expand: \(chunkId)")
        let targetContent = (target["summary"] as? String) ?? (target["content"] as? String) ?? ""
        if !targetContent.isEmpty {
            lines.append("\u{251c}\u{2500} Target")
            lines.append("\u{2502} \(String(targetContent.prefix(200)))")
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
        guard let db = database else { throw ToolError.noDatabase }
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
        guard let db = database else { throw ToolError.noDatabase }

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
        guard let db = database else { throw ToolError.noDatabase }
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
        guard let db = database else { throw ToolError.noDatabase }

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
        guard let db = database else { throw ToolError.noDatabase }
        let batchSize = args["batch_size"] as? Int ?? 1_000
        let cancelRequested = args["cancel"] as? Bool ?? false
        let maxReturnedEvents = 25
        var events: [BrainDatabase.TrigramMaintenanceProgress] = []
        let final = try db.triggerTrigramRebuild(
            batchSize: batchSize,
            shouldCancel: { cancelRequested },
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

        var errorDescription: String? {
            switch self {
            case .unknownTool(let name): return "Unknown tool: \(name)"
            case .missingParameter(let param): return "Missing required parameter: \(param)"
            case .noDatabase: return "Database not available"
            case .notFound(let message): return message
            case .notImplemented(let tool): return "\(tool) not yet implemented in BrainBar (use Python MCP server)"
            case .schemaValidation(let message): return "Schema validation error: \(message)"
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
            "description": "Search through past conversations and learnings. Hybrid semantic + keyword search.",
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
            "description": "Save decisions, learnings, mistakes, ideas, todos to memory.",
            "annotations": MCPRouter.writeAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "content": ["type": "string", "description": "Content to store"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Tags for categorization"],
                    "importance": ["type": "integer", "description": "Importance score (1-10)"],
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any])
        ],
        [
            "name": "brain_recall",
            "description": "Get current working context, browse sessions, or inspect session details.",
            "annotations": MCPRouter.readOnlyAnnotations,
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
            "description": "Look up a known entity in the knowledge graph.",
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
            "description": "Get a person's profile, relations, and linked memories in one call.",
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
            "description": "Ingest raw content (transcripts, docs, articles). Extracts entities, relations, action items.",
            "annotations": MCPRouter.writeAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "content": ["type": "string", "description": "Raw content to digest"],
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any])
        ],
        [
            "name": "brain_update",
            "description": "Update, archive, or merge existing memories.",
            "annotations": MCPRouter.writeIdempotentAnnotations,
            "inputSchema": MCPRouter.limitedInputSchema([
                "type": "object",
                "properties": [
                    "action": ["type": "string", "enum": ["update", "archive", "merge"], "description": "Action to perform"],
                    "chunk_id": ["type": "string", "description": "Chunk ID to update"],
                ] as [String: Any],
                "required": ["action", "chunk_id"]
            ] as [String: Any])
        ],
        [
            "name": "brain_expand",
            "description": "Drill into a specific search result. Returns full content + surrounding chunks.",
            "annotations": MCPRouter.readOnlyAnnotations,
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
            "description": "List, search, or suggest tags across the knowledge base.",
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
            "description": "Mark an old chunk as replaced by a newer one and hide it from default search.",
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
            "description": "Archive a chunk so it stops appearing in default search but remains recoverable.",
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
            "description": "Subscribe an agent to push notifications for matching tags.",
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
            "description": "Remove some or all tag subscriptions for an agent.",
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
            "description": "Acknowledge that an agent processed messages through the given chunk rowid.",
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
