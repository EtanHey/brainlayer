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
    private var database: BrainDatabase?

    /// Inject database for tool handlers.
    func setDatabase(_ db: BrainDatabase) {
        self.database = db
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
        case "resources/read":
            return handleResourcesRead(id: id, params: request["params"] as? [String: Any] ?? [:])
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
                    "resources": [
                        "subscribe": true,
                        "listChanged": false
                    ],
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
        let resources = (try? database?.resourceList()) ?? []
        return jsonRPCResult(id: id, result: ["resources": resources])
    }

    private func handleResourcesRead(id: Any, params: [String: Any]) -> [String: Any] {
        guard let uri = params["uri"] as? String else {
            return jsonRPCError(id: id, code: -32602, message: "Missing resource uri")
        }

        do {
            let rows = try database?.resourceRead(uri: uri) ?? []
            let text = String(data: try JSONSerialization.data(withJSONObject: rows), encoding: .utf8) ?? "[]"
            return jsonRPCResult(id: id, result: [
                "contents": [[
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text
                ]]
            ])
        } catch {
            return jsonRPCError(id: id, code: -32000, message: error.localizedDescription)
        }
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
            let result = try dispatchTool(name: toolName, arguments: arguments)
            return [
                "jsonrpc": "2.0",
                "id": id,
                "result": [
                    "content": [
                        ["type": "text", "text": result]
                    ]
                ] as [String: Any]
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

    private func dispatchTool(name: String, arguments: [String: Any]) throws -> String {
        switch name {
        case "brain_search":
            return try handleBrainSearch(arguments)
        case "brain_store":
            return try handleBrainStore(arguments)
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
        case "brain_subscribe":
            return try handleBrainSubscribe(arguments)
        case "brain_unsubscribe":
            return try handleBrainUnsubscribe(arguments)
        case "brain_ack":
            return try handleBrainAck(arguments)
        default:
            throw ToolError.unknownTool(name)
        }
    }

    // MARK: - Tool Handlers

    private func handleBrainSearch(_ args: [String: Any]) throws -> String {
        guard let query = args["query"] as? String else {
            throw ToolError.missingParameter("query")
        }
        let limit = min(args["num_results"] as? Int ?? 5, 100)
        let project = args["project"] as? String
        let tag = args["tag"] as? String
        let subscriberID = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String)
        let unreadOnly = args["unread_only"] as? Bool ?? false
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
        let results = try db.search(
            query: query,
            limit: limit,
            project: project,
            tag: tag,
            importanceMin: importanceMin,
            subscriberID: subscriberID,
            unreadOnly: unreadOnly
        )
        return Formatters.formatSearchResults(query: query, results: results, total: results.count)
    }

    private func handleBrainStore(_ args: [String: Any]) throws -> String {
        guard let content = args["content"] as? String else {
            throw ToolError.missingParameter("content")
        }
        let tags = args["tags"] as? [String] ?? []
        let importance = args["importance"] as? Int ?? 5
        guard let db = database else {
            throw ToolError.noDatabase
        }
        let stored = try db.store(content: content, tags: tags, importance: importance, source: "mcp")
        return Formatters.formatStoreResult(chunkId: stored.chunkID)
    }

    private func handleBrainRecall(_ args: [String: Any]) throws -> String {
        guard let db = database else { throw ToolError.noDatabase }
        let mode = args["mode"] as? String ?? "stats"
        if mode == "context" {
            let sessionId = args["session_id"] as? String ?? ""
            if sessionId.isEmpty {
                let stats = try db.recallStats()
                return Formatters.formatStats(stats: stats)
            }
            let results = try db.recallSession(sessionId: sessionId, limit: 20)
            return Formatters.formatSearchResults(query: "session:\(sessionId)", results: results, total: results.count)
        }
        let stats = try db.recallStats()
        return Formatters.formatStats(stats: stats)
    }

    private func handleBrainEntity(_ args: [String: Any]) throws -> String {
        guard let query = args["query"] as? String else {
            throw ToolError.missingParameter("query")
        }
        guard let db = database else { throw ToolError.noDatabase }
        guard let entity = try db.lookupEntity(query: query) else {
            return "\u{2502} No entity found for \"\(query)\""
        }
        return Formatters.formatEntityCard(entity: entity)
    }

    private func handleBrainDigest(_ args: [String: Any]) throws -> String {
        guard let content = args["content"] as? String else {
            throw ToolError.missingParameter("content")
        }
        guard let db = database else { throw ToolError.noDatabase }
        let result = try db.digest(content: content)
        return Formatters.formatDigestResult(result: result)
    }

    private func handleBrainUpdate(_ args: [String: Any]) throws -> String {
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
        return "\u{2714} Updated \(chunkId)" + (importance != nil ? " imp:\(importance!)" : "") + (tags != nil ? " tags:\(tags!.joined(separator: ","))" : "")
    }

    private func handleBrainExpand(_ args: [String: Any]) throws -> String {
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
        return lines.joined(separator: "\n")
    }

    private func handleBrainTags(_ args: [String: Any]) throws -> String {
        guard let db = database else { throw ToolError.noDatabase }
        let query = args["query"] as? String
        let limit = args["limit"] as? Int ?? 50
        let tags = try db.listTags(query: query, limit: limit)
        if tags.isEmpty {
            return "\u{2502} No tags found" + (query != nil ? " matching \"\(query!)\"" : "")
        }
        var lines: [String] = []
        lines.append("\u{250c}\u{2500} brain_tags (\(tags.count) tags)")
        for t in tags {
            let name = t["tag"] as? String ?? ""
            let count = t["count"] as? Int ?? 0
            lines.append("\u{2502}  \(name) (\(count))")
        }
        lines.append("\u{2514}\u{2500}")
        return lines.joined(separator: "\n")
    }

    private func handleBrainSubscribe(_ args: [String: Any]) throws -> String {
        guard let _ = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String) else {
            throw ToolError.missingParameter("agent_id")
        }
        guard let _ = args["tags"] as? [String] else {
            throw ToolError.missingParameter("tags")
        }
        throw ToolError.notImplemented("brain_subscribe")
    }

    private func handleBrainUnsubscribe(_ args: [String: Any]) throws -> String {
        guard let _ = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String) else {
            throw ToolError.missingParameter("agent_id")
        }
        throw ToolError.notImplemented("brain_unsubscribe")
    }

    private func handleBrainAck(_ args: [String: Any]) throws -> String {
        guard let _ = (args["agent_id"] as? String) ?? (args["subscriber_id"] as? String) else {
            throw ToolError.missingParameter("agent_id")
        }
        guard args["seq"] is Int || args["seq"] is Int64 else {
            throw ToolError.missingParameter("seq")
        }
        throw ToolError.notImplemented("brain_ack")
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

    enum ToolError: LocalizedError {
        case unknownTool(String)
        case missingParameter(String)
        case noDatabase
        case notImplemented(String)

        var errorDescription: String? {
            switch self {
            case .unknownTool(let name): return "Unknown tool: \(name)"
            case .missingParameter(let param): return "Missing required parameter: \(param)"
            case .noDatabase: return "Database not available"
            case .notImplemented(let tool): return "\(tool) not yet implemented in BrainBar (use Python MCP server)"
            }
        }
    }

    // MARK: - Tool Definitions

    nonisolated(unsafe) static let toolDefinitions: [[String: Any]] = [
        [
            "name": "brain_search",
            "description": "Search through past conversations and learnings. Hybrid semantic + keyword search.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Natural language search query"],
                    "num_results": ["type": "integer", "description": "Number of results (default: 5, max: 100)"],
                    "project": ["type": "string", "description": "Filter by project name"],
                    "tag": ["type": "string", "description": "Filter by tag"],
                    "importance_min": ["type": "number", "description": "Minimum importance score (1-10)"],
                    "agent_id": ["type": "string", "description": "Optional stable agent id for unread filtering"],
                    "unread_only": ["type": "boolean", "description": "Return only chunks not yet acknowledged by agent_id"],
                    "detail": ["type": "string", "enum": ["compact", "full"], "description": "Result detail level"],
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any]
        ],
        [
            "name": "brain_store",
            "description": "Save decisions, learnings, mistakes, ideas, todos to memory.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "content": ["type": "string", "description": "Content to store"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Tags for categorization"],
                    "importance": ["type": "integer", "description": "Importance score (1-10)"],
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any]
        ],
        [
            "name": "brain_recall",
            "description": "Get current working context, browse sessions, or inspect session details.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "mode": ["type": "string", "enum": ["context", "sessions", "operations", "plan", "summary", "stats"], "description": "Recall mode"],
                    "session_id": ["type": "string", "description": "Session ID for operations/summary mode"],
                ] as [String: Any],
            ] as [String: Any]
        ],
        [
            "name": "brain_entity",
            "description": "Look up a known entity in the knowledge graph.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Entity name to look up"],
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any]
        ],
        [
            "name": "brain_digest",
            "description": "Ingest raw content (transcripts, docs, articles). Extracts entities, relations, action items.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "content": ["type": "string", "description": "Raw content to digest"],
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any]
        ],
        [
            "name": "brain_update",
            "description": "Update, archive, or merge existing memories.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "action": ["type": "string", "enum": ["update", "archive", "merge"], "description": "Action to perform"],
                    "chunk_id": ["type": "string", "description": "Chunk ID to update"],
                ] as [String: Any],
                "required": ["action", "chunk_id"]
            ] as [String: Any]
        ],
        [
            "name": "brain_expand",
            "description": "Drill into a specific search result. Returns full content + surrounding chunks.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "chunk_id": ["type": "string", "description": "Chunk ID to expand"],
                    "before": ["type": "integer", "description": "Context chunks before (default: 3)"],
                    "after": ["type": "integer", "description": "Context chunks after (default: 3)"],
                ] as [String: Any],
                "required": ["chunk_id"]
            ] as [String: Any]
        ],
        [
            "name": "brain_tags",
            "description": "List, search, or suggest tags across the knowledge base.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Optional search query to filter tags"],
                ] as [String: Any],
            ] as [String: Any]
        ],
        [
            "name": "brain_subscribe",
            "description": "Subscribe an agent to push notifications for matching tags.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "agent_id": ["type": "string", "description": "Stable agent identifier"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Tags to receive live notifications for"],
                ] as [String: Any],
                "required": ["agent_id", "tags"]
            ] as [String: Any]
        ],
        [
            "name": "brain_unsubscribe",
            "description": "Remove some or all tag subscriptions for an agent.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "agent_id": ["type": "string", "description": "Stable agent identifier"],
                    "tags": ["type": "array", "items": ["type": "string"], "description": "Optional subset of tags to remove"],
                ] as [String: Any],
                "required": ["agent_id"]
            ] as [String: Any]
        ],
        [
            "name": "brain_ack",
            "description": "Acknowledge that an agent processed messages through the given chunk rowid.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "agent_id": ["type": "string", "description": "Stable agent identifier"],
                    "seq": ["type": "integer", "description": "Highest chunk rowid acknowledged by the agent"],
                ] as [String: Any],
                "required": ["agent_id", "seq"]
            ] as [String: Any]
        ],
    ]
}
