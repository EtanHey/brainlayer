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
            // Belt-and-suspenders: some clients send this with an id.
            return [:]
        case "tools/list":
            return handleToolsList(id: id)
        case "tools/call":
            return handleToolsCall(id: id, params: request["params"] as? [String: Any] ?? [:])
        case "resources/list":
            return jsonRPCResult(id: id, result: ["resources": [Any]()])
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
                    "tools": ["listChanged": false]
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
        guard let db = database else {
            return "[]"
        }
        let results = try db.search(query: query, limit: limit)
        let data = try JSONSerialization.data(withJSONObject: results)
        return String(data: data, encoding: .utf8) ?? "[]"
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
        let id = try db.store(content: content, tags: tags, importance: importance, source: "mcp")
        return jsonEncode(["chunk_id": id, "status": "stored"])
    }

    private func handleBrainRecall(_ args: [String: Any]) throws -> String {
        throw ToolError.notImplemented("brain_recall")
    }

    private func handleBrainEntity(_ args: [String: Any]) throws -> String {
        guard let _ = args["query"] as? String else {
            throw ToolError.missingParameter("query")
        }
        throw ToolError.notImplemented("brain_entity")
    }

    private func handleBrainDigest(_ args: [String: Any]) throws -> String {
        guard args["content"] is String else {
            throw ToolError.missingParameter("content")
        }
        throw ToolError.notImplemented("brain_digest")
    }

    private func handleBrainUpdate(_ args: [String: Any]) throws -> String {
        guard let _ = args["action"] as? String else {
            throw ToolError.missingParameter("action")
        }
        throw ToolError.notImplemented("brain_update")
    }

    private func handleBrainExpand(_ args: [String: Any]) throws -> String {
        guard let _ = args["chunk_id"] as? String else {
            throw ToolError.missingParameter("chunk_id")
        }
        throw ToolError.notImplemented("brain_expand")
    }

    private func handleBrainTags(_ args: [String: Any]) throws -> String {
        throw ToolError.notImplemented("brain_tags")
    }

    /// Safe JSON encoding — never use string interpolation with user data.
    private func jsonEncode(_ dict: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
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
    ]
}
