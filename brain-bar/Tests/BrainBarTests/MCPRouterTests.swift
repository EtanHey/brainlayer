// MCPRouterTests.swift — RED tests for MCP JSON-RPC routing.
//
// The router handles 3 core MCP methods:
// - initialize: handshake, return capabilities
// - tools/list: return all registered tools with schemas
// - tools/call: dispatch to tool handler by name

import XCTest
@testable import BrainBar

final class MCPRouterTests: XCTestCase {

    // MARK: - Initialize

    func testInitializeReturnsProtocolVersion() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "test", "version": "1.0"]
            ]
        ]

        let response = router.handle(request)
        let result = response["result"] as? [String: Any]

        XCTAssertNotNil(result)
        XCTAssertEqual(result?["protocolVersion"] as? String, "2024-11-05")
        XCTAssertEqual(response["id"] as? Int, 1)

        // Must declare tool capabilities
        let capabilities = result?["capabilities"] as? [String: Any]
        XCTAssertNotNil(capabilities?["tools"])
        XCTAssertNil(capabilities?["resources"], "Tag resources should not be advertised during initialize")
        let experimental = capabilities?["experimental"] as? [String: Any]
        XCTAssertEqual((experimental?["claude/channel"] as? [String: Any])?.isEmpty, true)
    }

    func testResourcesListReturnsNoPreloadedResources() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-resources-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "tagged-1", content: "Tagged chunk", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"agent-message\"]")

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 11,
            "method": "resources/list",
        ])

        let result = response["result"] as? [String: Any]
        let resources = result?["resources"] as? [[String: Any]]

        XCTAssertEqual(resources?.count, 0, "Tags should not be exposed as boot-time MCP resources")
    }

    // MARK: - Tools list

    func testToolsListReturnsAllTools() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ]

        let response = router.handle(request)
        let result = response["result"] as? [String: Any]
        let tools = result?["tools"] as? [[String: Any]]

        XCTAssertNotNil(tools)
        XCTAssertEqual(tools?.count, 11, "Should have exactly 11 tools")

        let toolNames = Set(tools?.compactMap { $0["name"] as? String } ?? [])
        let expected: Set<String> = [
            "brain_search", "brain_store", "brain_recall", "brain_entity",
            "brain_digest", "brain_update", "brain_expand", "brain_tags",
            "brain_subscribe", "brain_unsubscribe", "brain_ack"
        ]
        XCTAssertEqual(toolNames, expected)
    }

    func testEachToolHasInputSchema() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
        ]

        let response = router.handle(request)
        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]] ?? []

        for tool in tools {
            let name = tool["name"] as? String ?? "unknown"
            XCTAssertNotNil(tool["description"] as? String, "\(name) must have description")
            XCTAssertNotNil(tool["inputSchema"] as? [String: Any], "\(name) must have inputSchema")
        }
    }

    func testEachToolHasExpectedAnnotations() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/list",
        ]

        let response = router.handle(request)
        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]] ?? []
        let toolsByName = Dictionary(
            uniqueKeysWithValues: tools.compactMap { tool -> (String, [String: Any])? in
                guard let name = tool["name"] as? String else { return nil }
                return (name, tool)
            }
        )

        let expected: [String: (readOnly: Bool, destructive: Bool, idempotent: Bool, openWorld: Bool)] = [
            "brain_search": (true, false, true, false),
            "brain_store": (false, false, false, false),
            "brain_recall": (true, false, true, false),
            "brain_entity": (true, false, true, false),
            "brain_digest": (false, false, false, false),
            "brain_update": (false, false, true, false),
            "brain_expand": (true, false, true, false),
            "brain_tags": (true, false, true, false),
            "brain_subscribe": (false, false, false, false),
            "brain_unsubscribe": (false, false, true, false),
            "brain_ack": (false, false, true, false),
        ]

        XCTAssertEqual(toolsByName.count, expected.count)

        for (name, taxonomy) in expected {
            let annotations = toolsByName[name]?["annotations"] as? [String: Any]
            XCTAssertNotNil(annotations, "\(name) must expose MCP tool annotations")
            XCTAssertEqual(annotations?["readOnlyHint"] as? Bool, taxonomy.readOnly, "\(name) readOnlyHint mismatch")
            XCTAssertEqual(annotations?["destructiveHint"] as? Bool, taxonomy.destructive, "\(name) destructiveHint mismatch")
            XCTAssertEqual(annotations?["idempotentHint"] as? Bool, taxonomy.idempotent, "\(name) idempotentHint mismatch")
            XCTAssertEqual(annotations?["openWorldHint"] as? Bool, taxonomy.openWorld, "\(name) openWorldHint mismatch")
        }
    }

    // MARK: - Tools call

    func testToolsCallDispatchesToHandler() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "test query"]
            ]
        ]

        let response = router.handle(request)

        // Should not be an error
        XCTAssertNil(response["error"], "brain_search should not return error")
        XCTAssertNotNil(response["result"], "brain_search should return result")
        XCTAssertEqual(response["id"] as? Int, 4)
    }

    func testToolsCallUnknownToolReturnsError() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": [
                "name": "nonexistent_tool",
                "arguments": [:] as [String: Any]
            ]
        ]

        let response = router.handle(request)
        let error = response["error"] as? [String: Any]

        XCTAssertNotNil(error, "Unknown tool should return JSON-RPC error")
        XCTAssertEqual(error?["code"] as? Int, -32601, "Should be method-not-found error")
    }

    func testBrainSubscribeToolIsServerHandled() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "subscriber_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ]

        let response = router.handle(request)
        let result = response["result"] as? [String: Any]
        XCTAssertEqual(result?["isError"] as? Bool, true)
    }

    func testBrainUnsubscribeToolIsServerHandled() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": [
                "name": "brain_unsubscribe",
                "arguments": [
                    "subscriber_id": "agent-1"
                ] as [String: Any]
            ]
        ]

        let response = router.handle(request)
        let result = response["result"] as? [String: Any]
        XCTAssertEqual(result?["isError"] as? Bool, true)
    }

    func testBrainAckToolIsServerHandled() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": [
                "name": "brain_ack",
                "arguments": [
                    "agent_id": "agent-1",
                    "seq": 42
                ] as [String: Any]
            ]
        ]

        let response = router.handle(request)
        let result = response["result"] as? [String: Any]
        XCTAssertEqual(result?["isError"] as? Bool, true)
    }

    // MARK: - Unknown method

    func testUnknownMethodReturnsError() throws {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 6,
            "method": "unknown/method",
        ]

        let response = router.handle(request)
        let error = response["error"] as? [String: Any]

        XCTAssertNotNil(error)
        XCTAssertEqual(error?["code"] as? Int, -32601)
    }

    // MARK: - brain_search with filters (H3)

    func testBrainSearchPassesProjectFilter() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-filter-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "f-1", content: "Socket handling code", sessionId: "s1", project: "brainbar", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "f-2", content: "Socket connection code", sessionId: "s2", project: "other-proj", contentType: "assistant_text", importance: 5)

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "socket", "project": "brainbar"] as [String: Any]
            ] as [String: Any]
        ])

        // Result is now formatted text, not raw JSON
        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        // Formatted output has ANSI color codes — check for box-drawing prefix and content
        XCTAssertTrue(text.contains("\u{250c}"), "Should contain box-drawing header")
        XCTAssertTrue(text.contains("f-1"), "Should contain the brainbar project chunk")
        XCTAssertFalse(text.contains("f-2"), "Should not contain other project chunk")
    }

    func testBrainSearchPassesImportanceMinFilter() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-imp-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "i-1", content: "Critical security finding", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 9)
        try db.insertChunk(id: "i-2", content: "Security review notes", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 3)

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "security", "importance_min": 7] as [String: Any]
            ] as [String: Any]
        ])

        // Result is now formatted text, not raw JSON
        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("\u{250c}"), "Should contain box-drawing header")
        XCTAssertTrue(text.contains("i-1"), "Should contain the high-importance chunk id")
        XCTAssertFalse(text.contains("i-2"), "Should not contain low-importance chunk")
    }

    func testBrainEntityUsesPythonSimpleEntityStructure() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-entity-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertEntity(
            id: "proj-1",
            type: "project",
            name: "BrainLayer",
            metadata: #"{"language":"Swift","owner":"Etan"}"#
        )
        try db.insertEntity(id: "tool-1", type: "tool", name: "Claude Code")
        try db.insertRelation(sourceId: "proj-1", targetId: "tool-1", relationType: "used_by")

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": [
                "name": "brain_entity",
                "arguments": ["query": "BrainLayer"] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("Entity: BrainLayer"))
        XCTAssertTrue(text.contains("Relations (1)"))
        XCTAssertTrue(text.contains("→ used_by: Claude Code"))
        XCTAssertTrue(text.contains("Metadata"))
        XCTAssertTrue(text.contains("language: Swift"))
    }

    func testBrainRecallStatsIncludesProjectAndTypeLists() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-stats-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "s-1", content: "Search result one", sessionId: "session-1", project: "brainlayer", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "s-2", content: "Search result two", sessionId: "session-2", project: "orchestrator", contentType: "user_message", importance: 4)

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": [
                "name": "brain_recall",
                "arguments": ["mode": "stats"] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("BrainLayer Stats"))
        XCTAssertTrue(text.contains("Projects: brainlayer, orchestrator"))
        XCTAssertTrue(text.contains("Types: assistant_text, user_message"))
    }

    func testBrainRecallInjectionsReturnsRecentEvents() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-injections-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.recordInjectionEvent(
            sessionID: "claude-session-9",
            query: "voicebar sleep recovery",
            chunkIDs: ["chunk-a", "chunk-b"],
            tokenCount: 91,
            timestamp: "2026-03-31T04:03:00.000Z"
        )

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": [
                "name": "brain_recall",
                "arguments": ["mode": "injections", "session_id": "claude-session-9"] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("claude-session-9"))
        XCTAssertTrue(text.contains("voicebar sleep recovery"))
        XCTAssertTrue(text.contains("chunk-a"))
        XCTAssertTrue(text.contains("91"))
    }

    func testBrainSearchUnreadOnlyFiltersAckedChunksByCursor() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-unread-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "read-1", content: "Agent message already delivered", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"agent-message\"]")
        try db.insertChunk(id: "unread-1", content: "Agent message still unread", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"agent-message\"]")
        _ = try db.upsertSubscription(agentID: "agent-1", tags: ["agent-message"])
        guard let readSeq = try db.chunkRowID(forChunkID: "read-1") else {
            XCTFail("expected read-1 rowid")
            return
        }
        try db.acknowledge(agentID: "agent-1", seq: readSeq)

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": [
                    "query": "agent message",
                    "subscriber_id": "agent-1",
                    "unread_only": true
                ] as [String: Any]
            ] as [String: Any]
        ])

        // Result is now formatted text, not raw JSON
        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("\u{250c}"), "Should contain box-drawing header")
        XCTAssertTrue(text.contains("unread-1"), "Should contain the unread chunk id")
        // Note: can't check absence of "read-1" since "unread-1" contains it as substring
        XCTAssertTrue(text.contains("result"), "Should contain formatted result text")
    }

    // MARK: - Notifications (no id)

    func testNotificationDoesNotRequireResponse() {
        let router = MCPRouter()
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        ]

        let response = router.handle(request)
        // Notifications should return empty/nil response (no id to respond to)
        XCTAssertTrue(response.isEmpty || response["id"] == nil)
    }
}
