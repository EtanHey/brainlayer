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
        let experimental = capabilities?["experimental"] as? [String: Any]
        XCTAssertEqual((experimental?["claude/channel"] as? [String: Any])?.isEmpty, true)
    }

    // MARK: - Tools list

    func testToolsListReturnsAllEightTools() throws {
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
        XCTAssertEqual(tools?.count, 10, "Should have exactly 10 tools")

        let toolNames = Set(tools?.compactMap { $0["name"] as? String } ?? [])
        let expected: Set<String> = [
            "brain_search", "brain_store", "brain_recall", "brain_entity",
            "brain_digest", "brain_update", "brain_expand", "brain_tags",
            "brain_subscribe", "brain_unsubscribe"
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

        // Parse the result text as JSON array
        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "[]"
        let results = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [[String: Any]] ?? []

        XCTAssertEqual(results.count, 1, "Should return only brainbar project result")
        XCTAssertEqual(results.first?["project"] as? String, "brainbar")
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

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "[]"
        let results = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [[String: Any]] ?? []

        XCTAssertEqual(results.count, 1, "Should return only high-importance result")
        XCTAssertEqual(results.first?["chunk_id"] as? String, "i-1")
    }

    func testBrainSearchUnreadOnlyFiltersReadChunks() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-unread-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "read-1", content: "Agent message already delivered", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"agent-message\"]")
        try db.insertChunk(id: "unread-1", content: "Agent message still unread", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"agent-message\"]")
        _ = try db.upsertSubscription(agentID: "agent-1", tags: ["agent-message"])
        try db.markChunkRead(agentID: "agent-1", chunkID: "read-1")

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

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "[]"
        let results = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [[String: Any]] ?? []

        XCTAssertEqual(results.count, 1, "Should return only unread chunks for the subscriber")
        XCTAssertEqual(results.first?["chunk_id"] as? String, "unread-1")
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
