// MCPRouterTests.swift — RED tests for MCP JSON-RPC routing.
//
// The router handles 3 core MCP methods:
// - initialize: handshake, return capabilities
// - tools/list: return all registered tools with schemas
// - tools/call: dispatch to tool handler by name

import XCTest
import SQLite3
@testable import BrainBar

private func collectStringFields(in schema: [String: Any], path: String = "") -> [(String, [String: Any])] {
    guard let type = schema["type"] as? String else {
        return []
    }

    switch type {
    case "string":
        return [(path, schema)]
    case "array":
        if let items = schema["items"] as? [String: Any], (items["type"] as? String) == "string" {
            return [("\(path)[]", items)]
        }
        return []
    case "object":
        let properties = schema["properties"] as? [String: Any] ?? [:]
        return properties.flatMap { propertyName, value -> [(String, [String: Any])] in
            guard let propertySchema = value as? [String: Any] else { return [] }
            let nextPath = path.isEmpty ? propertyName : "\(path).\(propertyName)"
            return collectStringFields(in: propertySchema, path: nextPath)
        }
    default:
        return []
    }
}

private func collectStringArrays(in schema: [String: Any], path: String = "") -> [(String, [String: Any], [String: Any])] {
    guard let type = schema["type"] as? String else {
        return []
    }

    switch type {
    case "array":
        if let items = schema["items"] as? [String: Any], (items["type"] as? String) == "string" {
            return [(path, schema, items)]
        }
        return []
    case "object":
        let properties = schema["properties"] as? [String: Any] ?? [:]
        return properties.flatMap { propertyName, value -> [(String, [String: Any], [String: Any])] in
            guard let propertySchema = value as? [String: Any] else { return [] }
            let nextPath = path.isEmpty ? propertyName : "\(path).\(propertyName)"
            return collectStringArrays(in: propertySchema, path: nextPath)
        }
    default:
        return []
    }
}

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
        XCTAssertEqual(tools?.count, 16, "Should have exactly 16 tools")

        let toolNames = Set(tools?.compactMap { $0["name"] as? String } ?? [])
        let expected: Set<String> = [
            "brain_search", "brain_store", "brain_recall", "brain_entity",
            "brain_digest", "brain_update", "brain_expand", "brain_tags",
            "brain_subscribe", "brain_unsubscribe", "brain_ack",
            "brain_get_person", "brain_supersede", "brain_archive", "brain_enrich",
            "brain_maintenance_rebuild_trigram",
        ]
        XCTAssertEqual(toolNames, expected)
    }

    func testEncodedToolsListEnvelopeStartsWithJSONRPC() throws {
        let router = MCPRouter()
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ])

        let framed = try MCPFraming.encode(response)
        let frame = try XCTUnwrap(String(data: framed, encoding: .utf8))
        let body = try XCTUnwrap(frame.components(separatedBy: "\r\n\r\n").last)

        XCTAssertTrue(
            body.hasPrefix(#"{"jsonrpc":"2.0","id":2,"result":"#),
            "Claude Desktop expects jsonrpc to be the first envelope key for tools/list; got: \(body.prefix(80))"
        )
    }

    func testToolsListPreservesCanonicalAnnotations() throws {
        let router = MCPRouter()
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ])

        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]] ?? []

        XCTAssertEqual(tools.count, 16)
        for tool in tools {
            XCTAssertNotNil(
                tool["annotations"],
                "\(tool["name"] ?? "unknown") should keep annotations in the canonical tools/list response"
            )
        }
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
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
        ])
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
            "brain_get_person": (true, false, true, false),
            "brain_recall": (true, false, true, false),
            "brain_entity": (true, false, true, false),
            "brain_digest": (false, false, false, false),
            "brain_update": (false, false, true, false),
            "brain_expand": (true, false, true, false),
            "brain_tags": (true, false, true, false),
            "brain_supersede": (false, true, false, false),
            "brain_archive": (false, true, false, false),
            "brain_enrich": (false, false, false, false),
            "brain_subscribe": (false, false, false, false),
            "brain_unsubscribe": (false, false, true, false),
            "brain_ack": (false, false, true, false),
            "brain_maintenance_rebuild_trigram": (false, false, true, false),
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

    func testEachToolSchemaBoundsStringInputs() throws {
        for tool in MCPRouter.toolDefinitions {
            let name = tool["name"] as? String ?? "unknown"
            let schema = try XCTUnwrap(tool["inputSchema"] as? [String: Any], "\(name) missing inputSchema")

            for (fieldPath, stringSchema) in collectStringFields(in: schema) {
                XCTAssertNotNil(stringSchema["maxLength"] as? Int, "\(name).\(fieldPath) must declare maxLength")
            }

            for (fieldPath, arraySchema, itemSchema) in collectStringArrays(in: schema) {
                XCTAssertNotNil(arraySchema["maxItems"] as? Int, "\(name).\(fieldPath) must declare maxItems")
                XCTAssertNotNil(itemSchema["maxLength"] as? Int, "\(name).\(fieldPath)[] must declare maxLength")
            }
        }
    }

    func testBrainSearchSchemaIncludesSourceParameter() throws {
        let tool = try XCTUnwrap(MCPRouter.toolDefinitions.first { ($0["name"] as? String) == "brain_search" })
        let schema = try XCTUnwrap(tool["inputSchema"] as? [String: Any])
        let properties = try XCTUnwrap(schema["properties"] as? [String: Any])
        let source = try XCTUnwrap(properties["source"] as? [String: Any])
        let values = try XCTUnwrap(source["enum"] as? [String])
        let description = try XCTUnwrap(source["description"] as? String)

        XCTAssertEqual(values, ["claude_code", "whatsapp", "youtube", "mcp", "all"])
        XCTAssertEqual(description, "Filter by data source. Omit or use 'all' to search everything.")
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

    func testBrainMaintenanceRebuildTrigramToolReturnsProgressMetadata() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-maintenance-\(UUID().uuidString).db"
        defer {
            try? FileManager.default.removeItem(atPath: tempDB)
            try? FileManager.default.removeItem(atPath: tempDB + "-wal")
            try? FileManager.default.removeItem(atPath: tempDB + "-shm")
        }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }
        for index in 0..<30 {
            try db.insertChunk(id: "mcp-trigram-\(index)", content: "MCP trigram maintenance fixture \(index)", sessionId: "mcp", project: "brainlayer", contentType: "assistant_text", importance: 5)
        }
        try sqliteExec(path: tempDB, sql: "DELETE FROM chunks_fts_trigram")

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": [
                "name": "brain_maintenance_rebuild_trigram",
                "arguments": ["batch_size": 1]
            ]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNotEqual(result?["isError"] as? Bool, true)
        let progress = result?["progress"] as? [String: Any]
        XCTAssertEqual(progress?["state"] as? String, "done")
        XCTAssertEqual(progress?["processed"] as? Int, 30)
        XCTAssertEqual(progress?["total"] as? Int, 30)
        let events = result?["events"] as? [[String: Any]] ?? []
        XCTAssertLessThanOrEqual(events.count, 25)
        XCTAssertEqual(events.last?["state"] as? String, "done")
    }

    func testBrainMaintenanceRebuildTrigramCancelReturnsBeforeSchemaWrites() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-maintenance-cancel-\(UUID().uuidString).db"
        defer {
            try? FileManager.default.removeItem(atPath: tempDB)
            try? FileManager.default.removeItem(atPath: tempDB + "-wal")
            try? FileManager.default.removeItem(atPath: tempDB + "-shm")
        }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }
        try sqliteExec(path: tempDB, sql: "DROP TABLE IF EXISTS chunks_fts_trigram")

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": [
                "name": "brain_maintenance_rebuild_trigram",
                "arguments": ["cancel": true]
            ]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNotEqual(result?["isError"] as? Bool, true)
        let progress = result?["progress"] as? [String: Any]
        XCTAssertEqual(progress?["state"] as? String, "cancelled")
        XCTAssertEqual(progress?["processed"] as? Int, 0)
        XCTAssertEqual(progress?["total"] as? Int, 0)
        XCTAssertFalse(try db.tableExists("chunks_fts_trigram"))
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

    func testBrainDigestRejectsOversizedContentAtSchemaLayer() throws {
        let router = MCPRouter()
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": [
                "name": "brain_digest",
                "arguments": [
                    "content": String(repeating: "x", count: 200_001)
                ] as [String: Any]
            ] as [String: Any]
        ])

        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = try XCTUnwrap(content.first?["text"] as? String)

        XCTAssertEqual(result["isError"] as? Bool, true)
        XCTAssertTrue(text.contains("Schema validation error"))
        XCTAssertTrue(text.contains("content length 200001 exceeds maxLength 200000"))
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

    func testBrainSearchPassesSourceFilter() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-source-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        _ = try db.store(content: "Sagit meeting notes", tags: ["meeting"], importance: 6, source: "whatsapp")
        _ = try db.store(content: "Sagit meeting notes", tags: ["meeting"], importance: 6, source: "youtube")

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "Sagit meeting", "source": "whatsapp"] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("Sagit meeting notes"))
        XCTAssertEqual(text.components(separatedBy: "Sagit meeting notes").count - 1, 1, "Only one matching source should be returned")
    }

    func testBrainSearchExcludesAuditRecursionByDefaultAndAllowsOptIn() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-audit-filter-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "audit-recursion-source",
            content: "why restart BrainBar audit recursion contamination exact match",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8,
            tags: "[\"r02\", \"audit\"]"
        )
        try db.insertChunk(
            id: "ordinary-brainbar-memory",
            content: "why restart BrainBar because launchd replaced the old degraded binary",
            sessionId: "s2",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8,
            tags: "[\"brainbar\", \"reliability\"]"
        )

        let router = MCPRouter()
        router.setDatabase(db)
        let defaultResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 160,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "why restart BrainBar", "num_results": 3] as [String: Any]
            ] as [String: Any]
        ])
        let defaultText = ((defaultResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""

        XCTAssertTrue(defaultText.contains("ordinary-bra"), defaultText)
        XCTAssertFalse(defaultText.contains("audit-recurs"), defaultText)

        let optInResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 161,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "why restart BrainBar", "num_results": 3, "include_audit": true] as [String: Any]
            ] as [String: Any]
        ])
        let optInText = ((optInResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""

        XCTAssertTrue(optInText.contains("audit-recurs"), optInText)
    }

    func testBrainSearchSourceAllKeepsKGAugmentation() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-source-all-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertEntity(id: "person-sagit", type: "person", name: "Sagit Stern", metadata: "{}")
        try db.insertEntity(id: "project-techgym", type: "project", name: "TechGym", metadata: "{}")
        try db.insertRelation(sourceId: "person-sagit", targetId: "project-techgym", relationType: "lectures_at")
        try db.insertChunk(
            id: "kg-search-target",
            content: "Sagit Stern delivered the TechGym lecture about retrieval quality.",
            sessionId: "s1",
            project: "test",
            contentType: "assistant_text",
            importance: 8
        )

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 26,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "Sagit Stern TechGym", "source": "all"] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("### ◆ Sagit Stern"))
        XCTAssertTrue(text.contains("→ LECTURES_AT: TechGym"))
        XCTAssertTrue(text.contains("kg-search-ta"))
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

    func testBrainArchiveHidesChunkFromDefaultSearch() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-archive-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "archive-target", content: "Archive this stale memory", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)
        let router = MCPRouter()
        router.setDatabase(db)

        let archiveResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 17,
            "method": "tools/call",
            "params": [
                "name": "brain_archive",
                "arguments": ["chunk_id": "archive-target", "reason": "stale"] as [String: Any]
            ] as [String: Any]
        ])

        let archiveText = ((archiveResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(archiveText.contains("archived"))

        let searchResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 18,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "Archive this stale memory"] as [String: Any]
            ] as [String: Any]
        ])

        let searchText = ((searchResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(searchText.contains("No results found"))
    }

    func testBrainSupersedeHidesOldChunkAndKeepsReplacement() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-supersede-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(id: "old-chunk", content: "TechGym guidance old version", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "new-chunk", content: "TechGym guidance new version", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 9)
        let router = MCPRouter()
        router.setDatabase(db)

        let supersedeResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 19,
            "method": "tools/call",
            "params": [
                "name": "brain_supersede",
                "arguments": [
                    "old_chunk_id": "old-chunk",
                    "new_chunk_id": "new-chunk"
                ] as [String: Any]
            ] as [String: Any]
        ])

        let supersedeText = ((supersedeResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(supersedeText.contains("superseded"))

        let oldSearch = router.handle([
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "old-chunk"] as [String: Any]
            ] as [String: Any]
        ])
        let oldText = ((oldSearch["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(oldText.contains("No results found"))

        let newSearch = router.handle([
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "new-chunk"] as [String: Any]
            ] as [String: Any]
        ])
        let newText = ((newSearch["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(newText.contains("new-chunk"))
    }

    func testBrainGetPersonReturnsRelationsAndMemories() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-person-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertEntity(id: "person-sagit", type: "person", name: "Sagit Stern", metadata: #"{"role":"Founder","preferences":{"topic":"TechGym"}}"#)
        try db.insertEntity(id: "project-techgym", type: "project", name: "TechGym", metadata: "{}")
        try db.insertRelation(sourceId: "person-sagit", targetId: "project-techgym", relationType: "lectures_at")
        try db.insertChunk(id: "mem-sagit-1", content: "Sagit Stern gave the TechGym lecture about search ranking.", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 8)
        try db.linkEntityChunk(entityId: "person-sagit", chunkId: "mem-sagit-1", relevance: 0.9)

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": [
                "name": "brain_get_person",
                "arguments": ["name": "Sagit Stern", "num_memories": 5] as [String: Any]
            ] as [String: Any]
        ])

        let text = ((response["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(text.contains("Entity:"))
        XCTAssertTrue(text.contains("Sagit Stern"))
        XCTAssertTrue(text.contains("lectures_at"))
        XCTAssertTrue(text.contains("TechGym lecture"))
    }

    func testBrainEnrichRealtimeBackfillsEligibleChunksAndStatsReflectIt() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-enrich-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "enrich-target",
            content: "This chunk is long enough to qualify for enrichment and should get a generated summary after backfill runs.",
            sessionId: "s1",
            project: "test",
            contentType: "assistant_text",
            importance: 5
        )

        let router = MCPRouter()
        router.setDatabase(db)
        let enrichResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 23,
            "method": "tools/call",
            "params": [
                "name": "brain_enrich",
                "arguments": ["mode": "realtime", "limit": 1] as [String: Any]
            ] as [String: Any]
        ])

        let enrichText = ((enrichResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(enrichText.contains("Enriched:"))

        let statsResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 24,
            "method": "tools/call",
            "params": [
                "name": "brain_enrich",
                "arguments": ["stats": true] as [String: Any]
            ] as [String: Any]
        ])
        let statsText = ((statsResponse["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(statsText.contains("Enrichment Stats"))
        XCTAssertNotNil(try chunkEnrichedAt(path: tempDB, id: "enrich-target"))
    }

    func testBrainEnrichClampsNegativeLimit() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-enrich-limit-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "enrich-limit-target",
            content: "This chunk is also long enough to qualify for enrichment even if the requested limit is negative.",
            sessionId: "s1",
            project: "test",
            contentType: "assistant_text",
            importance: 5
        )

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 25,
            "method": "tools/call",
            "params": [
                "name": "brain_enrich",
                "arguments": ["mode": "realtime", "limit": -100] as [String: Any]
            ] as [String: Any]
        ])

        let text = ((response["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(text.contains("Enriched:"))
        XCTAssertNotNil(try chunkEnrichedAt(path: tempDB, id: "enrich-limit-target"))
    }

    func testBrainEnrichEmptyChunkIDsDoesNotBroadenScope() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-enrich-empty-ids-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "enrich-empty-ids-target",
            content: "This chunk should stay untouched when brain_enrich receives an explicit empty chunk_ids list.",
            sessionId: "s1",
            project: "test",
            contentType: "assistant_text",
            importance: 5
        )

        let router = MCPRouter()
        router.setDatabase(db)
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 27,
            "method": "tools/call",
            "params": [
                "name": "brain_enrich",
                "arguments": ["mode": "realtime", "limit": 5, "chunk_ids": []] as [String: Any]
            ] as [String: Any]
        ])

        let text = ((response["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String ?? ""
        XCTAssertTrue(text.contains("Attempted: 0"))
        XCTAssertTrue(text.contains("Enriched: 0"))
        XCTAssertNil(try chunkEnrichedAt(path: tempDB, id: "enrich-empty-ids-target"))
    }

    // MARK: - brain_store queue fallback

    func testBrainStoreQueuesWhenWriteHitsTransientSQLiteLock() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        db.exec("PRAGMA busy_timeout = 1")

        let lockDB = try openSQLiteConnection(path: dbPath)
        defer { sqlite3_close(lockDB) }
        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)
        defer { sqlite3_exec(lockDB, "ROLLBACK", nil, nil, nil) }

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Store should queue after transient SQLite lock",
                    "tags": ["queue-fallback"],
                    "importance": 7
                ] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        let text = ((result?["content"] as? [[String: Any]])?.first?["text"] as? String) ?? ""

        XCTAssertNotEqual(result?["isError"] as? Bool, true, "brain_store should queue instead of surfacing a transient lock")
        XCTAssertTrue(text.localizedCaseInsensitiveContains("queued"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: queuePath.path))

        let queuedText = try String(contentsOf: queuePath, encoding: .utf8)
        XCTAssertTrue(queuedText.contains("Store should queue after transient SQLite lock"))
    }

    func testBrainStoreFlushesPendingQueueAfterSuccessfulStore() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let queuedPayload = """
        {"content":"Queued item should flush","tags":["queued"],"importance":4,"source":"mcp"}
        """
        try queuedPayload.write(to: queuePath, atomically: true, encoding: .utf8)

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Live write triggers flush",
                    "tags": ["live"],
                    "importance": 5
                ] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNil(result?["isError"])
        XCTAssertEqual(result?["flushed_count"] as? Int, 1)
        let flushed = result?["_brainbarFlushedQueuedChunks"] as? [[String: Any]]
        XCTAssertEqual(flushed?.count, 1)
        XCTAssertEqual(flushed?.first?["content"] as? String, "Queued item should flush")
        XCTAssertFalse(FileManager.default.fileExists(atPath: queuePath.path), "successful store should drain the pending queue")

        let contents = try chunkContents(path: dbPath)
        XCTAssertTrue(contents.contains("Queued item should flush"))
        XCTAssertTrue(contents.contains("Live write triggers flush"))
    }

    func testBrainStoreFlushKeepsMalformedQueueLines() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let seededQueue = """
        not-json
        {"content":"Queued valid item survives malformed sibling","tags":["queued"],"importance":4,"source":"mcp"}
        """
        try seededQueue.write(to: queuePath, atomically: true, encoding: .utf8)

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Live write tolerates malformed queue lines",
                    "tags": ["live"],
                    "importance": 5
                ] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNil(result?["isError"])
        XCTAssertEqual(result?["flushed_count"] as? Int, 1)
        let flushed = result?["_brainbarFlushedQueuedChunks"] as? [[String: Any]]
        XCTAssertEqual(flushed?.count, 1)
        XCTAssertEqual(flushed?.first?["content"] as? String, "Queued valid item survives malformed sibling")

        let remainingText = try String(contentsOf: queuePath, encoding: .utf8)
        XCTAssertEqual(remainingText.trimmingCharacters(in: .whitespacesAndNewlines), "not-json")

        let contents = try chunkContents(path: dbPath)
        XCTAssertTrue(contents.contains("Queued valid item survives malformed sibling"))
        XCTAssertTrue(contents.contains("Live write tolerates malformed queue lines"))
    }

    func testBrainStoreFlushKeepsInvalidUTF8QueueLines() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let invalidLine = Data([0xC3, 0x28, 0x0A])
        let validLine = Data("""
        {"content":"Queued valid item survives invalid utf8 sibling","tags":["queued"],"importance":4,"source":"mcp"}
        """.utf8)
        try (invalidLine + validLine + Data([0x0A])).write(to: queuePath, options: .atomic)

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let router = MCPRouter()
        router.setDatabase(db)

        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 23,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Live write tolerates invalid utf8 queue lines",
                    "tags": ["live"],
                    "importance": 5
                ] as [String: Any]
            ] as [String: Any]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNil(result?["isError"])
        XCTAssertEqual(result?["flushed_count"] as? Int, 1)
        let flushed = result?["_brainbarFlushedQueuedChunks"] as? [[String: Any]]
        XCTAssertEqual(flushed?.count, 1)
        XCTAssertEqual(flushed?.first?["content"] as? String, "Queued valid item survives invalid utf8 sibling")

        let remainingData = try Data(contentsOf: queuePath)
        XCTAssertEqual(remainingData, invalidLine)

        let contents = try chunkContents(path: dbPath)
        XCTAssertTrue(contents.contains("Queued valid item survives invalid utf8 sibling"))
        XCTAssertTrue(contents.contains("Live write tolerates invalid utf8 queue lines"))
    }

    func testBrainStoreLegacyQueueLinesStayIdempotentAcrossReplay() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let legacyLine = """
        {"content":"Legacy queue line without queue id","tags":["queued"],"importance":4,"source":"mcp"}
        """
        try (legacyLine + "\n").write(to: queuePath, atomically: true, encoding: .utf8)

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let firstFlush = db.flushPendingStores()
        XCTAssertEqual(firstFlush.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: queuePath.path))

        let metadata = try chunkMetadata(path: dbPath, content: "Legacy queue line without queue id")
        XCTAssertNotNil(metadata)
        XCTAssertTrue(metadata?.contains("brainbar_queue_id") == true)

        try (legacyLine + "\n").write(to: queuePath, atomically: true, encoding: .utf8)

        let secondFlush = db.flushPendingStores()
        XCTAssertTrue(secondFlush.isEmpty)

        let contents = try chunkContents(path: dbPath)
        XCTAssertEqual(contents.filter { $0 == "Legacy queue line without queue id" }.count, 1)
    }

    func testBrainStoreLegacyQueuePreservesDuplicatePayloadLines() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let legacyLine = """
        {"content":"Repeated legacy queue payload","tags":["queued"],"importance":4,"source":"mcp"}
        """
        try (legacyLine + "\n" + legacyLine + "\n").write(to: queuePath, atomically: true, encoding: .utf8)

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let flushed = db.flushPendingStores()
        XCTAssertEqual(flushed.count, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: queuePath.path))

        let contents = try chunkContents(path: dbPath)
        XCTAssertEqual(contents.filter { $0 == "Repeated legacy queue payload" }.count, 2)
    }

    func testBrainStoreLegacyQueueDuplicateSurvivesPartialFlushCompaction() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let legacyLine = """
        {"content":"Repeated legacy queue payload with transient failure","tags":["queued"],"importance":4,"source":"mcp"}
        """
        try (legacyLine + "\n" + legacyLine + "\n").write(to: queuePath, atomically: true, encoding: .utf8)

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        try sqliteExec(path: dbPath, sql: """
            CREATE TRIGGER fail_second_repeated_legacy_insert
            BEFORE INSERT ON chunks
            WHEN NEW.content = 'Repeated legacy queue payload with transient failure'
                 AND (SELECT COUNT(*) FROM chunks WHERE content = NEW.content) > 0
            BEGIN
                SELECT RAISE(ABORT, 'fail second repeated legacy insert');
            END;
        """)

        let firstFlush = db.flushPendingStores()
        XCTAssertEqual(firstFlush.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: queuePath.path))
        let compactedQueue = try String(contentsOf: queuePath, encoding: .utf8)
        XCTAssertTrue(compactedQueue.contains("queue_id"))
        XCTAssertFalse(compactedQueue.contains("\n\n"))

        try sqliteExec(path: dbPath, sql: "DROP TRIGGER fail_second_repeated_legacy_insert")

        let secondFlush = db.flushPendingStores()
        XCTAssertEqual(secondFlush.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: queuePath.path))

        let contents = try chunkContents(path: dbPath)
        XCTAssertEqual(contents.filter { $0 == "Repeated legacy queue payload with transient failure" }.count, 2)
    }

    func testQueuePendingStorePreservesConcurrentFirstWrites() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let db = BrainDatabase(path: tempDir.appendingPathComponent("brainbar.db").path)
        defer { db.close() }

        let iterations = 24
        let failureLock = NSLock()
        var failures: [String] = []

        DispatchQueue.concurrentPerform(iterations: iterations) { index in
            do {
                try db.queuePendingStore(
                    content: "Concurrent queue item \(index)",
                    tags: ["queued"],
                    importance: 5,
                    source: "mcp"
                )
            } catch {
                failureLock.lock()
                failures.append(String(describing: error))
                failureLock.unlock()
            }
        }

        XCTAssertTrue(failures.isEmpty, "queuePendingStore should serialize concurrent first writes without errors: \(failures)")

        let queuedText = try String(contentsOf: queuePath, encoding: .utf8)
        let lines = queuedText.split(whereSeparator: \.isNewline)
        XCTAssertEqual(lines.count, iterations)
    }

    func testQueuePendingStoreCreatesPrivateQueueFile() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let db = BrainDatabase(path: tempDir.appendingPathComponent("brainbar.db").path)
        defer { db.close() }

        try db.queuePendingStore(
            content: "Private queued content",
            tags: ["queued"],
            importance: 5,
            source: "mcp"
        )

        XCTAssertEqual(try posixPermissions(path: queuePath), 0o600)
    }

    func testQueuePendingStoreRejectsAppendsPastConfiguredLineCap() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }
        let restoreMaxLines = setPendingStoreQueueMaxLines(3)
        defer { restoreMaxLines() }

        let db = BrainDatabase(path: tempDir.appendingPathComponent("brainbar.db").path)
        defer { db.close() }

        for index in 0..<3 {
            try db.queuePendingStore(
                content: "Queued item \(index)",
                tags: ["queued"],
                importance: 5,
                source: "mcp"
            )
        }

        XCTAssertThrowsError(try db.queuePendingStore(
            content: "Rejected queued item",
            tags: ["queued"],
            importance: 5,
            source: "mcp"
        ))

        let queuedText = try String(contentsOf: queuePath, encoding: .utf8)
        let lines = queuedText.split(whereSeparator: \.isNewline)
        XCTAssertEqual(lines.count, 3)
        XCTAssertTrue(queuedText.contains("Queued item 0"))
        XCTAssertTrue(queuedText.contains("Queued item 2"))
        XCTAssertFalse(queuedText.contains("Rejected queued item"))
        XCTAssertEqual(try posixPermissions(path: queuePath), 0o600)
    }

    func testBrainStoreFlushRewriteKeepsQueueFilePrivate() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let restoreQueuePath = setPendingStoreQueuePath(queuePath)
        defer { restoreQueuePath() }

        let seededQueue = """
        not-json
        {"content":"Private rewritten queued item","tags":["queued"],"importance":4,"source":"mcp"}
        """
        try seededQueue.write(to: queuePath, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: queuePath.path)

        let db = BrainDatabase(path: tempDir.appendingPathComponent("brainbar.db").path)
        defer { db.close() }

        let flushed = db.flushPendingStores()

        XCTAssertEqual(flushed.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: queuePath.path))
        XCTAssertEqual(try posixPermissions(path: queuePath), 0o600)
    }

    func testShouldQueueOnlyTransientSQLiteStoreErrors() throws {
        let tempDir = makeTempTestDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let db = BrainDatabase(path: tempDir.appendingPathComponent("brainbar.db").path)
        defer { db.close() }

        XCTAssertTrue(db.shouldQueueStoreError(BrainDatabase.DBError.prepare(SQLITE_BUSY)))
        XCTAssertTrue(db.shouldQueueStoreError(BrainDatabase.DBError.step(SQLITE_LOCKED)))
        XCTAssertTrue(db.shouldQueueStoreError(BrainDatabase.DBError.exec(SQLITE_BUSY, "database is busy")))

        XCTAssertFalse(db.shouldQueueStoreError(BrainDatabase.DBError.prepare(SQLITE_ERROR)))
        XCTAssertFalse(db.shouldQueueStoreError(BrainDatabase.DBError.step(SQLITE_CORRUPT)))
        XCTAssertFalse(db.shouldQueueStoreError(BrainDatabase.DBError.exec(SQLITE_CANTOPEN, "cannot open database")))
        XCTAssertFalse(db.shouldQueueStoreError(NSError(domain: "test", code: 1)))
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

private func makeTempTestDirectory() -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func setPendingStoreQueuePath(_ path: URL) -> () -> Void {
    let previous = ProcessInfo.processInfo.environment["BRAINBAR_PENDING_STORES_PATH"]
    setenv("BRAINBAR_PENDING_STORES_PATH", path.path, 1)
    return {
        if let previous {
            setenv("BRAINBAR_PENDING_STORES_PATH", previous, 1)
        } else {
            unsetenv("BRAINBAR_PENDING_STORES_PATH")
        }
    }
}

private func setPendingStoreQueueMaxLines(_ maxLines: Int) -> () -> Void {
    let key = "BRAINBAR_PENDING_STORES_MAX_LINES"
    let previous = ProcessInfo.processInfo.environment[key]
    setenv(key, String(maxLines), 1)
    return {
        if let previous {
            setenv(key, previous, 1)
        } else {
            unsetenv(key)
        }
    }
}

private func posixPermissions(path: URL) throws -> Int {
    let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
    return (attributes[.posixPermissions] as? NSNumber)?.intValue ?? 0
}

private func chunkContents(path: String) throws -> [String] {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "MCPRouterTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    var stmt: OpaquePointer?
    let sql = "SELECT content FROM chunks ORDER BY rowid ASC"
    let prepareRC = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
    guard prepareRC == SQLITE_OK else {
        throw NSError(domain: "MCPRouterTests", code: Int(prepareRC))
    }
    defer { sqlite3_finalize(stmt) }

    var results: [String] = []
    while sqlite3_step(stmt) == SQLITE_ROW {
        if let value = sqlite3_column_text(stmt, 0) {
            results.append(String(cString: value))
        }
    }
    return results
}

private func chunkMetadata(path: String, content: String) throws -> String? {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "MCPRouterTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    var stmt: OpaquePointer?
    let sql = "SELECT metadata FROM chunks WHERE content = ? ORDER BY rowid DESC LIMIT 1"
    let prepareRC = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
    guard prepareRC == SQLITE_OK else {
        throw NSError(domain: "MCPRouterTests", code: Int(prepareRC))
    }
    defer { sqlite3_finalize(stmt) }

    let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    sqlite3_bind_text(stmt, 1, content, -1, transient)
    guard sqlite3_step(stmt) == SQLITE_ROW else {
        return nil
    }
    guard let value = sqlite3_column_text(stmt, 0) else {
        return nil
    }
    return String(cString: value)
}

private func chunkEnrichedAt(path: String, id: String) throws -> String? {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "MCPRouterTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    var stmt: OpaquePointer?
    let sql = "SELECT enriched_at FROM chunks WHERE id = ? LIMIT 1"
    let prepareRC = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
    guard prepareRC == SQLITE_OK else {
        throw NSError(domain: "MCPRouterTests", code: Int(prepareRC))
    }
    defer { sqlite3_finalize(stmt) }

    let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    sqlite3_bind_text(stmt, 1, id, -1, transient)
    guard sqlite3_step(stmt) == SQLITE_ROW else {
        return nil
    }
    guard let value = sqlite3_column_text(stmt, 0) else {
        return nil
    }
    return String(cString: value)
}

private func sqliteExec(path: String, sql: String) throws {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "MCPRouterTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    let execRC = sqlite3_exec(db, sql, nil, nil, nil)
    guard execRC == SQLITE_OK else {
        throw NSError(domain: "MCPRouterTests", code: Int(execRC))
    }
}

private func openSQLiteConnection(path: String) throws -> OpaquePointer {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "MCPRouterTests", code: Int(rc))
    }
    return db
}
