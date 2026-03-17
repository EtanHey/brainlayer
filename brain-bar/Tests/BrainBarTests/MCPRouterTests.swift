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
        XCTAssertEqual(tools?.count, 8, "Should have exactly 8 tools")

        let toolNames = Set(tools?.compactMap { $0["name"] as? String } ?? [])
        let expected: Set<String> = [
            "brain_search", "brain_store", "brain_recall", "brain_entity",
            "brain_digest", "brain_update", "brain_expand", "brain_tags"
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
