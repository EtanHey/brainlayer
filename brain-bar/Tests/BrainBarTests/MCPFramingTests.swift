// MCPFramingTests.swift — RED tests for MCP Content-Length framing parser.
//
// MCP uses Content-Length headers (like LSP):
//   Content-Length: N\r\n\r\n{json payload of exactly N bytes}
//
// The framing parser must:
// 1. Parse complete messages from a byte buffer
// 2. Handle partial messages (return nil, keep buffer)
// 3. Handle multiple messages in a single buffer
// 4. Handle split headers (Content-Length arrives in one read, body in next)

import XCTest
@testable import BrainBar

final class MCPFramingTests: XCTestCase {

    // MARK: - Single complete message

    func testParsesSingleCompleteMessage() throws {
        var framing = MCPFraming()
        let json = #"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#
        let frame = "Content-Length: \(json.utf8.count)\r\n\r\n\(json)"

        framing.append(Data(frame.utf8))
        let messages = framing.extractMessages()

        XCTAssertEqual(messages.count, 1)
        let msg = try XCTUnwrap(messages.first)
        XCTAssertEqual(msg["method"] as? String, "initialize")
        XCTAssertEqual(msg["id"] as? Int, 1)
    }

    // MARK: - Partial message (header only)

    func testBuffersPartialMessage() {
        var framing = MCPFraming()
        let json = #"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#
        let header = "Content-Length: \(json.utf8.count)\r\n\r\n"

        // Only send header, no body yet
        framing.append(Data(header.utf8))
        let messages = framing.extractMessages()
        XCTAssertTrue(messages.isEmpty, "Should not yield message when body is incomplete")
    }

    // MARK: - Partial message completed in second append

    func testCompletesPartialMessageOnSecondAppend() throws {
        var framing = MCPFraming()
        let json = #"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#
        let header = "Content-Length: \(json.utf8.count)\r\n\r\n"

        framing.append(Data(header.utf8))
        XCTAssertTrue(framing.extractMessages().isEmpty)

        framing.append(Data(json.utf8))
        let messages = framing.extractMessages()
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages.first?["method"] as? String, "tools/list")
    }

    // MARK: - Multiple messages in one buffer

    func testParsesMultipleMessagesInOneBuffer() {
        var framing = MCPFraming()
        let json1 = #"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#
        let json2 = #"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#
        let frame1 = "Content-Length: \(json1.utf8.count)\r\n\r\n\(json1)"
        let frame2 = "Content-Length: \(json2.utf8.count)\r\n\r\n\(json2)"

        framing.append(Data((frame1 + frame2).utf8))
        let messages = framing.extractMessages()

        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["method"] as? String, "initialize")
        XCTAssertEqual(messages[1]["method"] as? String, "tools/list")
    }

    // MARK: - Split header across reads

    func testHandlesSplitHeader() throws {
        var framing = MCPFraming()
        let json = #"{"jsonrpc":"2.0","id":3,"method":"tools/call"}"#
        let fullFrame = "Content-Length: \(json.utf8.count)\r\n\r\n\(json)"

        // Split in the middle of "Content-Length"
        let splitPoint = 8
        let part1 = Data(Array(fullFrame.utf8)[0..<splitPoint])
        let part2 = Data(Array(fullFrame.utf8)[splitPoint...])

        framing.append(part1)
        XCTAssertTrue(framing.extractMessages().isEmpty)

        framing.append(part2)
        let messages = framing.extractMessages()
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages.first?["method"] as? String, "tools/call")
    }

    // MARK: - Encodes response with Content-Length framing

    func testEncodesResponse() throws {
        let response: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "result": ["protocolVersion": "2024-11-05"]
        ]
        let framed = try MCPFraming.encode(response)
        let str = try XCTUnwrap(String(data: framed, encoding: .utf8))

        XCTAssertTrue(str.hasPrefix("Content-Length: "))
        XCTAssertTrue(str.contains("\r\n\r\n"))
        XCTAssertTrue(str.contains("protocolVersion"))
    }

    // MARK: - Empty body

    func testRejectsZeroContentLength() {
        var framing = MCPFraming()
        framing.append(Data("Content-Length: 0\r\n\r\n".utf8))
        let messages = framing.extractMessages()
        XCTAssertTrue(messages.isEmpty, "Zero-length messages should be discarded")
    }
}
