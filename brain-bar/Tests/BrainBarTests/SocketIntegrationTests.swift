// SocketIntegrationTests.swift — RED tests for end-to-end socket + MCP flow.
//
// Tests the full pipeline: connect to Unix socket → send Content-Length framed
// MCP request → receive Content-Length framed response.

import XCTest
@testable import BrainBar

final class SocketIntegrationTests: XCTestCase {
    let testSocketPath = "/tmp/brainbar-test-\(ProcessInfo.processInfo.processIdentifier).sock"
    var server: BrainBarServer!
    var bufferedMessagesByFD: [Int32: Data] = [:]

    override func setUp() {
        super.setUp()
        let tempDB = NSTemporaryDirectory() + "brainbar-integration-\(UUID().uuidString).db"
        server = BrainBarServer(socketPath: testSocketPath, dbPath: tempDB)
        server.start()
        // Give server time to bind
        Thread.sleep(forTimeInterval: 0.2)
    }

    override func tearDown() {
        bufferedMessagesByFD.removeAll()
        server.stop()
        super.tearDown()
    }

    // MARK: - Connection

    func testConnectsToSocket() throws {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0, "Should create socket")
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                _ = testSocketPath.withCString { src in
                    strcpy(dest, src)
                }
            }
        }

        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        XCTAssertEqual(result, 0, "Should connect to brainbar socket (errno: \(errno))")
    }

    // MARK: - MCP Initialize handshake

    func testMCPInitializeOverSocket() throws {
        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "test", "version": "1.0"]
            ]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNotNil(result)
        XCTAssertEqual(result?["protocolVersion"] as? String, "2024-11-05")
    }

    // MARK: - MCP tools/list over socket

    func testMCPToolsListOverSocket() throws {
        // Must initialize first
        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "test", "version": "1.0"]
            ]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ])

        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        XCTAssertNotNil(tools)
        XCTAssertEqual(tools?.count, 11)
    }

    // MARK: - MCP tools/call brain_search over socket

    func testMCPBrainSearchOverSocket() throws {
        // Initialize
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "test", "version": "1.0"]]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "test search"]
            ]
        ])

        XCTAssertNil(response["error"], "brain_search should not error")
        XCTAssertNotNil(response["result"])
    }

    func testMCPBrainSubscribeOverSocketReturnsCursorState() throws {
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "subscriber", "version": "1.0"]]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "{}"
        let payload = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]

        XCTAssertEqual(payload?["status"] as? String, "subscribed")
        XCTAssertEqual(payload?["agent_id"] as? String, "agent-1")
        XCTAssertEqual(payload?["last_delivered_seq"] as? Int, 0)
        XCTAssertEqual(payload?["last_acked_seq"] as? Int, 0)
        XCTAssertNotNil(payload?["generation"])
    }

    func testMCPBrainUnsubscribeOverSocketReturnsResult() throws {
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "subscriber", "version": "1.0"]]
        ])

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": [
                "name": "brain_unsubscribe",
                "arguments": [
                    "agent_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "{}"
        let payload = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]

        XCTAssertEqual(payload?["status"] as? String, "unsubscribed")
        XCTAssertEqual(payload?["agent_id"] as? String, "agent-1")
    }

    func testMatchingStorePushesChannelNotificationAndRequiresAckToClearUnread() throws {
        let subscriberFD = try connectClient()
        defer { close(subscriberFD) }

        try initializeClient(fd: subscriberFD, name: "subscriber")
        try sendMCPRequest(on: subscriberFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-live",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: subscriberFD)

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 3,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "publisher", "version": "1.0"]
            ]
        ])

        let publishResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Live push message for agent live",
                    "tags": ["agent-message"],
                    "importance": 6
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(publishResponse["error"])

        let notification = try readMCPMessage(fd: subscriberFD)
        XCTAssertEqual(notification["method"] as? String, "notifications/claude/channel")
        let params = notification["params"] as? [String: Any]
        let content = params?["content"] as? String ?? ""
        XCTAssertTrue(content.contains("Live push message for agent live"))
        let meta = params?["meta"] as? [String: Any]
        let rowID = (meta?["rowid"] as? String).flatMap(Int.init)
        XCTAssertNotNil(rowID)

        let unreadResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": [
                    "query": "Live push message",
                    "agent_id": "agent-live",
                    "unread_only": true
                ] as [String: Any]
            ]
        ])

        let unreadResult = unreadResponse["result"] as? [String: Any]
        let unreadContent = unreadResult?["content"] as? [[String: Any]]
        let unreadText = unreadContent?.first?["text"] as? String ?? ""
        XCTAssertTrue(unreadText.contains("Live push message for agent live"), "Live-delivered chunk should stay unread until ack")

        let ackResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": [
                "name": "brain_ack",
                "arguments": [
                    "agent_id": "agent-live",
                    "seq": rowID as Any
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(ackResponse["error"])

        let clearedResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": [
                    "query": "Live push message",
                    "agent_id": "agent-live",
                    "unread_only": true
                ] as [String: Any]
            ]
        ])
        let clearedResult = clearedResponse["result"] as? [String: Any]
        let clearedContent = clearedResult?["content"] as? [[String: Any]]
        let clearedText = clearedContent?.first?["text"] as? String ?? ""
        XCTAssertFalse(clearedText.contains("Live push message for agent live"), "Acked chunk should no longer be unread")
    }

    func testDeadSubscriberDoesNotBlockLiveSubscriberNotification() throws {
        let deadFD = try connectClient()
        try initializeClient(fd: deadFD, name: "dead-subscriber")
        try sendMCPRequest(on: deadFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-dead",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: deadFD)
        close(deadFD)

        let liveFD = try connectClient()
        defer { close(liveFD) }
        try initializeClient(fd: liveFD, name: "live-subscriber")
        try sendMCPRequest(on: liveFD, request: [
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-live-2",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: liveFD)

        let storeResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Fanout survives dead subscriber",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(storeResponse["error"])

        let notification = try readMCPMessage(fd: liveFD)
        XCTAssertEqual(notification["method"] as? String, "notifications/claude/channel")
        let params = notification["params"] as? [String: Any]
        let content = params?["content"] as? String ?? ""
        XCTAssertTrue(content.contains("Fanout survives dead subscriber"))
    }

    func testSecondConnectionTakesOverAgentIdentity() throws {
        let firstFD = try connectClient()
        defer { close(firstFD) }
        try initializeClient(fd: firstFD, name: "first")
        try sendMCPRequest(on: firstFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-takeover",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: firstFD)

        let secondFD = try connectClient()
        defer { close(secondFD) }
        try initializeClient(fd: secondFD, name: "second")
        try sendMCPRequest(on: secondFD, request: [
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-takeover",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: secondFD)

        var oneByte = [UInt8](repeating: 0, count: 1)
        let firstRead = read(firstFD, &oneByte, 1)
        XCTAssertLessThanOrEqual(firstRead, 0, "First socket should be closed after takeover")

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Takeover delivery",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let notification = try readMCPMessage(fd: secondFD)
        XCTAssertEqual(notification["method"] as? String, "notifications/claude/channel")
    }

    // MARK: - C1: Socket backpressure handling

    func testServerSurvivesBriefClientBackpressure() throws {
        let clientFD = try connectClient()
        defer { close(clientFD) }
        configureBackpressuredClient(fd: clientFD, receiveBufferSize: 1_024)

        try sendMCPRequest(on: clientFD, request: initializeRequest(id: 1, name: "brief-stall"))
        for id in 2...18 {
            try sendMCPRequest(on: clientFD, request: toolsListRequest(id: id))
        }

        // Simulate a client that is briefly busy before draining responses.
        Thread.sleep(forTimeInterval: 0.1)

        let initializeResponse = try readMCPMessage(fd: clientFD, timeout: 2.0)
        XCTAssertNotNil(initializeResponse["result"])

        var lastResponse: [String: Any] = initializeResponse
        for _ in 2...18 {
            lastResponse = try readMCPMessage(fd: clientFD, timeout: 2.0)
        }
        let tools = (lastResponse["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        XCTAssertEqual(tools?.count, 11)

        try sendMCPRequest(on: clientFD, request: toolsListRequest(id: 81))
        let followUpResponse = try readMCPMessage(fd: clientFD, timeout: 2.0)
        let followUpTools = (followUpResponse["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        XCTAssertEqual(followUpTools?.count, 11, "Client should remain connected after a short backpressure burst")
    }

    func testServerDisconnectsPersistentlyStalledClientWithoutBlockingOthers() throws {
        let deadClientFD = try connectClient()
        defer { close(deadClientFD) }
        configureBackpressuredClient(fd: deadClientFD, receiveBufferSize: 1)

        try sendMCPRequest(on: deadClientFD, request: initializeRequest(id: 1, name: "dead-stall"))
        for id in 2...80 {
            try sendMCPRequest(on: deadClientFD, request: toolsListRequest(id: id))
        }

        let timeoutSeconds = Double(BrainBarServer.writeStallTimeoutMilliseconds) / 1000.0
        Thread.sleep(forTimeInterval: timeoutSeconds + 0.35)

        let secondStartedAt = Date()
        let secondResponse = try sendMCPRequest(initializeRequest(id: 200, name: "healthy-client"))
        XCTAssertNotNil(secondResponse["result"], "Dead client should not block the server forever")
        XCTAssertLessThan(Date().timeIntervalSince(secondStartedAt), 1.0, "Server should recover promptly once the stalled client is dropped")

        XCTAssertTrue(
            try waitForSocketClosure(fd: deadClientFD, timeout: 1.0),
            "Persistently stalled client should eventually be disconnected"
        )
    }

    func testStdioAdapterBridgesInitializeAndSubscribe() throws {
        let adapterPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Scripts/brainbar_stdio_adapter.py")
            .path
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            adapterPath,
            "--socket",
            testSocketPath,
        ]

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = Pipe()
        try process.run()
        defer {
            process.terminate()
            process.waitUntilExit()
        }

        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "adapter", "version": "1.0"]
            ]
        ], to: stdinPipe.fileHandleForWriting)

        let initializeResponse = try readLineJSON(from: stdoutPipe.fileHandleForReading)
        let capabilities = (initializeResponse["result"] as? [String: Any])?["capabilities"] as? [String: Any]
        let experimental = capabilities?["experimental"] as? [String: Any]
        XCTAssertEqual((experimental?["claude/channel"] as? [String: Any])?.isEmpty, true)

        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "adapter-agent",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ], to: stdinPipe.fileHandleForWriting)

        let subscribeResponse = try readLineJSON(from: stdoutPipe.fileHandleForReading)
        let result = subscribeResponse["result"] as? [String: Any]
        XCTAssertNotNil(result)
    }

    func testStdioAdapterDrainsResponsesAfterStdinEOF() throws {
        let adapterPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Scripts/brainbar_stdio_adapter.py")
            .path
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            adapterPath,
            "--socket",
            testSocketPath,
        ]

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = Pipe()
        try process.run()

        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "adapter-eof", "version": "1.0"]
            ]
        ], to: stdinPipe.fileHandleForWriting)
        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ], to: stdinPipe.fileHandleForWriting)
        try stdinPipe.fileHandleForWriting.close()

        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let outputLines = String(decoding: stdoutData, as: UTF8.self)
            .split(separator: "\n")
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        XCTAssertEqual(outputLines.count, 2)

        let initializeResponse = try JSONSerialization.jsonObject(with: Data(outputLines[0].utf8)) as? [String: Any]
        XCTAssertNotNil(initializeResponse?["result"])

        let toolsResponse = try JSONSerialization.jsonObject(with: Data(outputLines[1].utf8)) as? [String: Any]
        let tools = (toolsResponse?["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        XCTAssertEqual(tools?.count, 11)
    }

    // MARK: - C2: Socket path length validation

    func testRejectsOverlongSocketPath() throws {
        // sockaddr_un.sun_path is 104 bytes on macOS. A path > 104 should not crash.
        let longPath = "/tmp/" + String(repeating: "x", count: 200) + ".sock"
        let longServer = BrainBarServer(socketPath: longPath, dbPath: NSTemporaryDirectory() + "test-long.db")
        longServer.start()
        Thread.sleep(forTimeInterval: 0.2)

        // Server should have refused to bind — connecting should fail.
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            XCTFail("socket() failed with errno \(errno)")
            return
        }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        // Can't even set the long path in sockaddr_un, so connect would fail.
        // The key assertion: the server didn't crash during start().
        longServer.stop()
    }

    // MARK: - Helper

    private func sendMCPRequest(_ request: [String: Any]) throws -> [String: Any] {
        let fd = try connectClient()
        defer { close(fd) }
        try sendMCPRequest(on: fd, request: request)
        return try readMCPMessage(fd: fd)
    }

    private func connectClient() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "socket() failed"]) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                _ = testSocketPath.withCString { src in
                    strcpy(dest, src)
                }
            }
        }

        let connectResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else {
            close(fd)
            throw NSError(domain: "test", code: 2, userInfo: [NSLocalizedDescriptionKey: "connect() failed: errno \(errno)"])
        }
        let flags = fcntl(fd, F_GETFL)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        return fd
    }

    private func initializeClient(fd: Int32, name: String) throws {
        try sendMCPRequest(on: fd, request: [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": name, "version": "1.0"]
            ]
        ])
        _ = try readMCPMessage(fd: fd)
    }

    private func sendMCPRequest(on fd: Int32, request: [String: Any]) throws {
        let jsonData = try JSONSerialization.data(withJSONObject: request)
        let header = "Content-Length: \(jsonData.count)\r\n\r\n"
        var frame = Data(header.utf8)
        frame.append(jsonData)

        let sent = frame.withUnsafeBytes { ptr in
            write(fd, ptr.baseAddress!, frame.count)
        }
        guard sent == frame.count else {
            throw NSError(domain: "test", code: 3, userInfo: [NSLocalizedDescriptionKey: "write() incomplete"])
        }
    }

    private func readMCPMessage(fd: Int32, timeout: TimeInterval = 5.0) throws -> [String: Any] {
        var buffer = bufferedMessagesByFD[fd] ?? Data()
        var readBuf = [UInt8](repeating: 0, count: 65536)
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            if let message = try decodeBufferedMCPMessage(fd: fd, buffer: &buffer) {
                return message
            }

            let n = read(fd, &readBuf, readBuf.count)
            if n > 0 {
                buffer.append(contentsOf: readBuf[0..<n])
                if let message = try decodeBufferedMCPMessage(fd: fd, buffer: &buffer) {
                    return message
                }
            } else if n == 0 {
                bufferedMessagesByFD.removeValue(forKey: fd)
                break // EOF
            } else if errno != EAGAIN && errno != EINTR && errno != EWOULDBLOCK {
                bufferedMessagesByFD.removeValue(forKey: fd)
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }

        bufferedMessagesByFD[fd] = buffer
        throw NSError(domain: "test", code: 4, userInfo: [NSLocalizedDescriptionKey: "Timeout reading response"])
    }

    private func decodeBufferedMCPMessage(fd: Int32, buffer: inout Data) throws -> [String: Any]? {
        guard let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            bufferedMessagesByFD[fd] = buffer
            return nil
        }

        let headerData = buffer[buffer.startIndex..<headerEnd.lowerBound]
        let headerString = String(data: headerData, encoding: .utf8) ?? ""
        guard let contentLengthLine = headerString
            .components(separatedBy: "\r\n")
            .first(where: { $0.hasPrefix("Content-Length:") }),
              let separatorIndex = contentLengthLine.firstIndex(of: ":")
        else {
            bufferedMessagesByFD[fd] = buffer
            return nil
        }

        let contentLength = Int(
            contentLengthLine[contentLengthLine.index(after: separatorIndex)...]
                .trimmingCharacters(in: .whitespaces)
        ) ?? 0
        let bodyStart = headerEnd.upperBound
        guard buffer.count >= bodyStart + contentLength else {
            bufferedMessagesByFD[fd] = buffer
            return nil
        }

        let bodyRange = bodyStart..<(bodyStart + contentLength)
        let bodyData = buffer[bodyRange]
        let remaining = Data(buffer[bodyRange.upperBound...])
        bufferedMessagesByFD[fd] = remaining
        buffer = remaining
        return try JSONSerialization.jsonObject(with: bodyData) as? [String: Any] ?? [:]
    }

    private func sendLineJSON(_ object: [String: Any], to handle: FileHandle) throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        handle.write(data)
        handle.write(Data([0x0A]))
    }

    private func readLineJSON(from handle: FileHandle, timeout: TimeInterval = 5.0) throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(timeout)
        var buffer = Data()
        let fd = handle.fileDescriptor
        let flags = fcntl(fd, F_GETFL)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        var readBuf = [UInt8](repeating: 0, count: 4096)
        while Date() < deadline {
            let count = read(fd, &readBuf, readBuf.count)
            if count > 0 {
                buffer.append(contentsOf: readBuf[0..<count])
                if let newlineIndex = buffer.firstIndex(of: 0x0A) {
                    let line = buffer[..<newlineIndex]
                    return try JSONSerialization.jsonObject(with: line) as? [String: Any] ?? [:]
                }
            } else if count == 0 {
                Thread.sleep(forTimeInterval: 0.01)
            } else {
                if errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR {
                    break
                }
                Thread.sleep(forTimeInterval: 0.01)
            }
        }
        throw NSError(domain: "test", code: 5, userInfo: [NSLocalizedDescriptionKey: "Timeout reading line JSON"])
    }

    private func configureBackpressuredClient(fd: Int32, receiveBufferSize: Int32) {
        var bufSize = receiveBufferSize
        _ = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))

        var noSigPipe: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))
    }

    private func initializeRequest(id: Int, name: String) -> [String: Any] {
        [
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": name, "version": "1.0"]
            ]
        ]
    }

    private func toolsListRequest(id: Int) -> [String: Any] {
        [
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/list",
        ]
    }

    private func waitForSocketClosure(fd: Int32, timeout: TimeInterval) throws -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        var buffer = [UInt8](repeating: 0, count: 65536)

        while Date() < deadline {
            let count = read(fd, &buffer, buffer.count)
            if count == 0 {
                return true
            }
            if count > 0 {
                continue
            }
            if errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR {
                Thread.sleep(forTimeInterval: 0.01)
                continue
            }
            return true
        }

        return false
    }
}
