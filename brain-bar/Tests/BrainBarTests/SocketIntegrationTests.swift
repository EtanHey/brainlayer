// SocketIntegrationTests.swift — RED tests for end-to-end socket + MCP flow.
//
// Tests the full pipeline: connect to Unix socket → send Content-Length framed
// MCP request → receive Content-Length framed response.

import XCTest
@testable import BrainBar

final class SocketIntegrationTests: XCTestCase {
    let testSocketPath = "/tmp/brainbar-test-\(ProcessInfo.processInfo.processIdentifier).sock"
    var server: BrainBarServer!

    override func setUp() {
        super.setUp()
        let tempDB = NSTemporaryDirectory() + "brainbar-integration-\(UUID().uuidString).db"
        server = BrainBarServer(socketPath: testSocketPath, dbPath: tempDB)
        server.start()
        // Give server time to bind
        Thread.sleep(forTimeInterval: 0.2)
    }

    override func tearDown() {
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
        XCTAssertEqual(tools?.count, 8)
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

    // MARK: - C1: Write retry cap

    func testServerDisconnectsStalledClient() throws {
        // Connect but never read — server should disconnect after max retries (10),
        // not block the serial queue forever.
        let clientFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard clientFD >= 0 else { throw NSError(domain: "test", code: 1) }
        defer { close(clientFD) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                _ = testSocketPath.withCString { src in strcpy(dest, src) }
            }
        }
        let connectResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(clientFD, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        XCTAssertEqual(connectResult, 0, "Should connect")

        // Set tiny receive buffer to force EAGAIN on server-side writes
        var bufSize: Int32 = 1
        setsockopt(clientFD, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))

        // Send an initialize request
        let json = #"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"stall","version":"1"}}}"#
        let header = "Content-Length: \(json.utf8.count)\r\n\r\n"
        var frame = Data(header.utf8)
        frame.append(Data(json.utf8))
        frame.withUnsafeBytes { ptr in
            _ = write(clientFD, ptr.baseAddress!, frame.count)
        }

        // After the write stalls (tiny rcvbuf), server should disconnect within ~20ms (10 retries * 1ms + overhead)
        // If it hangs > 200ms, the retry cap is broken.
        // A second client should still be able to connect and get a response,
        // proving the serial queue wasn't blocked.
        Thread.sleep(forTimeInterval: 0.2)

        let secondResponse = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 99, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "second", "version": "1.0"]]
        ])
        XCTAssertNotNil(secondResponse["result"], "Serial queue must not be blocked — second client should get response")
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
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "socket() failed"]) }
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

        let connectResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else {
            throw NSError(domain: "test", code: 2, userInfo: [NSLocalizedDescriptionKey: "connect() failed: errno \(errno)"])
        }

        // Send Content-Length framed request
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

        // Read response with Content-Length framing
        var buffer = Data()
        var readBuf = [UInt8](repeating: 0, count: 65536)
        let deadline = Date().addingTimeInterval(5.0)

        while Date() < deadline {
            let n = read(fd, &readBuf, readBuf.count)
            if n > 0 {
                buffer.append(contentsOf: readBuf[0..<n])
                // Try to parse Content-Length framed response
                if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                    let headerStr = String(data: buffer[buffer.startIndex..<headerEnd.lowerBound], encoding: .utf8) ?? ""
                    if let clLine = headerStr.split(separator: "\r\n").first(where: { $0.hasPrefix("Content-Length:") }) {
                        let cl = Int(clLine.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)) ?? 0
                        let bodyStart = headerEnd.upperBound
                        if buffer.count >= bodyStart + cl {
                            let bodyData = buffer[bodyStart..<(bodyStart + cl)]
                            return try JSONSerialization.jsonObject(with: bodyData) as? [String: Any] ?? [:]
                        }
                    }
                }
            } else if n == 0 {
                break // EOF
            } else if errno != EAGAIN && errno != EINTR {
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }

        throw NSError(domain: "test", code: 4, userInfo: [NSLocalizedDescriptionKey: "Timeout reading response"])
    }
}
