import XCTest
import SQLite3
@testable import BrainBar

final class BrainBarReliabilityTests: XCTestCase {
    private var servers: [BrainBarServer] = []
    private var tempDirectories: [URL] = []

    override func tearDown() {
        for server in servers {
            server.stop()
        }
        servers.removeAll()
        for directory in tempDirectories {
            try? FileManager.default.removeItem(at: directory)
        }
        tempDirectories.removeAll()
        super.tearDown()
    }

    func testSingleInstanceLockRejectsSecondOwner() throws {
        let directory = makeReliabilityTempDirectory()
        let lockPath = directory.appendingPathComponent("brainbar.sock.lock").path

        let first = try BrainBarInstanceLock.acquire(lockPath: lockPath)
        defer { first.release() }

        XCTAssertThrowsError(try BrainBarInstanceLock.acquire(lockPath: lockPath)) { error in
            guard case BrainBarInstanceLock.AcquireError.alreadyRunning = error else {
                XCTFail("Expected alreadyRunning, got \(error)")
                return
            }
        }
    }

    func testInitializeReturnsImmediatelyWhileStartupDatabaseIsWriteLocked() throws {
        let directory = makeReliabilityTempDirectory()
        let dbPath = directory.appendingPathComponent("brainbar.db").path
        let socketPath = directory.appendingPathComponent("brainbar.sock").path

        let seededDB = BrainDatabase(path: dbPath)
        XCTAssertTrue(seededDB.isOpen)
        seededDB.close()

        let lockDB = try openReliabilitySQLiteConnection(path: dbPath)
        defer { sqlite3_close(lockDB) }
        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)
        var lockCommitted = false
        defer {
            if !lockCommitted {
                sqlite3_exec(lockDB, "COMMIT", nil, nil, nil)
            }
        }

        let databaseReady = expectation(description: "database ready after initialize lock clears")
        let server = BrainBarServer(
            socketPath: socketPath,
            dbPath: dbPath,
            databaseRecoveryPolicy: .init(
                busyTimeoutMillis: 1_000,
                initialRetryDelayMillis: 25,
                maximumRetryDelayMillis: 50
            )
        )
        server.onDatabaseReady = { _ in
            databaseReady.fulfill()
        }
        servers.append(server)
        server.start()

        let started = Date()
        let response = try sendReliabilityMCPRequest(
            to: socketPath,
            request: [
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": [
                    "protocolVersion": "2024-11-05",
                    "capabilities": [:] as [String: Any],
                    "clientInfo": ["name": "reliability-test", "version": "1.0"]
                ]
            ],
            timeout: 0.50
        )

        XCTAssertLessThan(Date().timeIntervalSince(started), 0.35)
        XCTAssertEqual((response["result"] as? [String: Any])?["protocolVersion"] as? String, "2024-11-05")
        XCTAssertEqual(sqlite3_exec(lockDB, "COMMIT", nil, nil, nil), SQLITE_OK)
        lockCommitted = true
        wait(for: [databaseReady], timeout: 1.0)
    }

    func testBrainStoreQueuesWithinBudgetWhenDatabaseWriteLockIsHeld() throws {
        let directory = makeReliabilityTempDirectory()
        let dbPath = directory.appendingPathComponent("brainbar.db").path
        let queuePath = directory.appendingPathComponent("pending-stores.jsonl").path
        setenv("BRAINBAR_PENDING_STORES_PATH", queuePath, 1)
        defer { unsetenv("BRAINBAR_PENDING_STORES_PATH") }

        let db = BrainDatabase(path: dbPath)
        XCTAssertTrue(db.isOpen)
        defer { db.close() }

        let router = MCPRouter()
        router.setDatabase(db)

        let lockDB = try openReliabilitySQLiteConnection(path: dbPath)
        defer { sqlite3_close(lockDB) }
        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)
        defer { sqlite3_exec(lockDB, "COMMIT", nil, nil, nil) }

        let started = Date()
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Queued while another writer holds the database lock",
                    "tags": ["reliability-test"],
                    "importance": 8
                ] as [String: Any]
            ] as [String: Any]
        ])

        XCTAssertLessThan(Date().timeIntervalSince(started), 0.20)
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        XCTAssertEqual(result["queued"] as? Bool, true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: queuePath))
    }

    func testQueuedWriteBurstDoesNotTakeReadPathDown() throws {
        let directory = makeReliabilityTempDirectory()
        let dbPath = directory.appendingPathComponent("brainbar.db").path
        let queuePath = directory.appendingPathComponent("pending-stores.jsonl").path
        setenv("BRAINBAR_PENDING_STORES_PATH", queuePath, 1)
        defer { unsetenv("BRAINBAR_PENDING_STORES_PATH") }

        let db = BrainDatabase(path: dbPath)
        XCTAssertTrue(db.isOpen)
        defer { db.close() }
        try db.insertChunk(
            id: "read-survives-write-burst",
            content: "BrainBar read path should survive queued write bursts",
            sessionId: "s1",
            project: "reliability",
            contentType: "assistant_text",
            importance: 8
        )

        let router = MCPRouter()
        router.setDatabase(db)

        let lockDB = try openReliabilitySQLiteConnection(path: dbPath)
        defer { sqlite3_close(lockDB) }
        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)
        defer { sqlite3_exec(lockDB, "COMMIT", nil, nil, nil) }

        for index in 0..<5 {
            let response = router.handle([
                "jsonrpc": "2.0",
                "id": index,
                "method": "tools/call",
                "params": [
                    "name": "brain_store",
                    "arguments": [
                        "content": "Queued burst item \(index)",
                        "tags": ["r02-regression", "burst"],
                        "importance": 7
                    ] as [String: Any]
                ] as [String: Any]
            ])
            let result = try XCTUnwrap(response["result"] as? [String: Any])
            XCTAssertEqual(result["queued"] as? Bool, true)
        }

        let started = Date()
        let searchResponse = router.handle([
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "read path survive", "num_results": 5]
            ] as [String: Any]
        ])

        XCTAssertLessThan(Date().timeIntervalSince(started), 0.20)
        let result = try XCTUnwrap(searchResponse["result"] as? [String: Any])
        XCTAssertNil(result["isError"])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = content.compactMap { $0["text"] as? String }.joined(separator: "\n")
        XCTAssertTrue(text.contains("read-survives-write-burst") || text.contains("read path should survive"))
    }

    private func makeReliabilityTempDirectory() -> URL {
        let shortID = UUID().uuidString.prefix(8)
        let directory = URL(fileURLWithPath: "/private/tmp/br-\(shortID)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        tempDirectories.append(directory)
        return directory
    }
}

private func openReliabilitySQLiteConnection(path: String) throws -> OpaquePointer {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        let message = db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
        if let db {
            sqlite3_close(db)
        }
        throw NSError(domain: "BrainBarReliabilityTests", code: Int(rc), userInfo: [
            NSLocalizedDescriptionKey: "Failed to open sqlite connection: \(message)"
        ])
    }
    return db
}

private func sendReliabilityMCPRequest(
    to socketPath: String,
    request: [String: Any],
    timeout: TimeInterval
) throws -> [String: Any] {
    let fd = try connectReliabilitySocket(path: socketPath, timeout: timeout)
    defer { close(fd) }

    let payload = try JSONSerialization.data(withJSONObject: request)
    var framed = Data("Content-Length: \(payload.count)\r\n\r\n".utf8)
    framed.append(payload)
    _ = framed.withUnsafeBytes { ptr in
        write(fd, ptr.baseAddress, ptr.count)
    }

    return try readReliabilityFramedMessage(from: fd, timeout: timeout)
}

private func connectReliabilitySocket(path: String, timeout: TimeInterval) throws -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = path.utf8CString
    let pathCapacity = MemoryLayout.size(ofValue: addr.sun_path)
    guard pathBytes.count <= pathCapacity else {
        close(fd)
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(ENAMETOOLONG), userInfo: [
            NSLocalizedDescriptionKey: "Socket path too long (\(pathBytes.count) > \(pathCapacity)): \(path)"
        ])
    }
    withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        ptr.withMemoryRebound(to: CChar.self, capacity: pathCapacity) { dest in
            pathBytes.withUnsafeBufferPointer { src in
                _ = memcpy(dest, src.baseAddress!, src.count)
            }
        }
    }

    let deadline = Date().addingTimeInterval(timeout)
    var lastErrno = ENOENT
    while Date() < deadline {
        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if result == 0 {
            return fd
        }
        lastErrno = errno
        if errno != ENOENT && errno != ECONNREFUSED {
            break
        }
        Thread.sleep(forTimeInterval: 0.01)
    }

    close(fd)
    throw NSError(domain: NSPOSIXErrorDomain, code: Int(lastErrno))
}

private func readReliabilityFramedMessage(from fd: Int32, timeout: TimeInterval) throws -> [String: Any] {
    let deadline = Date().addingTimeInterval(timeout)
    var buffer = Data()
    var readBuf = [UInt8](repeating: 0, count: 4096)

    while Date() < deadline {
        let count = read(fd, &readBuf, readBuf.count)
        if count > 0 {
            buffer.append(contentsOf: readBuf[0..<count])
            if let headerRange = buffer.range(of: Data("\r\n\r\n".utf8)),
               let header = String(data: buffer[..<headerRange.lowerBound], encoding: .utf8),
               let contentLengthLine = header
                .split(separator: "\r\n")
                .first(where: { $0.lowercased().hasPrefix("content-length:") }),
               let contentLength = Int(contentLengthLine.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)),
               buffer.count >= headerRange.upperBound + contentLength {
                let body = buffer[headerRange.upperBound..<(headerRange.upperBound + contentLength)]
                return try JSONSerialization.jsonObject(with: body) as? [String: Any] ?? [:]
            }
        } else if count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            break
        }
        Thread.sleep(forTimeInterval: 0.01)
    }

    throw NSError(domain: "BrainBarReliabilityTests", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "Timed out waiting for MCP response"
    ])
}
