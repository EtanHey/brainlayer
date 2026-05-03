import XCTest
import SQLite3
@testable import BrainBar

final class BrainBarStartupRecoveryTests: XCTestCase {
    private var server: BrainBarServer?
    private var socketPath: String?
    private var tempDirectory: URL?

    override func tearDown() {
        server?.stop()
        server = nil
        if let tempDirectory {
            try? FileManager.default.removeItem(at: tempDirectory)
        }
        tempDirectory = nil
        socketPath = nil
        super.tearDown()
    }

    func testServerRecoversAfterStartupMigrationLockContention() throws {
        let tempDirectory = makeTempTestDirectory()
        self.tempDirectory = tempDirectory

        let dbPath = tempDirectory.appendingPathComponent("brainbar.db").path
        let socketPath = tempDirectory.appendingPathComponent("brainbar.sock").path
        self.socketPath = socketPath

        let seededDB = BrainDatabase(path: dbPath)
        XCTAssertTrue(seededDB.isOpen)
        seededDB.close()

        let lockDB = try openSQLiteConnection(path: dbPath)
        defer { sqlite3_close(lockDB) }
        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)

        let databaseReady = expectation(description: "database ready after retry")
        let releaseLock = expectation(description: "release startup lock")
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) {
            sqlite3_exec(lockDB, "COMMIT", nil, nil, nil)
            releaseLock.fulfill()
        }

        let server = BrainBarServer(
            socketPath: socketPath,
            dbPath: dbPath,
            databaseRecoveryPolicy: .init(
                busyTimeoutMillis: 50,
                initialRetryDelayMillis: 25,
                maximumRetryDelayMillis: 50
            )
        )
        server.onDatabaseReady = { (_: BrainDatabase) in
            databaseReady.fulfill()
        }
        self.server = server

        server.start()

        wait(for: [releaseLock, databaseReady], timeout: 2.0)

        _ = try sendMCPRequest(
            to: socketPath,
            request: [
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": [
                    "protocolVersion": "2024-11-05",
                    "capabilities": [:] as [String: Any],
                    "clientInfo": ["name": "startup-recovery-test", "version": "1.0"]
                ]
            ]
        )

        let response = try sendMCPRequest(
            to: socketPath,
            request: [
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": [
                    "name": "brain_search",
                    "arguments": ["query": "startup contention"]
                ] as [String: Any]
            ]
        )

        XCTAssertNil(response["error"], "brain_search should recover once the startup lock clears")
        XCTAssertNotNil(response["result"])
    }
}

private func makeTempTestDirectory() -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func openSQLiteConnection(path: String) throws -> OpaquePointer {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        let message = db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
        if let db {
            sqlite3_close(db)
        }
        throw NSError(domain: "BrainBarStartupRecoveryTests", code: Int(rc), userInfo: [
            NSLocalizedDescriptionKey: "Failed to open sqlite connection: \(message)"
        ])
    }
    return db
}

private func sendMCPRequest(to socketPath: String, request: [String: Any]) throws -> [String: Any] {
    let fd = try connectToSocket(path: socketPath)
    defer { close(fd) }

    let payload = try JSONSerialization.data(withJSONObject: request)
    var framed = Data("Content-Length: \(payload.count)\r\n\r\n".utf8)
    framed.append(payload)
    _ = framed.withUnsafeBytes { ptr in
        write(fd, ptr.baseAddress, ptr.count)
    }

    return try readSingleFramedMessage(from: fd)
}

private func connectToSocket(path: String) throws -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
            _ = path.withCString { src in strcpy(dest, src) }
        }
    }

    let deadline = Date().addingTimeInterval(1.0)
    while Date() < deadline {
        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if result == 0 {
            return fd
        }
        if errno != ENOENT && errno != ECONNREFUSED {
            break
        }
        Thread.sleep(forTimeInterval: 0.01)
    }

    close(fd)
    throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
}

private func readSingleFramedMessage(from fd: Int32, timeout: TimeInterval = 2.0) throws -> [String: Any] {
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

    throw NSError(domain: "BrainBarStartupRecoveryTests", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "Timed out waiting for MCP response"
    ])
}
